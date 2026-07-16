#!/usr/bin/env python3
"""LiveKit AgentSession worker for the bank receptionist agent (Phase 0/1 of the LiveKit
migration). Reuses banking.py/db.py/rag.py/convo_log.py and config/bank_config.json unchanged
from the original Pipeline (see the migration plan) — this file replaces src/server.py's custom
WebSocket/VAD orchestration with LiveKit's AgentSession, and wraps the same faster-whisper model
and Chatterbox microservice as custom STT/TTS plugins (whisper_stt.py, chatterbox_tts.py).
"""
import json
import os
import re
from datetime import datetime

import aiohttp
from faster_whisper import WhisperModel

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.voice.agent import Agent as AgentCls
from livekit.plugins import openai, silero

import banking
import convo_log
import db
import rag
from chatterbox_tts import ChatterboxTTS
from whisper_stt import WhisperSTT

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen2.5-14b-awq")
CHATTERBOX_URL = os.environ.get("CHATTERBOX_URL", "http://localhost:8766/synthesize")
SEARXNG_URL = "http://localhost:1234/search"
WEB_SEARCH_RESULT_COUNT = 4
RAG_INJECT_TOP_K = 4

DOMAIN_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "bank_config.json")
MEMORY_FILE = os.path.join(PROJECT_ROOT, "data", "memory.json")
MAX_MEMORIES = 60

# Same suppression predicate the original server.py's guardrail used (banking.py, unchanged).
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

CARD_OUTCOME_LINES = {
    "blocked": "Your card has been blocked. Is there anything else I can help you with?",
    "declined": "I'm sorry, those details didn't match our records. Would you like to try once more?",
    "handed_off": "I couldn't verify your identity, so I've arranged for one of our representatives to "
                  "call you back within one business day.",
}
UNVERIFIED_BLOCK_FALLBACK = (
    "I'm sorry, I can't confirm that yet. To block your card I first need to verify your identity — "
    "could you tell me the last four digits of the card?"
)

DB_CONN = db.connect()
db.init_schema(DB_CONN)


# ── Domain config / long-term memory (ported unchanged from src/server.py) ──────────────────
def load_domain_config() -> dict:
    try:
        with open(DOMAIN_CONFIG_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"  Warning: couldn't load {DOMAIN_CONFIG_FILE}: {e}")
        return {}


def format_domain_block(cfg: dict) -> str:
    if not cfg:
        return ""
    lines = [f"You are the virtual receptionist for {cfg.get('bank_name', 'the bank')}."]
    if cfg.get("tagline"):
        lines.append(cfg["tagline"] + ".")
    hours = cfg.get("general_hours") or {}
    if hours:
        lines.append(
            "Banking hours: Monday to Thursday " + hours.get("monday_to_thursday", "n/a") +
            "; Friday " + hours.get("friday", "n/a") +
            "; Saturday " + hours.get("saturday", "n/a") +
            "; Sunday: " + hours.get("sunday", "n/a") + "."
        )
        if hours.get("note"):
            lines.append(hours["note"])
    branches = cfg.get("branches") or []
    if branches:
        lines.append("Branches you know about:")
        for b in branches:
            phone = f", phone {b['phone']}" if b.get("phone") else ""
            lines.append(f"- {b['name']}: {b['address']}{phone}")
    services = cfg.get("services") or []
    if services:
        lines.append("Services offered:")
        for s in services:
            lines.append(f"- {s['name']}: {s['description']}")
    if cfg.get("customer_care_number"):
        lines.append(f"Customer care helpline: {cfg['customer_care_number']}.")
    if cfg.get("website"):
        lines.append(f"Website: {cfg['website']}.")
    return "\n".join(lines) + "\n"


def load_memories() -> list[str]:
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        return [str(x) for x in data][-MAX_MEMORIES:]
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return []


def save_memory(fact: str) -> bool:
    fact = fact.strip()
    if not fact:
        return False
    existing = load_memories()
    if any(fact.lower() == e.lower() for e in existing):
        return False
    existing.append(fact)
    try:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, "w") as f:
            json.dump(existing[-MAX_MEMORIES:], f, indent=2)
    except OSError as e:
        print(f"  Failed to save memory: {e}")
        return False
    print(f"  Remembered: {fact}")
    return True


def build_instructions() -> str:
    """Ported from build_system_prompt() in src/server.py — same persona/scope/tool-usage rules,
    minus per-turn freshness (date/memory reloaded once at Agent construction for this spike;
    Phase 1 should re-render this per turn the way the original re-read it every prompt build)."""
    now = datetime.now().astimezone()
    now_str = now.strftime("%A, %B %d, %Y, %I:%M %p %Z")
    domain_block = format_domain_block(load_domain_config())
    memories = load_memories()
    memory_block = (
        "Here's what you remember about this caller from past conversations — use it naturally, "
        "don't recite it back:\n" + "\n".join(f"- {m}" for m in memories) + "\n"
    ) if memories else ""

    return (
        domain_block + memory_block +
        f"Right now it is {now_str}. Always use this as the true current date/time. "
        "You're speaking with a caller over the phone, not writing a document — talk like a real "
        "receptionist: warm, polite, and professional, but natural. Keep answers short — usually "
        "one to two sentences. No markdown, bullets, or code blocks — only plain speakable sentences. "
        "If the caller is speaking Urdu, Hindi, or any other language, reply using Roman "
        "transliteration in the Latin alphabet, never native script. "
        "Stay strictly in character: you only handle questions about this bank. Beyond answering "
        "questions, the only actions you can take are blocking a lost or stolen card after identity "
        "verification, and logging a callback request from a human representative. You cannot check "
        "balances or make transactions. If asked something unrelated to banking, redirect immediately. "
        "Never make up a branch, phone number, rate, or policy that isn't given to you. "
        "For questions about products, accounts, fees, rates, or policies, you'll be given reference "
        "material from the bank's documents just before the caller's question, when something relevant "
        "is found — answer ONLY from that material (plus the branch/hours/service info above), never "
        "from your own general knowledge. If nothing relevant was provided, say you don't have that "
        "specific information on file rather than guessing. Never read out a document name or heading. "
        "If a caller wants to block a lost or stolen card, collect three verification details ONE at "
        "a time: the last four digits of the card, their mother's maiden name, then their date of "
        "birth. Never judge whether an answer is right or wrong yourself — the system verifies them. "
        "This is critical: you cannot block a card or verify anyone by talking. The moment you have "
        "all three details, call the block_card tool and do NOT write any reply in that same turn — "
        "no confirmation, no summary, nothing but the tool call. The system will speak the outcome "
        "for you. Never state or imply a card is blocked or an identity verified yourself. "
        "You must NEVER give out the bank's phone number as a way to reach a human representative — "
        "call request_human_handoff instead when a caller asks to speak to a person."
    )


class BankReceptionistAgent(Agent):
    def __init__(self):
        super().__init__(instructions=build_instructions())

    # ── Always-on RAG: inject retrieved chunks right before the caller's question,
    # never as a tool call the model has to decide to invoke (src/rag.py, unchanged). ──────────
    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        query_text = new_message.text_content or ""
        history = [
            {"role": m.role, "content": m.text_content or ""} for m in turn_ctx.messages()
        ] + [{"role": "user", "content": query_text}]

        retrieval_query = rag.build_retrieval_query(history)
        queries = [query_text] if retrieval_query == query_text else [query_text, retrieval_query]
        retrieval = rag.search_docs_multi(queries, k=RAG_INJECT_TOP_K)

        print(f"  [rag] status={retrieval['status']} queries={queries}")
        if retrieval["status"] == "found":
            context_block = "\n\n".join(
                f"[{r['source']} :: {r.get('section')}]\n{r['text']}" for r in retrieval["results"]
            )
            # created_at just before new_message's own timestamp so ChatContext.insert() (which
            # sorts new_message in by created_at) places it *after* this system message, not before.
            turn_ctx.add_message(
                role="system",
                content=(
                    "Reference information from the bank's documents, relevant to the caller's "
                    "next question. Answer only from this and the bank info in your instructions; "
                    "if the specific detail isn't here, say you don't have it on file.\n\n"
                    + context_block
                ),
                created_at=new_message.created_at - 0.001,
            )

    # ── Output guardrail: never let a "card blocked / identity verified" claim reach audio
    # unless block_card actually returned "blocked" this turn (banking.py, unchanged). ─────────
    async def tts_node(self, text, model_settings: ModelSettings):
        tripped = False

        async def filtered():
            nonlocal tripped
            buf = ""
            async for chunk in text:
                buf += chunk
                while True:
                    m = SENTENCE_END.search(buf)
                    if not m:
                        break
                    sentence, buf = buf[: m.end()], buf[m.end():]
                    if banking.asserts_block_success(sentence):
                        tripped = True
                        print(f"  [guardrail] suppressed: {sentence[:70]!r}")
                        continue
                    yield sentence
            if buf.strip():
                if banking.asserts_block_success(buf):
                    tripped = True
                    print(f"  [guardrail] suppressed: {buf[:70]!r}")
                else:
                    yield buf

        async for frame in AgentCls.default.tts_node(self, filtered(), model_settings):
            yield frame

        if tripped:
            # Defense-in-depth backstop tripped with no corresponding tool result this turn —
            # speak a truthful correction. (Phase 1 TODO: only fire this when block_card wasn't
            # also called this turn, to avoid double-speaking alongside the tool's own outcome line.)
            self.session.say(UNVERIFIED_BLOCK_FALLBACK, add_to_chat_ctx=True)

    # ── Tools ────────────────────────────────────────────────────────────────────────────────
    @function_tool
    async def web_search(self, ctx: RunContext, query: str) -> str:
        """Search the web, but ONLY for genuinely bank/finance-relevant information that isn't
        already in your bank data — e.g. a current currency exchange rate. Do NOT use this for
        anything unrelated to banking."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    SEARXNG_URL, params={"q": query, "format": "json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
        except Exception as e:
            return f"Web search failed: {e}"
        results = data.get("results", [])[:WEB_SEARCH_RESULT_COUNT]
        if not results:
            return f"No web results found for '{query}'."
        lines = [f"Search results for '{query}':"]
        for r in results:
            lines.append(f"- {(r.get('title') or '').strip()}: {(r.get('content') or '').strip()}")
        return "\n".join(lines)

    @function_tool
    async def remember(self, ctx: RunContext, fact: str) -> str:
        """Save a short fact about this caller to long-term memory so you'll know it next time
        they call — their name, or anything they explicitly ask you to remember."""
        return "Saved to memory." if save_memory(fact) else "Already knew that."

    @function_tool
    async def block_card(
        self, ctx: RunContext, card_last4: str, mother_maiden_name: str, dob: str
    ) -> str:
        """Block a customer's lost or stolen card. Only call this once you have collected ALL
        THREE verification details: the last 4 digits of the card, mother's maiden name, and
        date of birth. Pass exactly what the customer said — the system verifies them."""
        outcome = banking.verify_and_block_card(
            DB_CONN, ctx.userdata,
            {"card_last4": card_last4, "mother_maiden_name": mother_maiden_name, "dob": dob},
        )
        status = outcome["status"]
        print(f"  block_card -> {status} (attempts={ctx.userdata.get('failed_card_attempts', 0)})")
        # Spoken deterministically from the tool result, never phrased by the model.
        ctx.session.say(CARD_OUTCOME_LINES.get(status, "I'm sorry, I can't confirm that."))
        return json.dumps(outcome)

    @function_tool
    async def request_human_handoff(
        self, ctx: RunContext, reason: str, customer_id: str | None = None
    ) -> str:
        """Log a callback request from a human representative. Use this when the customer
        explicitly asks to speak to a human, a person, or a representative. Do NOT give out the
        bank's phone number as a way to reach a human — call this tool instead."""
        outcome = banking.queue_handoff(DB_CONN, {"reason": reason, "customer_id": customer_id})
        print(f"  request_human_handoff -> ticket {outcome.get('ticket_id')}")
        return json.dumps(outcome)


# ── Worker entrypoint ─────────────────────────────────────────────────────────
def prewarm(proc: JobProcess):
    print("Loading Silero VAD...")
    proc.userdata["vad"] = silero.VAD.load()
    print("Loading faster-whisper (large-v3, GPU int8)...")
    proc.userdata["whisper_model"] = WhisperModel(
        "large-v3", compute_type="int8_float16", device="cuda", device_index=0
    )
    print("Loading RAG embedding index...")
    rag.init(DB_CONN)
    rag.warmup()
    print("Prewarm complete.")


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=WhisperSTT(model=ctx.proc.userdata["whisper_model"]),
        llm=openai.LLM(model=VLLM_MODEL, base_url=VLLM_URL, api_key="not-needed"),
        tts=ChatterboxTTS(url=CHATTERBOX_URL),
        userdata={"failed_card_attempts": 0},
    )

    await session.start(agent=BankReceptionistAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
