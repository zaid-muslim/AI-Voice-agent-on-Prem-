#!/usr/bin/env python3
"""LiveKit AgentSession worker for the bank receptionist agent (Phase 1 web-surface migration).
Reuses banking.py/db.py/rag.py/convo_log.py and config/bank_config.json unchanged from the
original Pipeline (see the migration plan) — this file replaces src/server.py's custom
WebSocket/VAD orchestration with LiveKit's AgentSession, and wraps the same faster-whisper model
and Chatterbox microservice as custom STT/TTS plugins (whisper_stt.py, chatterbox_tts.py).
"""
import asyncio
import json
import os
import re
import time
from datetime import datetime

import aiohttp
import redis.asyncio as aioredis
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    tts,
    utils,
)
# Imported by name (not the `tokenize` module) on purpose: FastStartSentenceTokenizer below has a
# method named `tokenize` (required by the base class), which would shadow the module inside the
# class body and break the `-> SentenceStream` annotations.
from livekit.agents.tokenize import SentenceStream, SentenceTokenizer, TokenData
from livekit.plugins import openai, silero

import banking
import convo_log
import db
import rag
from chatterbox_tts import ChatterboxTTS
from whisper_stt import WhisperSTT

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# These are fallback defaults only, for running this file directly (debugging, bypassing the
# orchestrator). In normal operation, src/orchestrator.py sets all of them as real environment
# variables on this process before it starts — load_dotenv() defaults to override=False, so .env's
# values never clobber what the orchestrator already set, they only apply when nothing else has.
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen2.5-14b-awq")
# STT/TTS are shared HTTP microservices (src/whisper_server.py, src/chatterbox_server.py), not
# per-process models — the job-executor process holds no GPU model of
# its own (that per-call GPU copy is what OOM'd the 2nd concurrent caller pre-Phase-1). Each is
# now a *pool* — one or more instances, comma-separated, possibly spanning multiple boxes (see
# orchestrator.py's _launch_worker, which builds this from config/models_config.json's confirmed
# entry + its extra_pool_urls). A single URL with no comma works fine too (today's box-1-only
# case). WHISPER_LANGUAGE stays a single value — same for every pool member.
CHATTERBOX_URLS = [
    u.strip() for u in os.environ.get("CHATTERBOX_URLS", "http://localhost:8766/synthesize").split(",") if u.strip()
]
WHISPER_URLS = [
    u.strip() for u in os.environ.get("WHISPER_URLS", "http://localhost:8768/transcribe").split(",") if u.strip()
]
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
# Redis already backs LiveKit itself in this stack (see docker-compose.yml) — reused here purely
# as an atomic counter for round-robin pool selection, not for anything LiveKit-related.
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
SEARXNG_URL = "http://localhost:1234/search"
WEB_SEARCH_RESULT_COUNT = 4
RAG_INJECT_TOP_K = 4

DOMAIN_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "bank_config.json")
MEMORY_FILE = os.path.join(PROJECT_ROOT, "data", "memory.json")
MAX_MEMORIES = 60

# Same suppression predicate the original server.py's guardrail used (banking.py, unchanged).
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# For the very first chunk of a reply only, also break at the earliest clause boundary
# (comma/semicolon/colon) once a small floor of characters has accumulated, so audio starts after
# a few words ("Sure,") instead of waiting for the model to finish a whole first sentence — the
# single biggest source of start latency. Ported verbatim from the original server.py, which
# benchmarked the 5-char floor against real vLLM streams + live Chatterbox (short lead-ins
# synthesize ~0.28s at 5 chars vs ~0.5-0.7s at a full sentence). After the first chunk, everything
# settles back into full-sentence chunks. Breaking mid-first-sentence is safe for the guardrail
# below: asserts_block_success() matches on word presence (\bblocked\b, etc.), not whole sentences.
FIRST_CHUNK_END = re.compile(r"(?<=[.!?,;:])\s+")
FIRST_CHUNK_MIN_CHARS = 5


class _FastStartSentenceStream(SentenceStream):
    """Streaming sentence tokenizer that emits the FIRST chunk of a reply at the earliest clause
    boundary (comma/semicolon/colon) past FIRST_CHUNK_MIN_CHARS, then full sentences after — the
    original server.py's start-latency optimization. Handing these chunks to tts.StreamAdapter
    (instead of splitting-then-synthesizing by hand in tts_node) keeps the adapter's pipelined
    synthesis, so audio stays gapless — the naive per-chunk approach stalled on each blocking
    Chatterbox call between chunks, which is what caused the noise cuts between words."""

    def __init__(self) -> None:
        super().__init__()
        self._buf = ""
        self._first_done = False
        self._seg = utils.shortuuid()

    def _emit(self, text: str) -> None:
        text = text.strip()
        if text:
            self._event_ch.send_nowait(TokenData(token=text, segment_id=self._seg))

    def _drain(self) -> None:
        if not self._first_done:
            for m in FIRST_CHUNK_END.finditer(self._buf):
                if len(self._buf[: m.start()].strip()) >= FIRST_CHUNK_MIN_CHARS:
                    self._emit(self._buf[: m.end()])
                    self._buf = self._buf[m.end():]
                    self._first_done = True
                    break
        while True:
            m = SENTENCE_END.search(self._buf)
            if not m:
                break
            self._emit(self._buf[: m.end()])
            self._buf = self._buf[m.end():]

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        if text:
            self._buf += text
            self._drain()

    def flush(self) -> None:
        self._check_not_closed()
        self._drain()
        if self._buf.strip():
            self._emit(self._buf)
        self._buf = ""

    def end_input(self) -> None:
        self.flush()
        self._event_ch.close()

    async def aclose(self) -> None:
        self._event_ch.close()


class FastStartSentenceTokenizer(SentenceTokenizer):
    """SentenceTokenizer wrapper around _FastStartSentenceStream (see it for the why)."""

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        out, first_done = [], False
        if not first_done:
            for m in FIRST_CHUNK_END.finditer(text):
                if len(text[: m.start()].strip()) >= FIRST_CHUNK_MIN_CHARS:
                    out.append(text[: m.end()].strip())
                    text = text[m.end():]
                    first_done = True
                    break
        for part in SENTENCE_END.split(text):
            if part.strip():
                out.append(part.strip())
        return out

    def stream(self, *, language: str | None = None) -> SentenceStream:
        return _FastStartSentenceStream()


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
    """Ported from build_system_prompt() in src/server.py — same persona/scope/tool-usage rules.
    Called once at Agent construction and again every turn (on_user_turn_completed calls
    update_instructions()) so date/memory freshness matches the original re-reading it on every
    prompt build."""
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
    def __init__(self, session_id: str):
        super().__init__(instructions=build_instructions())
        self._session_id = session_id
        # Set synchronously inside block_card so tts_node's guardrail can tell a genuine
        # success apart from a hallucinated one within the same turn (see tts_node below).
        self._last_block_card_status: str | None = None
        # Accumulates the in-flight turn for convo_log; flushed when the *next* turn starts
        # (or the session closes) since a turn's reply/tool-call events land as separate async
        # AgentSession events with no single synchronous "turn done" callback to log from.
        self._turn: dict | None = None
        # Lazily built on first tts_node call (needs self.session.tts, only available once the
        # session is running); reused so its metrics listener registers once, not per turn.
        self._tts_adapter: tts.StreamAdapter | None = None

    def flush_turn_log(self) -> None:
        t, self._turn = self._turn, None
        if t is None:
            return
        timings = {"total_ms": (time.monotonic() - t["t_start"]) * 1000} if t["t_start"] else None
        try:
            convo_log.log_turn(
                session_id=self._session_id,
                user_text=t["user_text"],
                retrieval=t["retrieval"],
                tool_events=t["tool_events"],
                reply=t["reply"].strip(),
                timings=timings,
            )
        except Exception as e:
            # Logging must never break a call (convo_log.py's own stated contract).
            print(f"  [convo_log] failed to log turn: {e}")

    # ── Always-on RAG: inject retrieved chunks right before the caller's question,
    # never as a tool call the model has to decide to invoke (src/rag.py, unchanged). Shared by
    # on_user_turn_completed (real speech turns) AND the text-input path (typed chat messages) —
    # AgentSession routes those through two different, non-overlapping code paths (STT-driven
    # turns call on_user_turn_completed; text_input_cb calls generate_reply() directly, which does
    # NOT go through on_user_turn_completed at all) that would otherwise silently skip RAG
    # grounding and instruction freshness for typed input. ───────────────────────────────────────
    async def prepare_turn(self, query_text: str, chat_ctx, insert_before_ts: float) -> None:
        self.flush_turn_log()  # the previous turn is done now that this new one is starting
        self._last_block_card_status = None
        self._turn = {
            "user_text": query_text, "t_start": time.monotonic(),
            "tool_events": [], "reply": "", "retrieval": None,
        }

        # Re-render instructions every turn (date/memory freshness) — matches the original
        # server.py rebuilding its system prompt on every LLM call, not just once at startup.
        await self.update_instructions(build_instructions())

        history = [
            {"role": m.role, "content": m.text_content or ""} for m in chat_ctx.messages()
        ] + [{"role": "user", "content": query_text}]

        retrieval_query = rag.build_retrieval_query(history)
        queries = [query_text] if retrieval_query == query_text else [query_text, retrieval_query]
        retrieval = rag.search_docs_multi(queries, k=RAG_INJECT_TOP_K)
        self._turn["retrieval"] = {**retrieval, "queries": queries}

        print(f"  [rag] status={retrieval['status']} queries={queries}")
        if retrieval["status"] == "found":
            context_block = "\n\n".join(
                f"[{r['source']} :: {r.get('section')}]\n{r['text']}" for r in retrieval["results"]
            )
            # created_at just before insert_before_ts so ChatContext's created_at-sorted insertion
            # places this system message right before the caller's turn, not after.
            chat_ctx.add_message(
                role="system",
                content=(
                    "Reference information from the bank's documents, relevant to the caller's "
                    "next question. Answer only from this and the bank info in your instructions; "
                    "if the specific detail isn't here, say you don't have it on file.\n\n"
                    + context_block
                ),
                created_at=insert_before_ts - 0.001,
            )

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        await self.prepare_turn(new_message.text_content or "", turn_ctx, new_message.created_at)

    # ── Output guardrail: never let a "card blocked / identity verified" claim reach audio
    # unless block_card actually returned "blocked" this turn (banking.py, unchanged). ─────────
    async def tts_node(self, text, model_settings: ModelSettings):
        tripped = False

        def _log_reply(sentence: str) -> None:
            # convo_log capture point: with sync_transcription=False (see session.start() below),
            # the framework never fires conversation_item_added for assistant messages — a real
            # SDK quirk, confirmed empirically, not something fixable from here — so nothing else
            # in this file reliably sees the model's spoken output. tts_node does, for every TTS
            # call (both generate_reply's streamed text and any session.say(), since Chatterbox is
            # the only registered TTS path either way), so capture the turn's reply here instead.
            if self._turn is not None:
                self._turn["reply"] += sentence

        # filtered() splits the LLM text stream into whole sentences purely for the guardrail +
        # convo_log capture; the actual audio chunking (fast first clause, then sentences) is done
        # downstream by FastStartSentenceTokenizer inside the StreamAdapter. Splitting on sentences
        # here is enough for the guardrail since asserts_block_success() matches on word presence.
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
                    if not banking.asserts_block_success(sentence):
                        _log_reply(sentence)
                        yield sentence
                    else:
                        tripped = True
                        print(f"  [guardrail] suppressed: {sentence[:70]!r}")
            if buf.strip():
                if not banking.asserts_block_success(buf):
                    _log_reply(buf)
                    yield buf
                else:
                    tripped = True
                    print(f"  [guardrail] suppressed: {buf[:70]!r}")

        # Feed the guardrail-approved text into a StreamAdapter whose FastStartSentenceTokenizer
        # emits the first clause early (fast speech start) and full sentences after. The adapter
        # keeps the framework's *pipelined* synthesis — it synthesizes upcoming sentences while the
        # current one plays — so audio stays gapless. (Bypassing it and synthesizing each chunk
        # inline stalled on every blocking Chatterbox call, which is what caused noise cuts between
        # words.) Cached on the agent so the adapter's metrics listener is registered once, not per
        # turn. Only the audio path changes; the browser transcript is forwarded separately.
        if self._tts_adapter is None:
            self._tts_adapter = tts.StreamAdapter(
                tts=self.session.tts, sentence_tokenizer=FastStartSentenceTokenizer()
            )
        async with self._tts_adapter.stream() as tts_stream:
            async def _forward() -> None:
                async for sentence in filtered():
                    tts_stream.push_text(sentence)
                tts_stream.end_input()

            forward_task = asyncio.create_task(_forward())
            try:
                async for ev in tts_stream:
                    yield ev.frame
            finally:
                await utils.aio.cancel_and_wait(forward_task)

        if tripped and self._last_block_card_status != "blocked":
            # Suppressed text wasn't backed by a real success this turn — speak a truthful
            # correction. When block_card DID return "blocked" this turn, its own outcome line
            # already spoke the (correct) confirmation; saying this fallback too would just be a
            # confusing, false-sounding double-speak on top of a genuine success.
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
        self._last_block_card_status = status
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


def _describe_tool_call(call, output) -> dict:
    """Build one convo_log tool-event dict from a FunctionCall/FunctionCallOutput pair —
    shapes matching what convo_log._tool_lines() knows how to render per tool name."""
    try:
        args = json.loads(call.arguments) if call.arguments else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    out = {}
    if output is not None and output.output:
        try:
            out = json.loads(output.output)
        except (json.JSONDecodeError, TypeError):
            out = {}
    ev = {"tool": call.name}
    if call.name == "web_search":
        ev["query"] = args.get("query", "")
    elif call.name == "remember":
        ev["fact"] = args.get("fact", "")
    elif call.name == "block_card":
        ev["status"] = out.get("status")
    elif call.name == "request_human_handoff":
        ev["ticket_id"] = out.get("ticket_id")
    return ev


# ── Worker entrypoint ─────────────────────────────────────────────────────────
def prewarm(proc: JobProcess):
    # flush=True on every line: this runs in a forked/spawned job-executor subprocess, not the
    # process `-u`/PYTHONUNBUFFERED was set on — its stdout is block-buffered by default once
    # redirected to a file, so without explicit flushing these prints (and the "Prewarm
    # complete." signal src/orchestrator.py greps for) can sit unflushed for a very long time
    # even though prewarm itself already finished, making the worker look hung when it isn't.
    print("Loading Silero VAD...", flush=True)
    proc.userdata["vad"] = silero.VAD.load()
    # No Whisper load here anymore: STT is a shared microservice (src/whisper_server.py) reached
    # over HTTP at call time, so this per-call job process holds no GPU STT model. That per-process
    # Whisper copy was what exhausted the GPU on the 2nd concurrent caller.
    print("Loading RAG embedding index...", flush=True)
    rag.init(DB_CONN)
    rag.warmup()
    print("Prewarm complete.", flush=True)
    # Touch a sentinel file (never deleted) as the container's readiness signal — Dockerfile.worker's
    # HEALTHCHECK just checks existence, so "at least one prewarm has ever succeeded since container
    # start" is enough; it doesn't need to track whether a warm process currently exists (that
    # fluctuates as calls come and go). Bare-metal (non-Docker) runs don't have a HEALTHCHECK
    # consumer for this, so a failed write there is harmless — don't let it crash prewarm.
    try:
        open("/tmp/prewarm-ready", "w").close()
    except OSError:
        pass


async def _pick_pool_url(urls: list[str], counter_key: str) -> str:
    """Round-robin pool selection via a Redis atomic counter, picked once per call (not once per
    request) — the same chosen URL is reused for a call's entire duration, so a caller hears one
    consistent voice throughout, never a mid-call switch. Redis INCR is atomic across concurrent
    callers (each call is its own OS process, no shared memory to keep a counter in), so N
    simultaneous new calls deterministically land on N different pool members whenever concurrent
    calls <= pool size — not just statistically likely, the way independent random picks would be.
    No fallback for Redis being unreachable: it's already a hard dependency for LiveKit itself in
    this stack, so this code path failing isn't a new fragility.
    """
    if len(urls) == 1:
        return urls[0]
    client = aioredis.from_url(REDIS_URL)
    try:
        idx = await client.incr(counter_key)
        picked = urls[idx % len(urls)]
        print(f"[pool] {counter_key}={idx} -> {picked}", flush=True)
        return picked
    finally:
        await client.aclose()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = BankReceptionistAgent(session_id=ctx.room.name)

    whisper_url = await _pick_pool_url(WHISPER_URLS, "whisper_pool_idx")
    chatterbox_url = await _pick_pool_url(CHATTERBOX_URLS, "chatterbox_pool_idx")

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=WhisperSTT(url=whisper_url, language=WHISPER_LANGUAGE),
        llm=openai.LLM(model=VLLM_MODEL, base_url=VLLM_URL, api_key="not-needed"),
        tts=ChatterboxTTS(url=chatterbox_url),
        userdata={"failed_card_attempts": 0},
    )

    # convo_log wiring: AgentSession has no single synchronous "turn done" callback, so the
    # in-flight turn (started in on_user_turn_completed) is filled in here as its pieces arrive
    # and flushed when the next turn starts (see BankReceptionistAgent.flush_turn_log). Reply
    # text itself is captured in tts_node, not via conversation_item_added — see the comment
    # there for why (sync_transcription=False silently stops that event firing for assistant
    # messages, confirmed empirically against the installed livekit-agents version).
    @session.on("function_tools_executed")
    def _on_tools_executed(ev) -> None:
        if agent._turn is None:
            return
        for call, output in zip(ev.function_calls, ev.function_call_outputs):
            agent._turn["tool_events"].append(_describe_tool_call(call, output))

    @session.on("close")
    def _on_close(ev) -> None:
        agent.flush_turn_log()

    # AgentSession's default text-input handling (typed chat messages, e.g. from the web
    # frontend's text form) calls generate_reply() directly and never touches
    # on_user_turn_completed — silently skipping RAG injection/instruction refresh for typed
    # input otherwise. Route it through the same prepare_turn() real speech turns use.
    async def _on_text_input(sess: AgentSession, ev) -> None:
        async with sess._claim_user_turn():
            await sess.interrupt()
            await agent.prepare_turn(ev.text, sess.history, time.time())
            sess.generate_reply(user_input=ev.text)

    await session.start(
        agent=agent, room=ctx.room,
        # RoomInputOptions defaults to 24kHz audio delivery; faster-whisper's feature extraction
        # is hardcoded for 16kHz (transcribe() takes a raw array with no sample-rate parameter to
        # tell it otherwise) — left at the default, every utterance reaches Whisper effectively
        # sped up 1.5x, producing badly garbled transcripts despite the identical model/settings
        # the original server.py used (which captured mic audio at 16kHz end-to-end). Match that
        # here instead of resampling by hand in whisper_stt.py.
        room_input_options=RoomInputOptions(text_input_cb=_on_text_input, audio_sample_rate=16000),
        # Default word-by-word transcription pacing assumes it can read real-time playback
        # progress from the TTS output to time each word's reveal — Chatterbox is a blocking,
        # whole-clip-per-sentence backend (see chatterbox_tts.py), not a smooth per-frame
        # streaming one, so that pacing estimate runs behind and reads as stuttery, with audio
        # sometimes finishing a word before the synced-paced text catches up to show it. Emitting
        # text as soon as it's generated instead (no audio-locked pacing) reads better here.
        # Side effect (confirmed empirically, not documented behavior): this also silently stops
        # the SDK emitting conversation_item_added for assistant messages, which is why reply
        # text for convo_log is captured in tts_node instead of via that event — see there.
        room_output_options=RoomOutputOptions(sync_transcription=False),
    )

    # Greet the caller immediately, before they say anything — matches src/server.py's
    # greet_caller(), which fires the instant the connection opens rather than waiting for a
    # first utterance. Not a real user turn (bypasses on_user_turn_completed / RAG injection),
    # so it's logged with the same placeholder marker the original used.
    agent._turn = {
        "user_text": "[call connected — greeting]", "t_start": time.monotonic(),
        "tool_events": [], "reply": "", "retrieval": None,
    }
    session.generate_reply(
        instructions="The call has just connected. Greet the caller now, briefly and warmly."
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # Now that STT is a shared microservice (whisper_server.py), a job process no longer
            # loads a Whisper onto the GPU — prewarm is just Silero VAD (CPU) + the RAG index
            # (~2GB RAM/process, no GPU). So concurrent calls no longer OOM the GPU; extra warm
            # processes cost host RAM, not VRAM. Was pinned to 1 during the pre-shared-Whisper
            # design (VRAM-constrained then); that constraint is gone, so this now just matches
            # livekit-agents' own production default (4) instead of a stale override — revisit
            # again once the shared services + all-workers-on-one-box concurrency ceiling is
            # actually load-tested (see plans/), rather than picking a number twice from guesswork.
            num_idle_processes=12,
            # Default is 10s — too tight for a cold container with no cached models yet: rag.py's
            # embedding model (~130MB, first download only, cached in data/fastembed_cache after
            # that — see rag.py) can take longer than that over the network, and LiveKit kills and
            # respawns the job process on timeout, restarting the download from scratch every
            # cycle — confirmed live, it never completed under the 10s default. 120s only matters
            # for that first-ever run; every run after is fast once the cache is populated.
            initialize_process_timeout=120,
        )
    )
