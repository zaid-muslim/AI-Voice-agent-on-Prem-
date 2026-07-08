#!/usr/bin/env python3
"""
Voice agent WebSocket server.
Run: python3 server.py
"""

import asyncio, base64, json, os, re, subprocess, time
from datetime import datetime
import numpy as np
import webrtcvad
import websockets
from faster_whisper import WhisperModel

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/chat"
OLLAMA_MODEL   = "qwen2.5:14b"
MAX_HISTORY_MESSAGES = 20   # ~10 exchanges of user+assistant turns, to bound context growth
CHATTERBOX_URL = "http://localhost:8766/synthesize"  # separate venv/process, see chatterbox_server.py
SEARXNG_URL    = "http://localhost:1234/search"       # SearXNG container: host port 1234 -> container 8080
WEB_SEARCH_RESULT_COUNT = 4
MIN_TTS_CHARS  = 25   # sentences shorter than this ("Sure!") merge into the next one
WS_PORT        = 8765

# ── Long-term memory (persists across sessions, unlike per-connection chat history) ──
MEMORY_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")
MAX_MEMORIES     = 60   # cap injected facts to bound prompt size

def load_memories() -> list[str]:
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        return [str(x) for x in data][-MAX_MEMORIES:]
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return []

def save_memory(fact: str) -> bool:
    """Append a durable fact to disk, skipping near-duplicates. Returns True if stored."""
    fact = fact.strip()
    if not fact:
        return False
    existing = load_memories()
    if any(fact.lower() == e.lower() for e in existing):
        return False
    existing.append(fact)
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(existing[-MAX_MEMORIES:], f, indent=2)
    except OSError as e:
        print(f"  Failed to save memory: {e}")
        return False
    print(f"  Remembered: {fact}")
    return True

# ── Domain config (swap this file to retarget the bot at a different brand/company) ──
DOMAIN_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank_config.json")

def load_domain_config() -> dict:
    """Read fresh on every prompt build (like memory) so editing the JSON takes effect
    on the next turn — no restart needed to retarget the bot at a different bank/brand."""
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


# ── Real-time VAD (continuous mic streaming) ───────────────────────────────────
VAD_SAMPLE_RATE      = 16000
VAD_FRAME_MS         = 20
VAD_FRAME_BYTES      = VAD_SAMPLE_RATE * VAD_FRAME_MS // 1000 * 2   # 640B = 320 int16 samples
VAD_AGGRESSIVENESS   = 3       # 0-3; 3 = most aggressive about rejecting non-speech
SPEECH_START_FRAMES  = 5       # ~100ms of speech to confirm the user started talking (idle)
BARGE_IN_FRAMES      = 12      # ~240ms of speech required to interrupt a speaking agent
ONSET_GAP_TOLERANCE  = 4       # allow up to ~80ms of dip mid-onset (between words) w/o resetting
SILENCE_END_FRAMES   = 30      # ~600ms of trailing silence to consider the utterance finished
MIN_UTTERANCE_FRAMES = 12      # ~240ms minimum — discard shorter blips as noise
MAX_UTTERANCE_FRAMES = 1500    # ~30s safety cap on one continuous utterance

# Energy gate: webrtcvad flags any voice-like spectrum as "speech" — including the agent's
# own voice echoing back through speakers or a loopback sink, and background chatter. Requiring
# a minimum loudness on top of webrtcvad rejects faint echo/noise so only speech spoken directly
# into the mic counts. RMS is on the int16 scale (0-32768). A headset mic near the mouth reads
# well above this; attenuated echo reads below it. Raise if the agent still barges in on itself;
# lower if your real speech gets ignored. Per-utterance RMS is logged to help tune.
SPEECH_RMS_THRESHOLD = 500.0

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web, but ONLY for genuinely bank/finance-relevant information that isn't "
            "already in your bank data — e.g. a current currency exchange rate. Do NOT use this for "
            "anything unrelated to banking (weather, news, sports, general trivia, other companies) — "
            "for those, redirect the caller back to banking topics instead of searching."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    },
}

REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Save a short fact about this caller to long-term memory so you'll know it next time "
            "they call — their name, or anything they explicitly ask you to remember. Do NOT use it "
            "to collect unrelated personal details (job, address, relationships), trivia, or "
            "one-off task details. Keep the fact short and self-contained, e.g. 'The caller's name "
            "is Ahmed'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The fact to remember, phrased about the caller"},
            },
            "required": ["fact"],
        },
    },
}

TOOLS = [WEB_SEARCH_TOOL, REMEMBER_TOOL]


def build_system_prompt() -> str:
    """Rebuilt on every call so the model always has the real current date/time (it has no innate
    awareness of "now"), the latest long-term memory, and the latest domain config injected."""
    now = datetime.now().astimezone()
    now_str = now.strftime("%A, %B %d, %Y, %I:%M %p %Z")

    domain_block = format_domain_block(load_domain_config())

    memories = load_memories()
    if memories:
        memory_block = (
            "Here's what you remember about this caller from past conversations — use it naturally, "
            "don't recite it back:\n" + "\n".join(f"- {m}" for m in memories) + "\n"
        )
    else:
        memory_block = ""

    return (
        domain_block +
        memory_block +
        f"Right now it is {now_str}. Always use this as the true current date/time — never state "
        "a date or year from your training data as if it were current, and never say things like "
        "'as of 2023' or guess what year it is. Use it to answer things like whether a branch is "
        "open right now, given the hours above. "
        "You're speaking with a caller over the phone, not writing a document — talk like a real "
        "receptionist: warm, polite, and professional, but natural — contractions are fine (it's, "
        "you're, that'll), and it's fine to acknowledge what they said before answering. "
        "Keep answers short — usually one to two sentences, like a real phone call, never a "
        "paragraph or a lecture. Give the key fact first (the address, the hours, the answer), and "
        "only add more if they ask. "
        "No markdown, bullets, numbered lists, or code blocks — only plain speakable sentences, since "
        "everything you write is converted directly to speech. If listing more than one branch or "
        "service, say them as a natural spoken sentence, not a list. "
        "The speech synthesizer only reads Latin/Roman script — if the caller is speaking Urdu, Hindi, "
        "or any other language, always reply using Roman transliteration (e.g. Roman Urdu) in the "
        "Latin alphabet, never in native script, or the synthesizer will fail. "
        "Stay strictly in character and in scope: you only handle questions about this bank — "
        "branches, hours, contact info, and the general services listed above. You are informational "
        "only right now — you cannot check anyone's account, balance, or card, make transactions, "
        "or take any action on an account; if asked, politely explain that and point them to a "
        "branch visit, the mobile app, or the customer care number above. If asked something "
        "completely unrelated to banking (weather, news, sports, trivia, other topics), do NOT try to "
        "help or search for it — redirect immediately: say you're the bank's assistant and ask what "
        "banking question you can help with. Never make up a branch, phone number, rate, or policy "
        "that isn't listed above — if you don't know, say so and offer the customer care number instead. "
        "You have a web_search tool for genuinely bank-relevant information not listed above (e.g. "
        "current exchange rates) — never for unrelated topics, and never in place of the branch/hours/"
        "service info you already have. If you do search, say ONE short natural line first, like "
        "'let me check that for you', exactly once, then call the tool. Once results come back, "
        "answer directly from them in your own words, never reading out titles or URLs. "
        "You also have a remember tool for long-term memory. Call it whenever the caller shares a "
        "durable fact worth keeping for next time — their name, or something they explicitly ask you "
        "to remember. Don't announce that you're saving it, and don't save trivia or one-off details. "
        "Never write bracketed stage directions or tags like [laugh], [sigh], or [pause] — write "
        "only plain words meant to be spoken aloud."
    )

stt_model = None


# ── STT ───────────────────────────────────────────────────────────────────────
def decode_to_pcm16k(audio_bytes: bytes) -> np.ndarray:
    """Decode an arbitrary-format audio file (upload) to 16kHz mono float32 PCM via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')[-500:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def pcm16_bytes_to_array(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16kHz mono int16 PCM (from the browser's continuous VAD stream) to float32."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_array(audio_array: np.ndarray) -> str:
    """Transcribe an already-decoded 16kHz mono float32 PCM array → text."""
    peak = float(np.abs(audio_array).max()) if audio_array.size else 0.0
    print(f"  Audio: {audio_array.size / 16000:.2f}s, peak amplitude {peak:.4f}")
    if peak < 0.01:
        return ""   # silence/near-silence — skip to avoid Whisper hallucination
    segments, _ = stt_model.transcribe(audio_array, language="en", vad_filter=True)
    return "".join(s.text for s in segments).strip()


def transcribe_upload(audio_bytes: bytes) -> str:
    """Transcribe an uploaded audio file (arbitrary format) → text."""
    try:
        audio_array = decode_to_pcm16k(audio_bytes)
    except RuntimeError as e:
        # A truncated/empty clip (cancelled upload) fails to decode — treat it the same
        # as silence rather than surfacing a scary error to the user.
        print(f"  Audio decode failed, treating as silence: {e}")
        return ""
    return transcribe_array(audio_array)


# ── LLM ───────────────────────────────────────────────────────────────────────
async def stream_llm(messages: list, tools: list | None = None):
    """Async generator yielding ('content', text) or ('tool_calls', [...]) events from
    Ollama's /api/chat streaming API."""
    import aiohttp
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": build_system_prompt()}] + messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload) as resp:
            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                message = data.get("message", {})
                content = message.get("content")
                if content:
                    yield ("content", content)
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    yield ("tool_calls", tool_calls)
                if data.get("done"):
                    break


# ── Web Search ────────────────────────────────────────────────────────────────
async def web_search(query: str) -> str:
    """Query the local SearXNG instance and return a compact text block of results."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                SEARXNG_URL,
                params={"q": query, "format": "json"},
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
        title = (r.get("title") or "").strip()
        snippet = (r.get("content") or "").strip()
        lines.append(f"- {title}: {snippet}")
    return "\n".join(lines)


# ── TTS ───────────────────────────────────────────────────────────────────────
BRACKET_TAG = re.compile(r'\s*\[[^\]]*\]\s*')

async def synthesize_to_wav_b64(text: str) -> str:
    """Synthesize one sentence via the Chatterbox Turbo microservice → base64 WAV."""
    import aiohttp
    text = BRACKET_TAG.sub(" ", text).strip()   # never speak stray [tags] aloud
    if not text:
        return ""
    async with aiohttp.ClientSession() as session:
        async with session.post(CHATTERBOX_URL, json={"text": text}) as resp:
            resp.raise_for_status()
            wav_bytes = await resp.read()
    return base64.b64encode(wav_bytes).decode()


# ── Sentence Splitter ─────────────────────────────────────────────────────────
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
# For the very first chunk of a reply, break at the earliest clause boundary too
# (comma/semicolon/colon), so audio starts after a few words instead of waiting for
# the model to finish a whole first sentence — the biggest source of start latency.
FIRST_CHUNK_END = re.compile(r'(?<=[.!?,;:])\s+')
FIRST_CHUNK_MIN_CHARS = 15

def split_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Split completed sentences out of a running buffer.
    Returns (list_of_complete_sentences, remaining_buffer).
    """
    parts = SENTENCE_END.split(buffer)
    if len(parts) == 1:
        return [], buffer          # no sentence boundary yet
    complete = parts[:-1]
    remainder = parts[-1]
    return complete, remainder


async def speak_stream(ws, sentence_q: asyncio.Queue, token_iter) -> tuple[str, list | None]:
    """Consume a stream_llm() event stream: forward text tokens to the client as they
    arrive and queue completed sentences for TTS. Returns (full_text_spoken, tool_calls),
    where tool_calls is None if the model didn't call a tool this pass."""
    buffer = ""
    carry = ""   # short lead-in ("Sure!") waiting to be merged with the next sentence
    full_text = ""
    tool_calls = None
    first_chunk_done = False

    async for kind, payload in token_iter:
        if kind == "tool_calls":
            tool_calls = payload
            continue

        token = payload
        await ws.send(json.dumps({"type": "token", "text": token}))
        buffer += token
        full_text += token

        # Fast path: get the first chunk out at the earliest clause boundary that already
        # has enough words, so speech starts quickly; then settle into full-sentence chunks.
        if not first_chunk_done:
            for m in FIRST_CHUNK_END.finditer(buffer):
                if len(buffer[:m.start()].strip()) >= FIRST_CHUNK_MIN_CHARS:
                    await sentence_q.put(buffer[:m.start()].strip())
                    buffer = buffer[m.end():]
                    first_chunk_done = True
                    break
            if not first_chunk_done:
                continue

        sentences, buffer = split_sentences(buffer)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if carry:
                sentence = carry + " " + sentence
                carry = ""
            if len(sentence) < MIN_TTS_CHARS:
                carry = sentence
                continue
            await sentence_q.put(sentence)

    # flush whatever is left (carry + incomplete final sentence)
    tail = (carry + " " + buffer.strip()).strip()
    if tail:
        await sentence_q.put(tail)

    return full_text, tool_calls


# ── Per-Turn Processing (runs as a cancellable task) ───────────────────────────
async def process_turn(ws, audio_array: np.ndarray, history: list):
    """Entry point for the real-time VAD-segmented mic stream — audio is already
    decoded 16kHz mono float32 PCM, no STT-format decoding needed."""
    t0 = time.time()
    try:
        transcript = await asyncio.get_event_loop().run_in_executor(
            None, transcribe_array, audio_array
        )
    except Exception as e:
        await ws.send(json.dumps({"type": "error", "message": f"STT error: {e}"}))
        return
    if not transcript:
        return
    await ws.send(json.dumps({"type": "transcript", "text": transcript}))
    print(f"STT ({time.time()-t0:.2f}s): {transcript}")
    await respond_to_transcript(ws, transcript, history, t0)


async def process_uploaded_audio(ws, audio_bytes: bytes, history: list):
    """Entry point for the file-upload path — arbitrary format, needs ffmpeg decode."""
    t0 = time.time()
    try:
        transcript = await asyncio.get_event_loop().run_in_executor(
            None, transcribe_upload, audio_bytes
        )
    except Exception as e:
        await ws.send(json.dumps({"type": "error", "message": f"STT error: {e}"}))
        return
    if not transcript:
        return
    await ws.send(json.dumps({"type": "transcript", "text": transcript}))
    print(f"STT ({time.time()-t0:.2f}s): {transcript}")
    await respond_to_transcript(ws, transcript, history, t0)


async def speak_and_return(ws, messages: list, tools: list | None) -> tuple[str, list | None]:
    """Run one LLM pass with TTS playback pipelined to it. Returns (full_text, tool_calls).
    Shared by respond_to_transcript (real user turns) and greet_caller (opening greeting)."""
    sentence_q = asyncio.Queue()

    async def tts_worker():
        while True:
            sentence = await sentence_q.get()
            if sentence is None:
                return
            print(f"  TTS: {sentence[:60]}...")
            t_syn = time.time()
            wav_b64 = await synthesize_to_wav_b64(sentence)
            if not wav_b64:
                continue   # sentence was empty (e.g. only a stray tag) — nothing to play
            print(f"  TTS took {time.time()-t_syn:.2f}s ({len(sentence)} chars)")
            await ws.send(json.dumps({"type": "audio", "data": wav_b64}))

    worker = asyncio.create_task(tts_worker())
    try:
        reply, tool_calls = await speak_stream(ws, sentence_q, stream_llm(messages, tools=tools))
        await sentence_q.put(None)
        await worker
        return reply, tool_calls
    finally:
        if not worker.done():
            worker.cancel()
            try:
                await worker
            except (asyncio.CancelledError, Exception):
                pass


async def greet_caller(ws, history: list):
    """Speak a welcome message the instant a call connects. The trigger isn't a real user
    message, so it's never committed to history — only the assistant's reply is — meaning
    history reads naturally starting from the caller's first real turn."""
    trigger = [{"role": "user", "content": "[The call has just connected. Greet the caller now.]"}]
    try:
        reply, _ = await speak_and_return(ws, trigger, None)
        if reply.strip():
            history.append({"role": "assistant", "content": reply})
        await ws.send(json.dumps({"type": "done"}))
    except asyncio.CancelledError:
        print("  Greeting interrupted")
        raise
    except Exception as e:
        print(f"  Greeting error: {e}")


async def respond_to_transcript(ws, transcript: str, history: list, t0: float):
    """LLM (+ optional web search / remember) + TTS for one user turn, shared by both
    audio entry points."""
    # Record the user's turn right away, so it's remembered even if this turn gets interrupted.
    history.append({"role": "user", "content": transcript})
    # Assistant/tool turns are staged here and only committed to `history` once each step
    # actually finishes — so an interruption preserves exactly what really happened, no more.
    turns_to_commit = []

    try:
        reply, tool_calls = await speak_and_return(ws, history, TOOLS)

        if tool_calls:
            turns_to_commit.append(
                {"role": "assistant", "content": reply, "tool_calls": tool_calls}
            )
            # Execute every tool the model asked for, appending each result.
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name")
                args = fn.get("arguments") or {}
                if name == "web_search":
                    query = args.get("query", "").strip()
                    print(f"  Web search: {query}")
                    result = await web_search(query) if query else "No search query was given."
                elif name == "remember":
                    fact = args.get("fact", "").strip()
                    result = "Saved to memory." if save_memory(fact) else "Already knew that."
                else:
                    result = f"Unsupported tool: {name}"
                turns_to_commit.append(
                    {"role": "tool", "content": result, "tool_call_id": call.get("id", "")}
                )

            # Follow-up pass, no tools this time — forces a final spoken answer grounded in the
            # tool results instead of calling tools again.
            follow_reply, _ = await speak_and_return(ws, history + turns_to_commit, None)
            if follow_reply.strip():
                turns_to_commit.append({"role": "assistant", "content": follow_reply})
        elif reply.strip():
            turns_to_commit.append({"role": "assistant", "content": reply})
    except asyncio.CancelledError:
        print(f"  Turn interrupted ({time.time()-t0:.2f}s in)")
        raise
    except Exception as e:
        print(f"  LLM/TTS error: {e}")
        await ws.send(json.dumps({"type": "error", "message": f"LLM/TTS error: {e}"}))
        return
    finally:
        history.extend(turns_to_commit)
        del history[:-MAX_HISTORY_MESSAGES]

    await ws.send(json.dumps({"type": "done"}))
    print(f"Turn done ({time.time()-t0:.2f}s total)")


async def cancel_current_turn(current_task):
    """Cancel an in-flight turn, if any, and wait for it to unwind."""
    if current_task and not current_task.done():
        current_task.cancel()
        try:
            await current_task
        except (asyncio.CancelledError, Exception):
            pass   # cancellation itself, or a connection error while unwinding — either way, done


# ── Main Handler ──────────────────────────────────────────────────────────────
async def handle_client(ws):
    print(f"Client connected: {ws.remote_address}")
    history = []   # conversation memory for this connection: [{"role": ..., "content": ...}, ...]
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    # VAD state — all mutated from the single-threaded frame loop below.
    state = {
        "current_task": None,
        "is_speaking": False,
        "speech_frames": 0,
        "silence_frames": 0,
        "utterance": bytearray(),
        "leftover": b"",       # incoming chunks don't always align to exact frame boundaries
        "peak_rms": 0.0,       # loudest frame in the current utterance (for tuning logs)
        "onset_gap": 0,        # consecutive non-speech frames during a speech onset
        "agent_audible": False,  # client is currently playing agent audio (reported by client)
    }

    def agent_active():
        # The agent is "active" (and thus interruptible) while it's generating a response
        # OR while the client is still playing its audio out.
        t = state["current_task"]
        return (t is not None and not t.done()) or state["agent_audible"]

    async def finalize_utterance():
        frames = len(state["utterance"]) // VAD_FRAME_BYTES
        if frames >= MIN_UTTERANCE_FRAMES:
            print(f"[VAD] utterance finalized: {frames * VAD_FRAME_MS / 1000:.2f}s, peak RMS {state['peak_rms']:.0f}")
            audio_array = pcm16_bytes_to_array(bytes(state["utterance"]))
            await cancel_current_turn(state["current_task"])
            state["current_task"] = asyncio.create_task(process_turn(ws, audio_array, history))
        else:
            print(f"[VAD] discarded short blip: {frames} frames, peak RMS {state['peak_rms']:.0f}")
        state["utterance"] = bytearray()
        state["is_speaking"] = False
        state["speech_frames"] = 0
        state["silence_frames"] = 0
        state["peak_rms"] = 0.0
        state["onset_gap"] = 0

    async def handle_pcm_chunk(chunk: bytes):
        state["leftover"] += chunk
        while len(state["leftover"]) >= VAD_FRAME_BYTES:
            frame = bytes(state["leftover"][:VAD_FRAME_BYTES])
            state["leftover"] = state["leftover"][VAD_FRAME_BYTES:]

            # Speech = voice-like spectrum (webrtcvad) AND loud enough (energy gate).
            # The energy gate is what rejects the agent's own echo and background noise.
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
            is_speech = rms >= SPEECH_RMS_THRESHOLD and vad.is_speech(frame, VAD_SAMPLE_RATE)

            if not state["is_speaking"]:
                if is_speech:
                    state["speech_frames"] += 1
                    state["onset_gap"] = 0
                    state["utterance"] += frame
                    state["peak_rms"] = max(state["peak_rms"], rms)
                    # While the agent is talking, demand more sustained speech before treating
                    # it as a real barge-in — a brief blip or leaked echo shouldn't cut the
                    # agent off, but a genuine sentence should.
                    needed = BARGE_IN_FRAMES if agent_active() else SPEECH_START_FRAMES
                    if state["speech_frames"] >= needed:
                        state["is_speaking"] = True
                        state["silence_frames"] = 0
                        if agent_active():
                            print(f"[VAD] barge-in (RMS {rms:.0f}) — interrupting agent")
                            await cancel_current_turn(state["current_task"])
                            state["current_task"] = None
                            state["agent_audible"] = False
                            await ws.send(json.dumps({"type": "interrupted"}))
                elif state["speech_frames"] > 0:
                    # Mid-onset: tolerate a brief dip (between words) rather than resetting on
                    # the first non-speech frame, which would make onset detection fragile.
                    state["onset_gap"] += 1
                    state["utterance"] += frame   # keep the gap so captured audio isn't choppy
                    if state["onset_gap"] > ONSET_GAP_TOLERANCE:
                        state["speech_frames"] = 0
                        state["onset_gap"] = 0
                        state["utterance"].clear()
                        state["peak_rms"] = 0.0
            else:
                state["utterance"] += frame
                state["peak_rms"] = max(state["peak_rms"], rms)
                if is_speech:
                    state["silence_frames"] = 0
                else:
                    state["silence_frames"] += 1
                    if state["silence_frames"] >= SILENCE_END_FRAMES:
                        await finalize_utterance()
                        continue
                if len(state["utterance"]) >= MAX_UTTERANCE_FRAMES * VAD_FRAME_BYTES:
                    await finalize_utterance()

    # Call connected: greet the caller immediately, before waiting for them to speak.
    # This runs through the same task/barge-in machinery as any other turn, so talking
    # over the greeting interrupts it exactly like interrupting any other response.
    state["current_task"] = asyncio.create_task(greet_caller(ws, history))

    try:
        async for message in ws:
            if isinstance(message, bytes):
                await handle_pcm_chunk(message)
            else:
                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue
                if data.get("type") == "playback_state":
                    # The client tells us when it's actually playing agent audio. Audio is
                    # buffered/scheduled ahead, so it keeps playing after the LLM/TTS task
                    # has already finished — without this, barge-in wouldn't fire against
                    # that still-playing audio.
                    state["agent_audible"] = bool(data.get("playing"))
                elif data.get("type") == "interrupt":
                    await cancel_current_turn(state["current_task"])
                    await ws.send(json.dumps({"type": "interrupted"}))
                elif data.get("type") == "upload_audio":
                    await cancel_current_turn(state["current_task"])
                    audio_bytes = base64.b64decode(data.get("data", ""))
                    state["current_task"] = asyncio.create_task(
                        process_uploaded_audio(ws, audio_bytes, history)
                    )

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {ws.remote_address}")
    finally:
        await cancel_current_turn(state["current_task"])


# ── Entry Point ───────────────────────────────────────────────────────────────
async def main():
    async with websockets.serve(handle_client, "0.0.0.0", WS_PORT):
        print(f"Voice agent listening on ws://0.0.0.0:{WS_PORT}")
        await asyncio.Future()   # run forever

if __name__ == "__main__":
    print("Loading STT model...")
    stt_model = WhisperModel("large-v3", compute_type="int8_float16", device="cuda", device_index=0)

    print("Models loaded. Starting server... (TTS served by chatterbox_server.py)")
    asyncio.run(main())
