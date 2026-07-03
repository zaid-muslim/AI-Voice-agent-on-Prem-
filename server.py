#!/usr/bin/env python3
"""
Voice agent WebSocket server.
Run: python3 server.py
"""

import asyncio, base64, json, re, subprocess, time
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
            "Search the web for current, real-time, or otherwise unknown information — news, "
            "prices, weather, recent events, sports scores, or specific facts you're not confident "
            "about. Do NOT use this for general knowledge, definitions, math, or anything you "
            "already know well — searching every time is slow and unnecessary."
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

def build_system_prompt() -> str:
    """Rebuilt on every call so the model always has the real current date/time — it has
    no innate awareness of "now" and will otherwise guess from stale training data."""
    now = datetime.now().astimezone()
    now_str = now.strftime("%A, %B %d, %Y, %I:%M %p %Z")
    return (
        f"Right now it is {now_str}. Always use this as the true current date/time — never state "
        "a date or year from your training data as if it were current, and never say things like "
        "'as of 2023' or guess what year it is. For anything date- or time-sensitive (someone's "
        "age, how long ago something happened, whether something has already occurred), calculate "
        "it from the real date above. "
        "You are a voice assistant having a real spoken conversation — you're a person talking, not a "
        "document being read aloud. "
        "Talk like a friend would: contractions always (it's, you're, that'll), casual interjections "
        "(oh nice, hmm, honestly, look, yeah so), and real reactions to what the person said before "
        "diving into an answer. "
        "Be frank and direct — have opinions, take a side when asked, admit when something's debatable "
        "or when you don't know. Don't hedge everything. "
        "Vary your rhythm: short punchy lines mixed with longer flowing ones. Never sound like an "
        "encyclopedia, a customer-service script, or a list read out loud. "
        "Keep answers short — usually one to three sentences, like a real spoken reply, never a "
        "paragraph or a lecture. Give the key point first; if there's more, stop and offer to go "
        "deeper rather than dumping it all at once. "
        "No markdown, bullets, numbered lists, or code blocks — only plain speakable sentences, since "
        "everything you write is converted directly to speech. "
        "The speech synthesizer only reads Latin/Roman script — if the person is speaking Urdu, Hindi, "
        "Arabic, or any other language, always reply using Roman transliteration (e.g. Roman Urdu) in "
        "the Latin alphabet, never in native script (no Urdu, Devanagari, Arabic, etc. script), or the "
        "synthesizer will fail. "
        "You have a web_search tool. Only reach for it when you genuinely don't know something or the "
        "answer could be outdated (news, weather, prices, recent events, scores, anything after your "
        "training data) — never for things you already know. "
        "If you do need to search: say ONE short natural line first, like 'good question, let me "
        "check' or 'hmm, let me look that up' — exactly once, never restate or rephrase it a second "
        "time, then immediately call the tool. "
        "Once you receive search results in a later message, do not say anything about searching or "
        "checking again — just answer directly using that information, summarized conversationally "
        "in your own words, never reading out titles, URLs, or a list. "
        
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


async def respond_to_transcript(ws, transcript: str, history: list, t0: float):
    """LLM (+ optional web search) + TTS for one user turn, shared by both entry points."""
    # Record the user's turn right away, so it's remembered even if this turn gets interrupted.
    history.append({"role": "user", "content": transcript})
    # Assistant/tool turns are staged here and only committed to `history` once each step
    # actually finishes — so an interruption preserves exactly what really happened, no more.
    turns_to_commit = []

    # 2. LLM + 3. TTS (pipelined: synthesis overlaps with LLM streaming and playback)
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
        reply, tool_calls = await speak_stream(
            ws, sentence_q, stream_llm(history, tools=[WEB_SEARCH_TOOL])
        )

        if tool_calls:
            call = tool_calls[0]
            turns_to_commit.append(
                {"role": "assistant", "content": reply, "tool_calls": tool_calls}
            )
            fn = call.get("function", {})
            if fn.get("name") == "web_search":
                query = (fn.get("arguments") or {}).get("query", "").strip()
                print(f"  Web search: {query}")
                results_text = await web_search(query) if query else "No search query was given."
            else:
                results_text = f"Unsupported tool: {fn.get('name')}"
            turns_to_commit.append(
                {"role": "tool", "content": results_text, "tool_call_id": call.get("id", "")}
            )

            # Follow-up pass, no tools this time — forces a final natural-language answer
            # grounded in the search results instead of calling the tool again.
            follow_reply, _ = await speak_stream(
                ws, sentence_q, stream_llm(history + turns_to_commit, tools=None)
            )
            if follow_reply.strip():
                turns_to_commit.append({"role": "assistant", "content": follow_reply})
        elif reply.strip():
            turns_to_commit.append({"role": "assistant", "content": reply})

        await sentence_q.put(None)
        await worker
    except asyncio.CancelledError:
        print(f"  Turn interrupted ({time.time()-t0:.2f}s in)")
        raise
    except Exception as e:
        print(f"  LLM/TTS error: {e}")
        await ws.send(json.dumps({"type": "error", "message": f"LLM/TTS error: {e}"}))
        return
    finally:
        if not worker.done():
            worker.cancel()
            try:
                await worker
            except (asyncio.CancelledError, Exception):
                pass
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
