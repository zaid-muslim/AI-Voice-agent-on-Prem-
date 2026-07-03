#!/usr/bin/env python3
"""
Voice agent WebSocket server.
Run: python3 server.py
"""

import asyncio, base64, json, re, subprocess, time
import numpy as np
import websockets
from faster_whisper import WhisperModel

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/chat"
OLLAMA_MODEL   = "qwen2.5:14b"
MAX_HISTORY_MESSAGES = 20   # ~10 exchanges of user+assistant turns, to bound context growth
CHATTERBOX_URL = "http://localhost:8766/synthesize"  # separate venv/process, see chatterbox_server.py
WS_PORT        = 8765
SYSTEM_PROMPT = (
    "You are a voice assistant having a real spoken conversation — you're a person talking, not a "
    "document being read aloud. "
    "Talk like a friend would: contractions always (it's, you're, that'll), casual interjections "
    "(oh nice, hmm, honestly, look, yeah so), and real reactions to what the person said before "
    "diving into an answer. "
    "Be frank and direct — have opinions, take a side when asked, admit when something's debatable "
    "or when you don't know. Don't hedge everything. "
    "Vary your rhythm: short punchy lines mixed with longer flowing ones. Never sound like an "
    "encyclopedia, a customer-service script, or a list read out loud. "
    "Keep answers tight — a spoken conversation, not a lecture. If a topic is big, give the "
    "interesting core and offer to go deeper. "
    "No markdown, bullets, numbered lists, or code blocks — only plain speakable sentences, since "
    "everything you write is converted directly to speech. "
    "The speech synthesizer only reads Latin/Roman script — if the person is speaking Urdu, Hindi, "
    "Arabic, or any other language, always reply using Roman transliteration (e.g. Roman Urdu) in "
    "the Latin alphabet, never in native script (no Urdu, Devanagari, Arabic, etc. script), or the "
    "synthesizer will fail."
)

stt_model = None


# ── STT ───────────────────────────────────────────────────────────────────────
def decode_to_pcm16k(audio_bytes: bytes) -> np.ndarray:
    """Decode browser-recorded audio (WebM/Opus) to 16kHz mono float32 PCM via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')[-500:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe raw browser audio bytes → text."""
    try:
        audio_array = decode_to_pcm16k(audio_bytes)
    except RuntimeError as e:
        # A truncated/empty clip (accidental tap, cancelled upload) fails to decode —
        # treat it the same as silence rather than surfacing a scary error to the user.
        print(f"  Audio decode failed, treating as silence: {e}")
        return ""
    peak = float(np.abs(audio_array).max()) if audio_array.size else 0.0
    print(f"  Audio: {audio_array.size / 16000:.2f}s, peak amplitude {peak:.4f}")
    if peak < 0.01:
        return ""   # silence/near-silence — skip to avoid Whisper hallucination
    segments, _ = stt_model.transcribe(audio_array, language="en", vad_filter=True)
    return "".join(s.text for s in segments).strip()


# ── LLM ───────────────────────────────────────────────────────────────────────
async def stream_llm(messages: list):
    """Async generator yielding LLM tokens from Ollama's /api/chat streaming API."""
    import aiohttp
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": True,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload) as resp:
            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content")
                if content:
                    yield content
                if data.get("done"):
                    break


# ── TTS ───────────────────────────────────────────────────────────────────────
async def synthesize_to_wav_b64(text: str) -> str:
    """Synthesize one sentence via the Chatterbox Turbo microservice → base64 WAV."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(CHATTERBOX_URL, json={"text": text}) as resp:
            resp.raise_for_status()
            wav_bytes = await resp.read()
    return base64.b64encode(wav_bytes).decode()


# ── Sentence Splitter ─────────────────────────────────────────────────────────
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

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


# ── Per-Turn Processing (runs as a cancellable task) ───────────────────────────
async def process_turn(ws, audio_bytes: bytes, history: list):
    t0 = time.time()

    # 1. STT
    try:
        transcript = await asyncio.get_event_loop().run_in_executor(
            None, transcribe, audio_bytes
        )
    except Exception as e:
        await ws.send(json.dumps({"type": "error", "message": f"STT error: {e}"}))
        return

    if not transcript:
        return

    await ws.send(json.dumps({"type": "transcript", "text": transcript}))
    print(f"STT ({time.time()-t0:.2f}s): {transcript}")

    # Record the user's turn right away, so it's remembered even if this turn gets interrupted.
    history.append({"role": "user", "content": transcript})
    full_reply = ""

    # 2. LLM + 3. TTS (pipelined: synthesis overlaps with LLM streaming and playback)
    MIN_TTS_CHARS = 25   # sentences shorter than this ("Sure!") merge into the next one

    sentence_q = asyncio.Queue()

    async def tts_worker():
        while True:
            sentence = await sentence_q.get()
            if sentence is None:
                return
            print(f"  TTS: {sentence[:60]}...")
            t_syn = time.time()
            wav_b64 = await synthesize_to_wav_b64(sentence)
            print(f"  TTS took {time.time()-t_syn:.2f}s ({len(sentence)} chars)")
            await ws.send(json.dumps({"type": "audio", "data": wav_b64}))

    worker = asyncio.create_task(tts_worker())
    try:
        buffer = ""
        carry = ""   # short lead-in ("Sure!") waiting to be merged with the next sentence
        async for token in stream_llm(history):
            await ws.send(json.dumps({"type": "token", "text": token}))
            buffer += token
            full_reply += token

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
        # Keep whatever the assistant managed to say (even if cut short by an interrupt)
        # so the next turn still has the context.
        if full_reply.strip():
            history.append({"role": "assistant", "content": full_reply})
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
    current_task = None
    history = []   # conversation memory for this connection: [{"role": ..., "content": ...}, ...]
    try:
        async for message in ws:
            if isinstance(message, bytes):
                # New audio always interrupts whatever turn is currently in flight.
                await cancel_current_turn(current_task)
                current_task = asyncio.create_task(process_turn(ws, message, history))
            else:
                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue
                if data.get("type") == "interrupt":
                    await cancel_current_turn(current_task)
                    await ws.send(json.dumps({"type": "interrupted"}))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {ws.remote_address}")
    finally:
        await cancel_current_turn(current_task)


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
