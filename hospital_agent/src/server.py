"""
WebSocket server -- lets you talk to the agent from a browser tab (client.html)
instead of only the terminal mic. Runs in venv-brain (it imports native_brain
directly, same as run_agent.py) and manages the TTS worker subprocess itself.

Uses the `websockets` library (current stable API, verified: `async def
handler(websocket)` with no `path` argument is the modern signature --
websockets.serve() is a stable async-context-manager entry point).

PROTOCOL (JSON text frames over the WebSocket):

  Client -> Server
    {"type": "audio_chunk", "audio_b64": "<int16 PCM16 16kHz mono, base64>"}
    {"type": "reset"}                      -- clear conversation memory

  Server -> Client
    {"type": "status", "state": "listening" | "thinking" | "speaking"}
    {"type": "caption", "role": "assistant", "text": "..."}
    {"type": "audio_chunk", "audio_b64": "...", "sample_rate": 24000}
    {"type": "interrupt"}                  -- stop already scheduled playback
    {"type": "turn_done"}
    {"type": "error", "message": "..."}

=== FIX IN THIS PASS: each VAD role gets its own model instance ===

Previously, `_silero_model` was one global object shared by BOTH `vad` and
`barge_vad`, in every connection:

    vad = VADIterator(_silero_model, threshold=0.5, ...)
    barge_vad = VADIterator(_silero_model, threshold=0.7, ...)

Silero VAD keeps hidden RNN state *inside the model object itself*, reset
only via `.reset_states()`. `vad` runs while listening; `barge_vad` runs
while a reply is being spoken. Because they take turns feeding the SAME
stateful model with no reset at the handoff, each was quietly corrupting
the other's hidden state -- on every single turn, in every connection (they
all shared the one global instance). This degrades detection accuracy
independently of whether generate()-cancellation works at all, and is a
very plausible reason barge-in felt unreliable even before that was fixed.

Fixed by loading one "template" model once, then giving every VADIterator
its own `copy.deepcopy()` of it. Silero VAD is small; this is cheap and
avoids re-invoking `torch.hub.load()` (network/cache lookup) per connection.

=== FIX FROM THE PREVIOUS PASS, still in effect: barge-in during generation ===

That part required a native_brain.py change (a real `StoppingCriteria` tied
to `cancel_event`, checked inside model.generate() itself) -- this file's
plumbing (buffering audio while busy, replaying it once a turn frees up,
serializing GPU access via `_gpu_lock`) was already correct; it just had
nothing underneath it that actually stopped generation quickly. With that
fixed, `turn_task.done()` now becomes True within about one token's
generation time of a barge-in, instead of only once the abandoned reply
finished generating in the background.

DESIGN NOTES / KNOWN LIMITATIONS (stated plainly rather than glossed over):

  - Gemma 4 + Qwen3-TTS are both single-GPU, single-instance resources in
    this setup. All turns, across all connected browser tabs, are serialized
    through one lock (`_gpu_lock`). That's correct for "stream on my
    laptop" -- it is NOT a multi-tenant server.

  - Barge-in over the socket stops audible playback immediately AND now
    actually stops the in-flight model.generate() call (see native_brain.py).
    Audio you speak while a turn is busy is never silently dropped -- it's
    buffered and replayed through the VAD the moment the turn frees up.
"""

import asyncio
import copy
import json
import queue as pyqueue
import subprocess
import threading
import time as _time
from collections.abc import Awaitable, Callable

import numpy as np
import torch
import websockets

from hospital_agent.src.vad_iterator import VADIterator
from audio_utils import SAMPLE_RATE, b64_pcm16_to_float_audio
import hospital_agent.src.native_brain as native_brain

# ---- CONFIG ----
VENV_VOICE_PYTHON = "/home/nauyan/voice-agent-pipeline/.venv-voice/bin/python"
TTS_WORKER_SCRIPT = "/home/nauyan/voice-agent-pipeline/src/tts_worker.py"
HOST = "0.0.0.0"
PORT = 8765
BARGE_IN_MIN_SPEECH_MS = 250
# Hard cap on how many incoming chunks we'll buffer while a turn is busy.
# This is a safety valve in case generation stalls for a long time -- without
# it a stuck turn could make this list grow forever. ~a few seconds of audio
# at typical browser chunk sizes.
MAX_PENDING_CHUNKS = 200

_gpu_lock = threading.Lock()  # serializes Gemma + Qwen3-TTS access, see docstring
_tts: "TTSClient | None" = None
_silero_template = None  # loaded once in main(); deep-copied per VAD role/connection


def _fresh_vad_pair() -> tuple[VADIterator, VADIterator]:
    """Build a turn-taking VAD and a barge-in VAD, each with its OWN Silero
    model instance (see module docstring, "FIX IN THIS PASS"). Call this
    once per connection."""
    vad = VADIterator(
        copy.deepcopy(_silero_template),
        threshold=0.5,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=500,
    )
    barge_vad = VADIterator(
        copy.deepcopy(_silero_template),
        threshold=0.7,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=100,
    )
    return vad, barge_vad


class TTSClient:
    """Same JSON/base64 protocol as run_agent.py's TTSClient, but yields
    (audio_b64, sample_rate) pairs instead of pushing PCM onto a local
    playback queue -- the caller here forwards them over a websocket."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc

    def synthesize(
        self,
        text: str,
        instruct: str | None,
        cancel_event: threading.Event | None = None,
    ):
        request = json.dumps({"text": text, "instruct": instruct})
        self.proc.stdin.write(request + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            if line == "<<END>>":
                return
            if line.startswith("<<ERROR>>"):
                print(f"❌ TTS worker: {line}")
                continue
            try:
                chunk = json.loads(line)
                if cancel_event is None or not cancel_event.is_set():
                    yield chunk["audio_b64"], chunk["sample_rate"]
            except Exception as e:  # noqa: BLE001
                print(f"❌ Could not decode TTS chunk: {e}")


def start_tts_worker() -> TTSClient:
    proc = subprocess.Popen(
        [VENV_VOICE_PYTHON, "-u", TTS_WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    _time.sleep(1.0)
    if proc.poll() is not None:
        raise RuntimeError(
            f"TTS worker process exited immediately (code {proc.returncode}). "
            f"Check TTS_WORKER_SCRIPT path: {TTS_WORKER_SCRIPT}"
        )
    return TTSClient(proc)


def process_turn_blocking(
    utterance: np.ndarray,
    out_q: "pyqueue.Queue",
    cancel_event: threading.Event,
) -> None:
    """Runs in a worker thread (kept off the asyncio event loop, since both
    model.generate() and TTS synthesis are blocking calls). Pushes dict
    events onto out_q; a None sentinel signals the turn is complete.

    Now that native_brain.respond_to_audio() actually honors cancel_event
    all the way down to generate() (via a StoppingCriteria), this loop
    exits -- and releases `_gpu_lock` -- within about one token's
    generation time of a barge-in, instead of only once an abandoned reply
    finished generating in the background.
    """
    try:
        with _gpu_lock:
            for event in native_brain.respond_to_audio(
                utterance, sampling_rate=SAMPLE_RATE, cancel_event=cancel_event
            ):
                if cancel_event.is_set():
                    continue
                if event["type"] == "sentence" and event["text"]:
                    out_q.put(
                        {"type": "caption", "role": "assistant", "text": event["text"]}
                    )
                    for audio_b64, sr in _tts.synthesize(
                        event["text"], event.get("tone"), cancel_event
                    ):
                        out_q.put(
                            {
                                "type": "audio_chunk",
                                "audio_b64": audio_b64,
                                "sample_rate": sr,
                            }
                        )
    except Exception as e:  # noqa: BLE001
        out_q.put({"type": "error", "message": str(e)})
    finally:
        out_q.put(None)


async def handle_turn(
    send_event: Callable[[dict], Awaitable[None]],
    utterance: np.ndarray,
    cancel_event: threading.Event,
) -> None:
    await send_event({"type": "status", "state": "thinking"})

    out_q: "pyqueue.Queue" = pyqueue.Queue()
    thread = threading.Thread(
        target=process_turn_blocking,
        args=(utterance, out_q, cancel_event),
        daemon=True,
    )
    thread.start()

    loop = asyncio.get_running_loop()
    sent_first_audio = False
    while True:
        item = await loop.run_in_executor(None, out_q.get)
        if item is None:
            break
        if cancel_event.is_set():
            continue
        if item["type"] == "audio_chunk" and not sent_first_audio:
            await send_event({"type": "status", "state": "speaking"})
            sent_first_audio = True
        await send_event(item)

    if not cancel_event.is_set():
        await send_event({"type": "turn_done"})
        await send_event({"type": "status", "state": "listening"})


def is_barge_in(audio_float: np.ndarray, barge_vad: VADIterator) -> bool:
    min_samples = SAMPLE_RATE * BARGE_IN_MIN_SPEECH_MS // 1000
    _ = barge_vad(torch.from_numpy(audio_float))
    return barge_vad.triggered and barge_vad.active_speech_samples >= min_samples


async def handler(websocket) -> None:
    print("🔌 Client connected")
    send_lock = asyncio.Lock()

    async def send_event(payload: dict) -> None:
        async with send_lock:
            await websocket.send(json.dumps(payload))

    # FIX: each connection gets its own two Silero model instances (one per
    # VAD role) instead of both roles sharing one global model -- see
    # module docstring.
    vad, barge_vad = _fresh_vad_pair()

    turn_task: asyncio.Task | None = None
    turn_cancel: threading.Event | None = None

    # Audio chunks that arrived while a turn was busy (thinking/speaking).
    # We can't safely run the primary VAD on them the instant they arrive
    # (its state needs to stay contiguous with whatever utterance starts
    # the *next* turn), so we hold them here and replay them through the
    # VAD, in order, the moment the turn frees up. This is what stops
    # speech from being silently lost during busy windows.
    pending_audio_chunks: list[np.ndarray] = []

    def feed_vad(chunk: np.ndarray) -> np.ndarray | None:
        """Run one chunk through the primary VAD. Returns a finished
        utterance (float32 numpy array) if speech just completed, else None."""
        speech = vad(torch.from_numpy(chunk))
        if speech is None:
            return None
        return torch.cat(speech).cpu().numpy()

    await send_event({"type": "status", "state": "listening"})
    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")
            if mtype == "reset":
                if turn_task is not None and not turn_task.done() and turn_cancel:
                    turn_cancel.set()
                    await send_event({"type": "interrupt"})
                native_brain.reset_memory()
                vad.reset_states()
                barge_vad.reset_states()
                pending_audio_chunks.clear()
                await send_event({"type": "status", "state": "listening"})
                continue

            if mtype != "audio_chunk":
                continue

            audio_float = b64_pcm16_to_float_audio(msg["audio_b64"])

            # --- A turn is currently generating and/or speaking. ---
            if turn_task is not None and not turn_task.done():
                if (
                    turn_cancel is not None
                    and not turn_cancel.is_set()
                    and is_barge_in(audio_float, barge_vad)
                ):
                    print("🛑 Web barge-in detected – stopping playback.")
                    turn_cancel.set()
                    vad.reset_states()
                    barge_vad.reset_states()
                    # This is a fresh utterance boundary -- anything buffered
                    # from before the interruption belongs to a turn we just
                    # killed, so drop it rather than replay stale audio.
                    pending_audio_chunks.clear()
                    await send_event({"type": "interrupt"})
                    await send_event({"type": "status", "state": "listening"})

                # Don't drop this chunk: native_brain/TTS are still running
                # in the worker thread (cancel_event stops them within
                # about one token's generation time, but not literally
                # instantly), so we can't safely run the primary VAD yet.
                # Buffer it and replay once the turn is free.
                pending_audio_chunks.append(audio_float)
                if len(pending_audio_chunks) > MAX_PENDING_CHUNKS:
                    pending_audio_chunks.pop(0)
                continue

            # --- Turn just finished; clean it up. ---
            if turn_task is not None and turn_task.done():
                try:
                    turn_task.result()
                except Exception as e:  # noqa: BLE001
                    print(f"❌ Turn error: {e}")
                    await send_event({"type": "error", "message": str(e)})
                    await send_event({"type": "status", "state": "listening"})
                turn_task = None
                turn_cancel = None
                barge_vad.reset_states()

            # --- Turn is free. Replay anything buffered while we were busy,
            # in order, then the chunk that just arrived. ---
            backlog = pending_audio_chunks
            pending_audio_chunks = []
            backlog.append(audio_float)

            for chunk in backlog:
                if turn_task is not None:
                    # An utterance completed partway through the backlog and
                    # a new turn already started -- everything after that
                    # belongs to the *next* turn, so buffer the remainder
                    # instead of feeding it into a VAD that's mid-turn again.
                    pending_audio_chunks.append(chunk)
                    continue

                utterance = feed_vad(chunk)
                if utterance is None:
                    continue

                print(f"🎤 Heard {len(utterance) / SAMPLE_RATE:.2f}s, thinking...")
                turn_cancel = threading.Event()
                turn_task = asyncio.create_task(
                    handle_turn(send_event, utterance, turn_cancel)
                )
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        if turn_cancel is not None:
            turn_cancel.set()
        if turn_task is not None and not turn_task.done():
            await asyncio.wait({turn_task}, timeout=0.1)
        print("🔌 Client disconnected")


async def main() -> None:
    global _tts, _silero_template

    print("⏳ Starting Qwen3-TTS worker...")
    _tts = start_tts_worker()

    print("⏳ Loading Silero VAD...")
    _silero_template, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True, skip_validation=True
    )

    print(f"✅ Serving on ws://{HOST}:{PORT}  (open client.html and point it here)")
    async with websockets.serve(handler, HOST, PORT, max_size=2**22):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
