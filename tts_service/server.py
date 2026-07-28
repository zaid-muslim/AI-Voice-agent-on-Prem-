"""
Shared Qwen3-TTS service - ONE persistent, GPU-warm process serving every
LiveKit worker/room, instead of each worker process spawning its own Qwen
subprocess (plugins/qwen_tts.py's QwenSubprocessTTS - still the right
choice for a single-worker/dev setup, but the same "N concurrent callers
= N model copies" scaling problem stt_service/server.py already fixed for
STT applies to it under real concurrent load).

WHY THIS EXISTS: measured on this project's own hardware, the local
Qwen3-TTS 0.6B model gets audio to first byte in ~245ms - 2-3x faster than
the default remote Qwen3-TTS 1.7B path on PC2 (~500-700ms, paying a real
network hop). That's a genuine latency win worth having as a real,
production-shaped option, not just a per-worker subprocess fallback - see
plugins/shared_qwen_tts.py for the LiveKit-side client and README §3/§4
for the measured numbers.

CONCURRENCY - HONEST LIMIT, read before assuming this scales like STT's
service: unlike faster-whisper/CTranslate2 (which documents a safe
`num_workers` replica pool), FasterQwen3TTS's thread-safety under truly
concurrent generate_custom_voice_streaming() calls is NOT verified here.
Requests are serialized through a single asyncio.Lock - one generation
owns the GPU at a time, everything else queues. This is the same
single-owner constraint plugins/qwen_tts.py's QwenSubprocessTTS already
has (one pipe, one _request_lock) - just now shared across every room
instead of per-worker-process. Fine for the "3 concurrent callers"
baseline this project is tuned for (see README §10); if TTS ever shows up
as a queuing bottleneck under real load, that's real evidence to
investigate whether the model supports concurrent calls before assuming
this needs a replica pool.

WIRE FORMAT: newline-delimited JSON over a chunked HTTP response - the
exact same shape qwen_worker.py already streams over its stdin/stdout
pipe (line = {"audio_b64": ..., "sample_rate": ...}, final line =
{"done": true}), just carried over HTTP instead of a subprocess pipe. One
simplification an HTTP-per-request transport buys for free that the
subprocess's shared-pipe protocol needed real work for: no
request-id/stale-line filtering. Each HTTP request gets its own private
response stream, so there's nothing to filter.

INTERRUPTION: if the client (LiveKit's ChunkedStream) is cancelled - a
caller barging in mid-reply - it stops reading and closes the HTTP
connection. This handler notices via request.is_disconnected(), sets a
per-request cancel event, and the generation thread stops within one
yielded chunk (same "can't abort mid-CUDA-step" granularity
qwen_worker.py already documents).

Run standalone:
    python server.py
(needs the same faster-qwen3-tts/torch stack as .venv-voice - see
requirements.txt.) Or via Docker: docker/tts.Dockerfile +
docker-compose.yml's opt-in "local-tts" profile - see README §6/§8.
requirements.txt pins torchaudio==2.6.0 and installs numpy/
typing_extensions before the rest explicitly - both were real, confirmed
Docker build failures (sox's legacy setup.py needs them at build time;
letting pip resolve torchaudio on its own picked an incompatible 2.11.0
needing a newer CUDA runtime than this image has) - see tts.Dockerfile's
comments for the full story if either ever needs touching again.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

MODEL_ID = os.environ.get(
    "QWEN_MODEL_ID", "/home/nauyan/voice-agent-pipeline/models/Qwen3-TTS-0.6B-custom"
)
DTYPE = os.environ.get("QWEN_TTS_DTYPE", "bfloat16")
DEFAULT_SPEAKER = os.environ.get("QWEN_SPEAKER", "aiden")
DEFAULT_LANGUAGE = os.environ.get("QWEN_LANGUAGE", "English")
CHUNK_SIZE = int(os.environ.get("QWEN_CHUNK_SIZE", "8"))
DISCONNECT_POLL_SECS = 0.1

_state = {"model": None, "in_flight": 0, "total_requests": 0}
_gpu_lock = asyncio.Lock()  # single-lane - see module docstring's CONCURRENCY note
# Generation itself runs sync (blocking CUDA calls) on one dedicated thread,
# same "one lane" choice plugins/whisper_stt.py's FasterWhisperSTT makes for
# the same reason: one model object, not safe for concurrent use.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-tts")


def _load_model():
    from faster_qwen3_tts import FasterQwen3TTS

    logger.info(f"Loading Qwen3-TTS '{MODEL_ID}' (dtype={DTYPE}) ...")
    model = FasterQwen3TTS.from_pretrained(MODEL_ID, dtype=DTYPE)

    # Warm-up: run the full streaming path a few times so CUDA graph
    # capture/kernel autotuning are paid here, not on the first real
    # caller - same 3-full-runs recipe qwen_worker.py already proved gets
    # the ~245ms first-chunk floor from turn one (see that file's comment
    # for the measured cold-vs-warm delta: ~750ms cold, ~245ms warm).
    t0 = time.perf_counter()
    for _ in range(3):
        for _chunk in model.generate_custom_voice_streaming(
            text="Warming up the speech synthesis engine now.",
            language=DEFAULT_LANGUAGE,
            speaker=DEFAULT_SPEAKER,
            chunk_size=CHUNK_SIZE,
        ):
            pass  # drain fully - every kernel in the path gets exercised
    logger.info(f"Qwen3-TTS: ready (warm), warm-up took {(time.perf_counter() - t0) * 1000:.0f}ms")
    return model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _state["model"] = await asyncio.get_running_loop().run_in_executor(None, _load_model)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Shared Qwen3-TTS service", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    speaker: str = DEFAULT_SPEAKER
    language: str = DEFAULT_LANGUAGE
    chunk_size: int = CHUNK_SIZE


def _generate_sync(
    text: str,
    language: str,
    speaker: str,
    chunk_size: int,
    out_q: "queue.Queue[tuple]",
    cancel_event: threading.Event,
) -> None:
    """Runs on _executor's single thread. Pushes ("chunk", pcm_bytes, sr),
    then either ("done", None, None) or ("error", message, None) - always
    exactly one terminal item, mirroring qwen_worker.py's
    always-signal-done-even-on-error guarantee."""
    model = _state["model"]
    try:
        for audio_chunk, sr, _ in model.generate_custom_voice_streaming(
            text=text, language=language, speaker=speaker, chunk_size=chunk_size
        ):
            if cancel_event.is_set():
                break
            pcm_bytes = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            out_q.put(("chunk", pcm_bytes, int(sr)))
    except Exception as exc:  # noqa: BLE001
        out_q.put(("error", str(exc), None))
        out_q.put(("done", None, None))
        return
    out_q.put(("done", None, None))


async def _stream_response(req: SynthesizeRequest, request: Request):
    out_q: "queue.Queue[tuple]" = queue.Queue()
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()

    async with _gpu_lock:  # one generation owns the GPU at a time - see docstring
        gen_future = loop.run_in_executor(
            _executor,
            _generate_sync,
            req.text,
            req.language,
            req.speaker,
            req.chunk_size,
            out_q,
            cancel_event,
        )
        try:
            while True:
                # Checked EVERY iteration, not just when the queue is empty -
                # chunks normally arrive well within DISCONNECT_POLL_SECS of
                # each other (~50-60ms apart for this model), so a
                # queue.Empty-gated check would almost never fire and a
                # barge-in interruption wouldn't be noticed until the WHOLE
                # reply finished generating on its own (confirmed live: a
                # cancelled long-reply request kept holding _gpu_lock for
                # its full ~6s, stalling the next real request behind it).
                if await request.is_disconnected():
                    # Caller (LiveKit ChunkedStream) was interrupted - stop
                    # burning GPU on audio nobody will hear.
                    cancel_event.set()
                    break
                try:
                    kind, payload, sr = await loop.run_in_executor(
                        None, out_q.get, True, DISCONNECT_POLL_SECS
                    )
                except queue.Empty:
                    continue

                if kind == "chunk":
                    yield (
                        json.dumps({"audio_b64": base64.b64encode(payload).decode("ascii"), "sample_rate": sr})
                        + "\n"
                    ).encode()
                elif kind == "error":
                    yield (json.dumps({"error": payload}) + "\n").encode()
                elif kind == "done":
                    yield (json.dumps({"done": True}) + "\n").encode()
                    break
        finally:
            # Unconditional, not just on the is_disconnected() path above:
            # this generator can also be torn down by Starlette itself
            # (GeneratorExit) when a write to an already-closed socket
            # fails, WITHOUT ever passing through our polling check above -
            # confirmed live, this was the actual bug: the explicit
            # is_disconnected() poll alone left a cancelled long-reply
            # request's background thread running for its full ~6s,
            # starving the next real request behind _gpu_lock. Setting
            # cancel_event here too is a safe no-op if generation already
            # reached "done" on its own.
            cancel_event.set()
            await gen_future


@app.get("/health")
async def health():
    return {
        "status": "ok" if _state["model"] is not None else "loading",
        "model_id": MODEL_ID,
        "dtype": DTYPE,
        "in_flight": _state["in_flight"],
        "total_requests": _state["total_requests"],
    }


@app.post("/v1/synthesize")
async def synthesize(req: SynthesizeRequest, request: Request):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model still loading")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    _state["total_requests"] += 1
    _state["in_flight"] += 1

    async def _wrapped():
        try:
            async for line in _stream_response(req, request):
                yield line
        finally:
            _state["in_flight"] -= 1

    return StreamingResponse(_wrapped(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("QWEN_TTS_PORT", "8021")))
