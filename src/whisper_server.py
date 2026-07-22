#!/usr/bin/env python3
"""Shared faster-whisper STT microservice.

One WhisperModel loaded once, serving every concurrent call over HTTP — the STT analog of the
Chatterbox TTS microservice (../Pipeline/src/chatterbox_server.py). This replaces loading a
separate WhisperModel inside each LiveKit job-executor process (worker.py's old prewarm()), which
put one full Whisper on the GPU per concurrent call and OOM'd the second caller. Now the per-call
process holds no STT model at all — it just POSTs utterance audio here.
"""
import asyncio
import os

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from faster_whisper import WhisperModel

# Model-load settings come from the environment (orchestrator.py sets them from
# config/models_config.json's stt entry); defaults match the values used bare-metal today.
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
DEVICE_INDEX = int(os.environ.get("WHISPER_DEVICE_INDEX", "0"))
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
# CTranslate2 runs up to this many transcriptions concurrently on the one model instance — the
# knob that lets a single shared Whisper serve multiple simultaneous calls without a per-call
# model copy.
NUM_WORKERS = int(os.environ.get("WHISPER_NUM_WORKERS", "2"))
PORT = int(os.environ.get("WHISPER_PORT", "8768"))

app = FastAPI()

print(
    f"Loading faster-whisper ({MODEL_SIZE}, {DEVICE} {COMPUTE_TYPE}, num_workers={NUM_WORKERS})...",
    flush=True,
)
model = WhisperModel(
    MODEL_SIZE, compute_type=COMPUTE_TYPE, device=DEVICE,
    device_index=DEVICE_INDEX, num_workers=NUM_WORKERS,
)
print("Whisper server ready.", flush=True)

# No lock around model.transcribe(): CTranslate2's WhisperModel is documented thread-safe and
# processes up to num_workers concurrent calls internally, so overlapping requests (dispatched to
# the event loop's default threadpool via run_in_executor) run in parallel rather than racing —
# unlike Chatterbox, whose model needed a serializing lock. If concurrent transcription ever
# proves unsafe on this stack, wrap the run_in_executor call below in an asyncio.Lock; STT of a
# short utterance is sub-second, so serializing is a cheap fallback.


def _transcribe(audio: np.ndarray, language: str) -> str:
    """Run the blocking faster-whisper transcription and join the segment texts."""
    segments, _ = model.transcribe(audio, language=language, vad_filter=True)
    return "".join(s.text for s in segments).strip()


@app.get("/health")
def health() -> dict[str, str]:
    """Readiness probe. The module-scope model load above already blocked until the model was
    ready, so a served response here means STT is ready (mirrors the Chatterbox /health contract)."""
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(request: Request) -> dict[str, str]:
    """Transcribe raw little-endian int16 PCM audio (request body) to text.

    The body is raw int16 PCM bytes; `language` and `sample_rate` are optional query params
    (`language` defaults to WHISPER_LANGUAGE). `sample_rate` is accepted for forward-compat/logging
    only — faster-whisper's feature extractor expects 16 kHz, which is what the worker delivers.
    Returns {"text": <transcript>}; an empty string for an empty body.
    """
    language = request.query_params.get("language", DEFAULT_LANGUAGE)
    pcm_bytes = await request.body()
    if not pcm_bytes:
        return {"text": ""}
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(None, _transcribe, audio, language)
    return {"text": text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
