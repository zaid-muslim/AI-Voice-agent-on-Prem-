"""
Shared faster-whisper STT service - ONE persistent, GPU-warm process
serving every LiveKit worker/room, instead of each worker process loading
its own WhisperModel copy.

WHY THIS EXISTS: before this service, plugins/whisper_stt.py loaded a
WhisperModel inside every LiveKit worker process's prewarm(). LiveKit
spawns one worker process per concurrent job, so N concurrent callers
meant N separate whisper models resident in GPU memory at once, each
competing with vLLM and TTS for the same card - the "a new STT gets
spawned per room" scaling problem. The fix is the same "one persistent,
shared, warm server" pattern already used for the LLM (vLLM) and TTS
(vLLM-Omni on PC2): load the model ONCE here, and let every worker process
talk to it over HTTP via plugins/shared_whisper_stt.py.

Concurrency across simultaneous callers comes from CTranslate2's own
`num_workers` replica pool (WhisperModel(..., num_workers=N) runs N
independent decode workers sharing the SAME loaded weights - not N
separate copies of the ~1-2GB model), bounded here by a thread pool of the
same size so requests queue instead of oversubscribing the GPU.

Run standalone:
    python server.py
Or via Docker (see docker/stt.Dockerfile + docker-compose.yml).
"""

from __future__ import annotations

import asyncio
import io
import os
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from loguru import logger

MODEL_NAME = os.environ.get("STT_MODEL", "distil-large-v3")
DEVICE = os.environ.get("STT_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8_float16")
NUM_WORKERS = int(os.environ.get("STT_NUM_WORKERS", "4"))
DEFAULT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en")
BEAM_SIZE = int(os.environ.get("STT_BEAM_SIZE", "1"))
SAMPLE_RATE = 16000

_state = {"model": None, "in_flight": 0, "total_requests": 0}
_executor = ThreadPoolExecutor(max_workers=NUM_WORKERS, thread_name_prefix="stt-worker")


def _load_model():
    from faster_whisper import WhisperModel

    logger.info(
        f"Loading faster-whisper '{MODEL_NAME}' ({DEVICE}/{COMPUTE_TYPE}, "
        f"num_workers={NUM_WORKERS}) ..."
    )
    model = WhisperModel(
        MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, num_workers=NUM_WORKERS
    )
    # Warm run so CUDA kernels/caches are paid here, not on the first real caller.
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    list(model.transcribe(silence, language=DEFAULT_LANGUAGE, beam_size=BEAM_SIZE)[0])
    logger.info("faster-whisper: ready (warm).")
    return model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _state["model"] = await asyncio.get_running_loop().run_in_executor(None, _load_model)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Shared faster-whisper STT service", lifespan=lifespan)


def _wav_bytes_to_float32(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    if sample_rate != SAMPLE_RATE:
        src_len = audio.shape[0]
        dst_len = int(round(src_len * SAMPLE_RATE / sample_rate))
        audio = np.interp(
            np.linspace(0.0, src_len - 1, dst_len, dtype=np.float64),
            np.arange(src_len, dtype=np.float64),
            audio,
        ).astype(np.float32)
    return audio


def _transcribe_sync(audio: np.ndarray, language: str) -> str:
    model = _state["model"]
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=BEAM_SIZE,
        vad_filter=False,  # the caller's own VAD already segmented this utterance
        condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


@app.get("/health")
async def health():
    return {
        "status": "ok" if _state["model"] is not None else "loading",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "num_workers": NUM_WORKERS,
        "in_flight": _state["in_flight"],
        "total_requests": _state["total_requests"],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...), language: str = Form(DEFAULT_LANGUAGE)
):
    """OpenAI-shaped transcription endpoint: multipart WAV in, {"text": ...}
    out. One utterance per call, matching how AgentSession's VAD already
    segments audio before handing it to STT - no streaming/partial results,
    same contract plugins/whisper_stt.py's in-process version had."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model still loading")

    data = await file.read()
    t0 = time.monotonic()
    _state["in_flight"] += 1
    _state["total_requests"] += 1
    try:
        loop = asyncio.get_running_loop()
        audio = _wav_bytes_to_float32(data)
        text = await loop.run_in_executor(_executor, _transcribe_sync, audio, language)
    finally:
        _state["in_flight"] -= 1
    logger.debug(f"transcribed {len(data)}B in {time.monotonic() - t0:.3f}s -> {text!r}")
    return {"text": text, "language": language}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("STT_PORT", "8020")))
