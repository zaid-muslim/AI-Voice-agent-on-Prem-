"""
NVIDIA Parakeet TDT as a LiveKit STT plugin - the SOTA local STT upgrade.

WHY PARAKEET OVER WHISPER HERE:
  - Top of the Hugging Face Open ASR leaderboard in its size class, with an
    enormous inference speed advantage (RTFx in the thousands on modern
    GPUs) - a 3-10s phone utterance transcribes in tens of milliseconds,
    which directly shrinks the end-of-turn -> LLM gap that dominates your
    perceived latency.
  - Punctuation + capitalization built in (whisper-distil is weaker here),
    which gives the semantic turn-detector cleaner input to judge
    end-of-utterance from.
CAVEATS (why whisper stays available as STT_BACKEND=whisper):
  - English-only (parakeet-tdt-0.6b-v2). Your callers may code-switch to
    Urdu mid-sentence; whisper degrades more gracefully there.
  - Requires the NeMo toolkit, a heavy install (pulls torch/lightning).
  - VRAM: ~1.5-2 GB on top of vLLM + Qwen TTS. Fits a 3090 at your 0.50
    vLLM utilization, but watch nvidia-smi on the first full run.

ARCHITECTURE: same shape as the whisper plugin - non-streaming STT that the
AgentSession segments with Silero VAD. Parakeet is so fast that VAD-chunked
"batch" transcription is effectively real-time; true streaming (cache-aware
FastConformer) is a later upgrade and pairs with the turn detector the same
way.

VERIFY AGAINST YOUR INSTALLED NeMo (see README): the transcribe() return
type changed across NeMo 2.x versions - older returns list[str], newer
returns list[Hypothesis] with a .text attribute. _extract_text() handles
both, but confirm on your box with the standalone check in the README.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger

from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    stt,
    utils,
)

PARAKEET_SAMPLE_RATE = 16000
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


class ParakeetSTT(stt.STT):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        device: str = "cuda",
        language: str = "en",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._model_name = model
        self._device = device
        self._language = language
        self._model = None
        # Single lane: one transcription at a time, no GPU contention spikes
        # against vLLM mid-turn (same rationale as the whisper plugin).
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="parakeet"
        )

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Blocking model load + one warm inference. Call from prewarm.
        First run downloads ~2.4 GB from Hugging Face; set HF_HOME if you
        want to control where."""
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr  # deferred: heavy import

        logger.info(f"ParakeetSTT: loading '{self._model_name}' ...")
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self._model_name
        )
        self._model.eval()
        if self._device == "cuda":
            self._model = self._model.cuda()
        # Warm run: CUDA kernels + graph paid before the first caller.
        self._transcribe_sync(np.zeros(PARAKEET_SAMPLE_RATE // 2, dtype=np.float32))
        logger.info("ParakeetSTT: ready (warm).")

    # ------------------------------------------------------------- transcribe
    @staticmethod
    def _extract_text(result) -> str:
        """NeMo's transcribe() return type drifted across versions:
        list[str] (older) vs list[Hypothesis] with .text (newer)."""
        if not result:
            return ""
        first = result[0]
        if isinstance(first, str):
            return first.strip()
        text = getattr(first, "text", None)
        return (text or str(first)).strip()

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        import torch

        with torch.inference_mode():
            out = self._model.transcribe([audio], batch_size=1, verbose=False)
        return self._extract_text(out)

    @staticmethod
    def _buffer_to_float32(buffer: utils.AudioBuffer) -> np.ndarray:
        """Merge frames -> mono float32 @ 16 kHz (same conversion as the
        whisper plugin; linear-interp resample is fine for speech)."""
        frame = rtc.combine_audio_frames(buffer)
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        if frame.sample_rate != PARAKEET_SAMPLE_RATE:
            src_len = audio.shape[0]
            dst_len = int(round(src_len * PARAKEET_SAMPLE_RATE / frame.sample_rate))
            audio = np.interp(
                np.linspace(0.0, src_len - 1, dst_len, dtype=np.float64),
                np.arange(src_len, dtype=np.float64),
                audio,
            ).astype(np.float32)
        return audio

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        if self._model is None:
            # Safety net if prewarm was skipped (e.g. `console` quick tests).
            await asyncio.get_running_loop().run_in_executor(self._executor, self.load)

        audio = self._buffer_to_float32(buffer)
        text = await asyncio.get_running_loop().run_in_executor(
            self._executor, self._transcribe_sync, audio
        )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(text=text, language=language or self._language)
            ],
        )
