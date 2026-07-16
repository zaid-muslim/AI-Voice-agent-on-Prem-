"""
faster-whisper STT for LiveKit Agents, running locally on the same GPU as
vLLM and the Qwen TTS worker.

Design notes:
- Non-streaming STT (capabilities.streaming=False). AgentSession wraps
  non-streaming STT with its VAD-driven StreamAdapter automatically when a
  `vad` is provided to the session - same net behavior as Pipecat's
  WhisperSTTService (VAD segments the utterance, whisper transcribes it).
- distil-large-v3 stays the default for the same reason as before: decoder
  depth dominates Whisper latency (2 decoder layers here vs turbo's 4), so
  "bigger name" models are not faster for short phone utterances.
- Transcription runs in a single-lane thread executor: faster-whisper's
  model object is not safe for concurrent transcribe() calls, and one lane
  also prevents GPU contention spikes against vLLM mid-turn.
- load() must be called from the worker's prewarm hook so the first caller
  never pays model-load time (same principle as the TTS/RAG/vLLM warm-ups).

VERIFY AGAINST YOUR INSTALLED livekit-agents (see README "Verify before
trusting"): the exact _recognize_impl signature and SpeechData fields have
drifted across 1.x minor versions.
"""

from __future__ import annotations

import asyncio
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

WHISPER_SAMPLE_RATE = 16000


class FasterWhisperSTT(stt.STT):
    def __init__(
        self,
        *,
        model: str = "distil-large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "en",
        beam_size: int = 1,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._model = None
        # Single lane on purpose - see module docstring.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Blocking model load + one warm inference. Call from prewarm."""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # deferred: heavy import

        logger.info(
            f"FasterWhisperSTT: loading '{self._model_name}' "
            f"({self._device}/{self._compute_type}) ..."
        )
        self._model = WhisperModel(
            self._model_name, device=self._device, compute_type=self._compute_type
        )
        # Warm run so CUDA kernels/caches are paid before the first caller.
        silence = np.zeros(WHISPER_SAMPLE_RATE // 2, dtype=np.float32)
        list(
            self._model.transcribe(
                silence, language=self._language, beam_size=self._beam_size
            )[0]
        )
        logger.info("FasterWhisperSTT: ready (warm).")

    # ------------------------------------------------------------- transcribe
    def _transcribe_sync(self, audio: np.ndarray) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=False,  # the session's Silero VAD already segmented this
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    @staticmethod
    def _buffer_to_float32(buffer: utils.AudioBuffer) -> np.ndarray:
        """Merge frames -> mono float32 @ 16 kHz. Linear-interp resample is
        fine here: whisper computes its own mel features and phone speech
        carries nothing useful above 8 kHz anyway."""
        frame = rtc.combine_audio_frames(buffer)
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        if frame.sample_rate != WHISPER_SAMPLE_RATE:
            src_len = audio.shape[0]
            dst_len = int(round(src_len * WHISPER_SAMPLE_RATE / frame.sample_rate))
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
