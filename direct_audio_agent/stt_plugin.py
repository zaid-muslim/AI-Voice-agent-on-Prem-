"""
stt_plugin.py - GemmaDirectAudioSTT: a livekit-agents `stt.STT` whose
"transcription" is Gemma 4 12B Unified's own native audio understanding,
not a separate ASR model. There is no Whisper/Parakeet/Canary anywhere in
this class - the raw audio for the turn goes straight to the shared vLLM
server via gemma_audio_client.transcribe_audio().

WHY THIS SHAPE (an STT plugin) INSTEAD OF A FULLY CUSTOM TURN LOOP:
LiveKit's turn-detection state machine (VAD start/stop, the semantic
end-of-turn model, interruption handling, on_user_turn_completed / the
safety gate) is all built around STT producing `stt.SpeechEvent` text -
see livekit/agents/voice/audio_recognition.py. Reimplementing that
machinery to accept raw audio instead of text would be a much larger,
much riskier undertaking than the actual goal here (get Gemma consuming
audio directly instead of a separate ASR engine). Slotting in at the
STT layer keeps 100% of that proven machinery - semantic turn detection,
the emergency safety gate, tool-calling, CHAT_CTX_MAX_ITEMS truncation -
completely unchanged, while still genuinely eliminating the separate STT
model: Gemma sees the raw waveform, not a Whisper transcript of it.

This mirrors app/plugins/whisper_stt.py's shape closely (buffer -> mono
16kHz PCM -> transcribe) so it's a drop-in replacement for that class in
an AgentSession(stt=...) call - see agent.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import aiohttp
import numpy as np
from loguru import logger

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    stt,
    utils,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma_audio_client as gac  # noqa: E402


class GemmaDirectAudioSTT(stt.STT):
    def __init__(
        self,
        *,
        base_url: str = gac.DEFAULT_BASE_URL,
        model: str = gac.DEFAULT_MODEL,
        language: str = "en",
        timeout_secs: float = gac.DEFAULT_TIMEOUT_SECS,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._base_url = base_url
        self._model = model
        self._language = language
        self._timeout_secs = timeout_secs
        self._session: aiohttp.ClientSession | None = None
        # Side-channel for llm_plugin.GemmaDirectAudioLLM: the ONE-call
        # design needs the same raw audio this STT call just transcribed,
        # since LiveKit's ChatContext only carries the resulting TEXT
        # transcript, not the audio itself (see llm_plugin.py's docstring
        # for the full "why" - AudioContent is silently dropped by the
        # standard OpenAI-compatible chat serialization path). One
        # attribute is sufficient because AgentSession processes one
        # user turn's STT call before its LLM call, sequentially - not a
        # general-purpose queue, and not meant to be one.
        self.last_turn_wav_bytes: bytes | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    @staticmethod
    def _buffer_to_pcm16_mono_16k(buffer: utils.AudioBuffer) -> np.ndarray:
        """Same merge-then-resample approach as FasterWhisperSTT
        (app/plugins/whisper_stt.py._buffer_to_float32), kept in int16
        instead of float32 since gemma_audio_client writes WAV directly
        from int16 PCM."""
        frame = rtc.combine_audio_frames(buffer)
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        return gac.resample_pcm16_mono(
            pcm, src_rate=frame.sample_rate, dst_rate=gac.SAMPLE_RATE
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        pcm = self._buffer_to_pcm16_mono_16k(buffer)
        wav_bytes = gac.pcm16_to_wav_bytes(pcm)
        # Stash BEFORE the network call, not after: llm_plugin's stream
        # only needs this to exist by the time llm_node runs (always
        # after STT completes for the same turn), and stashing early
        # means a slow/failed transcribe call still leaves the most
        # recent audio available rather than None.
        self.last_turn_wav_bytes = wav_bytes

        session = await self._ensure_session()
        client = gac.GemmaAudioClient(
            base_url=self._base_url,
            model=self._model,
            timeout_secs=self._timeout_secs,
            session=session,
        )
        try:
            result = await client.transcribe_audio(wav_bytes)
        except gac.GemmaAudioError as exc:
            logger.error(f"GemmaDirectAudioSTT: direct-audio transcription failed: {exc}")
            raise APIConnectionError(str(exc)) from exc

        logger.debug(
            f"GemmaDirectAudioSTT: {gac.wav_duration_seconds(wav_bytes):.2f}s audio -> "
            f"{result.latency_secs:.2f}s, {result.prompt_tokens} prompt tokens -> "
            f"{result.text!r}"
        )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(text=result.text, language=language or self._language)
            ],
        )

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
