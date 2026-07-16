"""Custom LiveKit STT plugin wrapping the same local faster-whisper model the original
Pipeline used (src/server.py's transcribe_array). Non-streaming: LiveKit's VAD segments an
utterance, then hands the whole buffer to _recognize_impl() once — matching today's
call-once-per-utterance behavior, no streaming partials needed.
"""
import asyncio

import numpy as np
from faster_whisper import WhisperModel

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.language import LanguageCode


class WhisperSTT(stt.STT):
    def __init__(self, model: WhisperModel, language: str = "en"):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._model = model
        self._language = language

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language=None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        # int16 PCM -> float32 in [-1, 1], same conversion server.py's pcm16_bytes_to_array used.
        audio_array = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0

        peak = float(np.abs(audio_array).max()) if audio_array.size else 0.0
        if peak < 0.01:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language=LanguageCode(self._language), text="")],
            )

        def _transcribe():
            segments, _ = self._model.transcribe(
                audio_array, language=self._language, vad_filter=True
            )
            return "".join(s.text for s in segments).strip()

        text = await asyncio.to_thread(_transcribe)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=LanguageCode(self._language), text=text)],
        )
