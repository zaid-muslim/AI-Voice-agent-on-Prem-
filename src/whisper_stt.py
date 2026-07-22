"""Custom LiveKit STT plugin that POSTs utterance audio to the shared Whisper microservice
(src/whisper_server.py) instead of holding a WhisperModel in-process. Non-streaming: LiveKit's VAD
segments an utterance, then hands the whole buffer to _recognize_impl() once. Mirrors the
ChatterboxTTS HTTP-client pattern (src/chatterbox_tts.py) — the per-call job process now holds no
STT model, so many concurrent calls share one Whisper instead of one Whisper loaded per call.
"""
import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.language import LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


class WhisperSTT(stt.STT):
    def __init__(self, url: str = "http://localhost:8768/transcribe", language: str = "en") -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._url = url
        self._language = language

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        # int16 PCM -> float32 in [-1, 1] just to measure the peak; the bytes sent to the service
        # stay int16 (the service does the same float conversion on its side).
        audio_array = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        peak = float(np.abs(audio_array).max()) if audio_array.size else 0.0
        if peak < 0.01:
            # Silence — short-circuit without a network round trip (kept from the in-process
            # version), returning the same empty FINAL_TRANSCRIPT the model would have produced.
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language=LanguageCode(self._language), text="")],
            )

        lang = language or self._language
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                params={"language": lang, "sample_rate": str(frame.sample_rate)},
                data=bytes(frame.data),
                headers={"Content-Type": "application/octet-stream"},
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(language=LanguageCode(self._language), text=payload.get("text", ""))
            ],
        )
