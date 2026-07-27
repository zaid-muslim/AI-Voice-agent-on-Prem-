"""
LiveKit STT plugin client for the shared faster-whisper STT service
(stt_service/server.py, run as its own standalone process/container).

WHY THIS EXISTS INSTEAD OF plugins/whisper_stt.py: FasterWhisperSTT loads
its own WhisperModel inside THIS worker process's prewarm(). LiveKit
spawns one worker process per concurrent job/room, so N concurrent
callers means N separate whisper models resident in GPU memory at once -
the "STT gets spawned per room" scaling problem. This plugin instead talks
to ONE shared, persistently-loaded faster-whisper server over HTTP - same
"one warm shared server" pattern already used for the LLM (vLLM) and TTS
(vLLM-Omni on PC2). Concurrency comes from the server's own CTranslate2
`num_workers` pool, not from spawning more model copies.

Same non-streaming contract as FasterWhisperSTT (AgentSession wraps this
with its VAD-driven StreamAdapter automatically since a `vad` is passed to
the session) - this is a drop-in replacement, just backed by a remote
server instead of an in-process model.
"""

from __future__ import annotations

import io
import json
import os
import urllib.request
import wave

import aiohttp
from livekit import rtc
from livekit.agents import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, stt, utils
from loguru import logger

DEFAULT_BASE_URL = os.environ.get("SHARED_STT_BASE_URL", "http://localhost:8020")


def _frame_to_wav_bytes(frame: rtc.AudioFrame) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(frame.num_channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(frame.sample_rate)
        wf.writeframes(bytes(frame.data))
    return buf.getvalue()


class SharedWhisperSTT(stt.STT):
    """Streaming-shaped (per LiveKit's API) but non-streaming under the
    hood, backed by the shared faster-whisper server.

    Args:
        base_url: HTTP base URL of stt_service/server.py, e.g.
            "http://localhost:8020" (same machine) or
            "http://<stt-host>:8020" (dedicated STT box, same idea as
            PC2's TTS split).
        language: language code passed to the server per-request.
    """

    def __init__(
        self, *, base_url: str = DEFAULT_BASE_URL, language: str = "en"
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return "shared-faster-whisper"

    @property
    def provider(self) -> str:
        return "self-hosted"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def load(self) -> None:
        """Eager health-check called from prewarm() - mirrors the
        "surface a misconfigured/down server loudly at worker startup"
        philosophy used for the PC2 TTS reachability check. The actual
        whisper model is loaded ONCE by stt_service/server.py itself, not
        by this client - this call is deliberately cheap."""
        health_url = f"{self._base_url}/health"
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Shared STT service unreachable at {health_url} ({exc}). Start "
                "it with `python stt_service/server.py` (or the stt-service "
                "Docker container) first, or fix SHARED_STT_BASE_URL. Falling "
                "back will NOT happen automatically here - fix this before "
                "real calls arrive."
            )
            return
        logger.info(
            f"STT: shared faster-whisper reachable ({self._base_url}), "
            f"model={body.get('model')}, num_workers={body.get('num_workers')}, "
            f"in_flight={body.get('in_flight')}, total_requests={body.get('total_requests')}"
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frame = utils.merge_frames(buffer)
        lang = language or self._language
        wav_bytes = _frame_to_wav_bytes(frame)

        session = self._ensure_session()
        form = aiohttp.FormData()
        form.add_field(
            "file", wav_bytes, filename="audio.wav", content_type="audio/wav"
        )
        form.add_field("language", lang)
        url = f"{self._base_url}/v1/audio/transcriptions"

        async with session.post(
            url, data=form, timeout=aiohttp.ClientTimeout(total=conn_options.timeout)
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()

        text = payload.get("text", "") if isinstance(payload, dict) else str(payload)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=lang, text=text)],
        )

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
