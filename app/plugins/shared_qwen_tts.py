"""
LiveKit TTS plugin client for the shared Qwen3-TTS service
(tts_service/server.py, run as its own standalone process). Same "one
warm shared server, talked to over HTTP" pattern as
plugins/shared_whisper_stt.py - see that file's docstring for the general
shape, and tts_service/server.py's docstring for why this exists (2-3x
faster time-to-first-audio than the remote PC2 path, measured on this
project's own hardware - see README §3/§4).

Reads newline-delimited JSON from a streaming HTTP response - the exact
wire format tts_service/server.py documents. Unlike
plugins/qwen_tts.py's QwenSubprocessTTS (one shared stdin/stdout pipe,
so requests need id-tagging + stale-line filtering to survive
interruption), each HTTP request here gets its own private response
stream - interruption is just "stop reading and let the connection
close," nothing to filter.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os

import aiohttp
from livekit.agents import APIConnectionError, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, tts
from loguru import logger

DEFAULT_BASE_URL = os.environ.get("SHARED_TTS_BASE_URL", "http://localhost:8021")
DEFAULT_SPEAKER = os.environ.get("QWEN_SPEAKER", "aiden")
DEFAULT_LANGUAGE = os.environ.get("QWEN_LANGUAGE", "English")
DEFAULT_SAMPLE_RATE = 24000  # matches tts_service/server.py's model output; overridden per-response if it differs


class SharedQwenTTS(tts.TTS):
    """Args mirror plugins/shared_whisper_stt.py's SharedWhisperSTT shape.

    Args:
        base_url: HTTP base URL of tts_service/server.py, e.g.
            "http://localhost:8021" (same machine) or a dedicated TTS
            box's URL, same idea as SHARED_STT_BASE_URL / QWEN_OMNI_BASE_URL.
        speaker, language, chunk_size: forwarded to the service per-request;
            same meaning as QwenSubprocessTTS's constructor args.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        speaker: str = DEFAULT_SPEAKER,
        language: str = DEFAULT_LANGUAGE,
        chunk_size: int = 8,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=1,
        )
        self._base_url = base_url.rstrip("/")
        self._speaker = speaker
        self._language = language
        self._chunk_size = chunk_size
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return "shared-qwen3-tts-0.6b"

    @property
    def provider(self) -> str:
        return "self-hosted"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def load(self) -> None:
        """Eager health-check called from prewarm() - same
        surface-a-down-server-loudly-at-startup philosophy as
        SharedWhisperSTT.load() / _qwen_omni_reachable(). The actual model
        is loaded ONCE by tts_service/server.py itself, not by this
        client - this call is deliberately cheap."""
        import urllib.request

        health_url = f"{self._base_url}/health"
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Shared Qwen-TTS service unreachable at {health_url} ({exc}). Start "
                "it with `python tts_service/server.py` first, or fix "
                "SHARED_TTS_BASE_URL. Falling back will NOT happen "
                "automatically here - fix this before real calls arrive."
            )
            return
        logger.info(
            f"TTS: shared Qwen3-TTS reachable ({self._base_url}), "
            f"model={body.get('model_id')}, status={body.get('status')}"
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "_SharedQwenChunkedStream":
        return _SharedQwenChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class _SharedQwenChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: SharedQwenTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._qtts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        q = self._qtts
        session = q._ensure_session()
        payload = {
            "text": self.input_text,
            "speaker": q._speaker,
            "language": q._language,
            "chunk_size": q._chunk_size,
        }
        url = f"{q._base_url}/v1/synthesize"

        initialized = False
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout)
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        raise APIConnectionError(f"Shared Qwen-TTS error: {data['error']}")
                    if data.get("done"):
                        break
                    if "audio_b64" in data:
                        if not initialized:
                            initialized = True
                            output_emitter.initialize(
                                request_id=resp.headers.get("x-request-id", ""),
                                sample_rate=data.get("sample_rate", DEFAULT_SAMPLE_RATE),
                                num_channels=1,
                                mime_type="audio/pcm",
                            )
                        output_emitter.push(base64.b64decode(data["audio_b64"]))

                if initialized:
                    output_emitter.flush()
        except aiohttp.ClientError as exc:
            raise APIConnectionError(f"Shared Qwen-TTS request failed: {exc}") from exc
        except asyncio.CancelledError:
            # Just let the connection close - the server notices via
            # request.is_disconnected() and stops the generation thread.
            # No separate cancel message needed (see module docstring).
            raise
