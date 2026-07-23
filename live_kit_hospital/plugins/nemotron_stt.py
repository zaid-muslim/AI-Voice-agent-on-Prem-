"""
LiveKit Agents STT plugin for the local Nemotron 3.5 ASR streaming server
(nemotron_stt_server.py, run as its own standalone process).

Why this plugin exists instead of pointing livekit.plugins.openai.STT at our
server: the OpenAI-compatible plugin only ever calls the whole-file
/v1/audio/transcriptions endpoint (buffered by VAD) - it never sees the
word-by-word interim deltas our server's WebSocket produces. This plugin
talks to /v1/audio/stream directly, so real interim transcripts reach
AgentSession's turn-detection and any UI hooks in real time, not just the
end-of-utterance final - matching the "semantic turn detection" latency
goal already documented at the top of agent.py.

Server contract (see nemotron_stt_server.py):
    WS  /v1/audio/stream
        client -> server: binary frames = int16 LE PCM @ 16kHz mono
                           text frames   = {"type": "config", "language": ..}
                                           {"type": "flush"}
        server -> client: {"type": "ready"}
                           {"type": "delta", "text": "<cumulative text>"}
                           {"type": "final", "text": "<final text>"}
                           {"type": "error", "message": "..."}
    GET /health -> {"status", "model_loaded", "active_streams", ...}

NOTE ON VERSION DRIFT: this targets the livekit-agents stt plugin ABI as of
mid-2026 (stt.STT / stt.SpeechStream / SpeechEventType, per
https://docs.livekit.io/reference/python/livekit/agents/stt/). If your
installed livekit-agents version raises an ImportError on the
`livekit.agents.types` imports below, check `python -c "import
livekit.agents; print(livekit.agents.__version__)"` and tell me the version -
the exact export path for NotGivenOr/NOT_GIVEN/APIConnectOptions has moved
between releases and I'll adjust the import line.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import urllib.request
import wave

import aiohttp
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import is_given
from loguru import logger

SAMPLE_RATE = 16000
DEFAULT_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=3, retry_interval=2.0, timeout=10.0
)
DEFAULT_BASE_URL = os.environ.get("NEMOTRON_STT_BASE_URL", "ws://localhost:8010")


def _http_base(ws_base_url: str) -> str:
    return ws_base_url.replace("wss://", "https://").replace("ws://", "http://")


def _frame_to_wav_bytes(frame: rtc.AudioFrame) -> bytes:
    """Encode a single (already 16kHz-mono) AudioFrame as WAV bytes for the
    batch /v1/audio/transcriptions fallback path."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(frame.num_channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(frame.sample_rate)
        wf.writeframes(bytes(frame.data))
    return buf.getvalue()


class NemotronSTT(stt.STT):
    """Streaming STT backed by our standalone Nemotron 3.5 ASR server.

    Args:
        base_url: WebSocket base URL of nemotron_stt_server.py, e.g.
            "ws://localhost:8010". Health-checks are derived from this by
            swapping the scheme to http(s).
        language: target_lang code (e.g. "en-US", "es-US") or "auto" to let
            the model detect it. Passed to the server as a `config` message
            when each stream opens.
    """

    def __init__(
        self,
        *,
        base_url: str = os.environ.get("NEMOTRON_STT_BASE_URL", "ws://localhost:8010"),
        language: str = "auto",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return "nemotron-3.5-asr-streaming-0.6b"

    @property
    def provider(self) -> str:
        return "nvidia-local"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def load(self) -> None:
        """Eager health-check called from prewarm(), mirroring
        _warm_up_vllm()'s "surface it loudly instead" philosophy: a
        misconfigured NEMOTRON_STT_BASE_URL or a not-yet-started server
        fails here, at worker startup, instead of silently on a caller's
        first turn."""
        health_url = f"{_http_base(self._base_url)}/health"
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                body = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Nemotron STT server unreachable at {health_url} ({exc}). "
                "Start it with `python nemotron_stt_server.py` first, or "
                "fix NEMOTRON_STT_BASE_URL / system_config's stt.model. "
                "Falling back will NOT happen automatically here - fix this "
                "before real calls arrive."
            )
            return
        if not body.get("model_loaded"):
            logger.warning(
                f"Nemotron STT server at {self._base_url} is up but model not loaded yet"
            )
        else:
            logger.info(
                f"STT: Nemotron 3.5 ASR reachable ({self._base_url}), "
                f"active_streams={body.get('active_streams')}, "
                f"max_concurrent={body.get('max_concurrent_inference')}, "
                f"language={self._language}"
            )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Batch fallback via /v1/audio/transcriptions. Not the normal path -
        AgentSession uses stream() below since capabilities.streaming=True -
        but implemented for completeness (FallbackAdapter, manual
        stt.recognize() calls, etc.)."""
        frame = utils.merge_frames(buffer)
        lang = language if is_given(language) else self._language
        wav_bytes = _frame_to_wav_bytes(frame)

        session = self._ensure_session()
        form = aiohttp.FormData()
        form.add_field(
            "file", wav_bytes, filename="audio.wav", content_type="audio/wav"
        )
        form.add_field("language", lang)
        url = f"{_http_base(self._base_url)}/v1/audio/transcriptions"

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

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        lang = language if is_given(language) else self._language
        return SpeechStream(nemotron_stt=self, conn_options=conn_options, language=lang)

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class SpeechStream(stt.SpeechStream):
    """One LiveKit room's live connection to /v1/audio/stream.

    push_frame()/flush()/end_input() are called by the AgentSession's audio
    pipeline (see stt.RecognizeStream in the base class) - we never call
    those ourselves. We only implement _run(): open the WS, pump input
    frames -> server, pump server messages -> SpeechEvents.
    """

    def __init__(
        self,
        *,
        nemotron_stt: NemotronSTT,
        conn_options: APIConnectOptions,
        language: str,
    ) -> None:
        # sample_rate=SAMPLE_RATE makes the base class auto-resample every
        # pushed AudioFrame to 16kHz mono before it reaches self._input_ch -
        # exactly what our server's feed() expects, no manual resampling here.
        super().__init__(
            stt=nemotron_stt, conn_options=conn_options, sample_rate=SAMPLE_RATE
        )
        self._nemotron_stt = nemotron_stt
        self._language = language

    async def _run(self) -> None:
        session = self._nemotron_stt._ensure_session()
        ws_url = f"{self._nemotron_stt._base_url}/v1/audio/stream"

        async with session.ws_connect(ws_url) as ws:
            await ws.send_str(
                json.dumps({"type": "config", "language": self._language})
            )

            async def send_task() -> None:
                async for item in self._input_ch:
                    if isinstance(item, self._FlushSentinel):
                        await ws.send_str(json.dumps({"type": "flush"}))
                        continue
                    # item is an rtc.AudioFrame, already resampled to 16kHz mono
                    await ws.send_bytes(bytes(item.data))
                await ws.send_str(json.dumps({"type": "flush"}))
                await ws.close()

            async def recv_task() -> None:
                last_text = ""
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    mtype = payload.get("type")

                    if mtype == "delta":
                        text = payload.get("text", "")
                        if text and text != last_text:
                            last_text = text
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            language=self._language, text=text
                                        )
                                    ],
                                )
                            )
                    elif mtype == "final":
                        text = payload.get("text", "")
                        last_text = ""
                        if text:
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            language=self._language, text=text
                                        )
                                    ],
                                )
                            )
                    elif mtype == "error":
                        logger.error(
                            f"Nemotron STT server error: {payload.get('message')}"
                        )

            send = asyncio.create_task(send_task(), name="nemotron_stt_send")
            recv = asyncio.create_task(recv_task(), name="nemotron_stt_recv")
            try:
                await asyncio.gather(send, recv)
            finally:
                await utils.aio.cancel_and_wait(send, recv)
