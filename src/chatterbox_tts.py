"""Custom LiveKit TTS plugin wrapping the existing Chatterbox Turbo microservice
(src/chatterbox_server.py in the original Pipeline, unchanged process/venv/HTTP API).
Non-streaming: one blocking HTTP POST per synthesis request, returning a full WAV body,
same as the original synthesize_to_wav_b64() — LiveKit wraps this in tts.StreamAdapter
(sentence-level chunking) automatically since capabilities.streaming=False.
"""
import aiohttp

from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

SAMPLE_RATE = 24000  # Chatterbox Turbo's native output rate
NUM_CHANNELS = 1


class ChatterboxTTS(tts.TTS):
    def __init__(self, url: str = "http://localhost:8766/synthesize"):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._url = url

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "ChatterboxChunkedStream":
        return ChatterboxChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChatterboxChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        tts_impl: ChatterboxTTS = self._tts  # type: ignore[assignment]
        async with aiohttp.ClientSession() as session:
            async with session.post(tts_impl._url, json={"text": self._input_text}) as resp:
                resp.raise_for_status()
                wav_bytes = await resp.read()

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=tts_impl.sample_rate,
            num_channels=tts_impl.num_channels,
            mime_type="audio/wav",
        )
        output_emitter.push(wav_bytes)
        output_emitter.flush()
