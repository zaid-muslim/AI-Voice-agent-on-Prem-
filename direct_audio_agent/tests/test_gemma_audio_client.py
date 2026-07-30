"""
Tests for gemma_audio_client.py.

Two tiers, run together by default:
  1. Unit tests (fake aiohttp session, no network) - encoding correctness,
     request shape, and every error path (non-200, connect failure,
     audio-too-long, missing session).
  2. A live integration test against the REAL shared vLLM server
     (127.0.0.1:8000) using a real synthesized WAV fixture with a known
     sentence - this is the actual "does Gemma really understand raw
     audio" check, not a mock standing in for one. It SKIPS (not fails)
     if the server isn't reachable, so this file still runs clean in an
     environment without the GPU stack up.

No pytest-asyncio dependency - this repo's existing convention
(tests/test_booking.py) is plain sync test functions driving async code
via asyncio.run(), so that's what's used here too rather than adding a
new package to the shared venv for this alone.

Run:
    pytest direct_audio_agent/tests/test_gemma_audio_client.py -v
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import socket
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemma_audio_client as gac  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_WAV = FIXTURES / "sample_cardiology_utterance.wav"
# Piper-synthesized locally (models/piper/en_US-lessac-medium.onnx) saying
# exactly this sentence - see README.md's accountability log.
SAMPLE_WAV_EXPECTED_WORDS = ("cardiology", "appointment", "tomorrow")


def _vllm_reachable(host: str = "127.0.0.1", port: int = 8000, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------
# Audio encoding helpers - no network, no asyncio involved
# --------------------------------------------------------------------------
def test_pcm16_to_wav_bytes_roundtrip():
    pcm = (np.sin(np.linspace(0, 10 * np.pi, 16_000)) * 10_000).astype(np.int16)
    wav_bytes = gac.pcm16_to_wav_bytes(pcm, sample_rate=16_000)

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16_000
        assert w.getnframes() == len(pcm)
        decoded = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    np.testing.assert_array_equal(decoded, pcm)


def test_pcm16_to_wav_bytes_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="int16"):
        gac.pcm16_to_wav_bytes(np.zeros(100, dtype=np.float32))


def test_wav_duration_seconds():
    pcm = np.zeros(32_000, dtype=np.int16)  # 2.0s @ 16kHz
    wav_bytes = gac.pcm16_to_wav_bytes(pcm, sample_rate=16_000)
    assert gac.wav_duration_seconds(wav_bytes) == pytest.approx(2.0)


def test_resample_pcm16_mono_changes_length_not_dtype():
    pcm_22050 = np.zeros(22_050, dtype=np.int16)  # 1.0s @ 22050Hz
    resampled = gac.resample_pcm16_mono(pcm_22050, src_rate=22_050, dst_rate=16_000)
    assert resampled.dtype == np.int16
    assert resampled.shape[0] == 16_000  # 1.0s @ 16kHz


def test_resample_pcm16_mono_noop_when_rates_match():
    pcm = np.arange(1000, dtype=np.int16)
    assert gac.resample_pcm16_mono(pcm, src_rate=16_000, dst_rate=16_000) is pcm


def test_wav_bytes_to_base64_is_decodable():
    pcm = np.zeros(100, dtype=np.int16)
    wav_bytes = gac.pcm16_to_wav_bytes(pcm)
    encoded = gac.wav_bytes_to_base64(wav_bytes)
    assert base64.b64decode(encoded) == wav_bytes


# --------------------------------------------------------------------------
# Fake aiohttp plumbing - lets us test GemmaAudioClient's HTTP handling
# without a real server or a new test dependency (no aioresponses).
# --------------------------------------------------------------------------
class _FakeStreamContent:
    """Stands in for aiohttp's resp.content (a StreamReader) - async
    iteration yields raw bytes lines, matching how GemmaAudioClient
    reads SSE events."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for line in self._lines:
            yield line.encode("utf-8")


def _sse_lines_from_full_response(body: dict) -> list[str]:
    """Converts an OpenAI-style non-streaming response body into an
    equivalent single-chunk SSE stream - the request shape
    GemmaAudioClient actually sends now (stream=True), built this way
    so existing test fixtures (a full message dict) stay easy to write
    while exercising the real streaming parse path."""
    message = body["choices"][0]["message"]
    events = [
        {
            "id": body.get("id", "x"),
            "choices": [
                {
                    "delta": {
                        "content": message.get("content"),
                        "tool_calls": message.get("tool_calls"),
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        {"id": body.get("id", "x"), "choices": [], "usage": body.get("usage") or {}},
    ]
    return [f"data: {json.dumps(e)}" for e in events] + ["data: [DONE]"]


class _FakeResponse:
    def __init__(self, status: int, body_text: str = "", sse_lines: list[str] | None = None):
        self.status = status
        self._body_text = body_text
        if sse_lines is None and body_text:
            # Back-compat convenience: a full JSON body_text (old
            # non-streaming test fixtures) is auto-converted to an
            # equivalent single-chunk SSE stream, UNLESS body_text isn't
            # valid JSON (e.g. a deliberately malformed-response test),
            # in which case it's left for resp.text()'s non-200 path or
            # surfaces as a parse error via empty content, same as before.
            try:
                sse_lines = _sse_lines_from_full_response(json.loads(body_text))
            except (json.JSONDecodeError, KeyError, IndexError):
                sse_lines = []
        self.content = _FakeStreamContent(sse_lines or [])

    async def text(self) -> str:
        return self._body_text

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeSession:
    """Records every POST it was given and returns canned responses in
    order (repeating the last one once exhausted), or raises whatever
    `raise_on_post` is set to. `responses=[a, b]` simulates a server that
    answers differently across successive calls - used to test the
    empty-content retry in GemmaAudioClient._chat."""

    def __init__(
        self,
        response: _FakeResponse | None = None,
        responses: list[_FakeResponse] | None = None,
        raise_on_post: Exception | None = None,
    ):
        self._responses = responses if responses is not None else [response]
        self._raise_on_post = raise_on_post
        self.last_url: str | None = None
        self.last_json: dict | None = None
        self.post_count = 0

    def post(self, url: str, *, json: dict, timeout=None) -> _FakeResponse:
        self.last_url = url
        self.last_json = json
        if self._raise_on_post is not None:
            raise self._raise_on_post
        index = min(self.post_count, len(self._responses) - 1)
        self.post_count += 1
        return self._responses[index]


def _short_silence_wav(seconds: float = 0.5) -> bytes:
    pcm = np.zeros(int(gac.SAMPLE_RATE * seconds), dtype=np.int16)
    return gac.pcm16_to_wav_bytes(pcm)


def test_transcribe_audio_success_parses_response():
    async def _run():
        body = (
            '{"choices":[{"message":{"content":"hello world"}}],'
            '"usage":{"prompt_tokens":130,"completion_tokens":3}}'
        )
        session = _FakeSession(response=_FakeResponse(200, body))
        client = gac.GemmaAudioClient(session=session)
        return await client.transcribe_audio(_short_silence_wav())

    result = asyncio.run(_run())
    assert result.text == "hello world"
    assert result.prompt_tokens == 130
    assert result.completion_tokens == 3
    assert result.latency_secs >= 0


def test_retries_on_suspicious_empty_content_and_succeeds():
    """Reproduces the real, measured server quirk (see
    gemma_audio_client.EMPTY_CONTENT_MAX_RETRIES's docstring): content
    null but completion_tokens nonzero. Client should retry and return
    the good result, not the empty one."""

    async def _run():
        empty_body = (
            '{"choices":[{"message":{"content":null},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":145,"completion_tokens":18}}'
        )
        good_body = (
            '{"choices":[{"message":{"content":"hello world"},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":145,"completion_tokens":14}}'
        )
        session = _FakeSession(
            responses=[_FakeResponse(200, empty_body), _FakeResponse(200, good_body)]
        )
        client = gac.GemmaAudioClient(session=session)
        result = await client.transcribe_audio(_short_silence_wav())
        return result, session

    result, session = asyncio.run(_run())
    assert result.text == "hello world"
    assert session.post_count == 2


def test_gives_up_after_max_empty_retries_and_returns_last_empty_result():
    async def _run():
        empty_body = (
            '{"choices":[{"message":{"content":null},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":145,"completion_tokens":18}}'
        )
        session = _FakeSession(responses=[_FakeResponse(200, empty_body)])
        client = gac.GemmaAudioClient(session=session)
        result = await client.transcribe_audio(_short_silence_wav())
        return result, session

    result, session = asyncio.run(_run())
    assert result.text == ""
    assert session.post_count == gac.EMPTY_CONTENT_MAX_RETRIES + 1


def test_does_not_retry_a_legitimately_empty_zero_token_result():
    """An empty response with completion_tokens=0 (or missing) is a
    genuinely-empty generation, not the suspicious quirk - retrying that
    would just waste a call, so it should return immediately."""

    async def _run():
        body = (
            '{"choices":[{"message":{"content":""},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":145,"completion_tokens":0}}'
        )
        session = _FakeSession(response=_FakeResponse(200, body))
        client = gac.GemmaAudioClient(session=session)
        result = await client.transcribe_audio(_short_silence_wav())
        return result, session

    result, session = asyncio.run(_run())
    assert result.text == ""
    assert session.post_count == 1


def test_transcribe_audio_sends_input_audio_content_part():
    async def _run():
        body = '{"choices":[{"message":{"content":"x"}}],"usage":{}}'
        session = _FakeSession(response=_FakeResponse(200, body))
        client = gac.GemmaAudioClient(session=session, model="gemma-4-12b")
        await client.transcribe_audio(_short_silence_wav())
        return session

    session = asyncio.run(_run())
    assert session.last_json["model"] == "gemma-4-12b"
    user_msg = session.last_json["messages"][-1]
    assert user_msg["role"] == "user"
    part = user_msg["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    assert isinstance(part["input_audio"]["data"], str) and part["input_audio"]["data"]


def test_respond_to_audio_includes_tools_and_history():
    async def _run():
        body = '{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
        session = _FakeSession(response=_FakeResponse(200, body))
        client = gac.GemmaAudioClient(session=session)
        tools = [{"type": "function", "function": {"name": "check_availability"}}]
        history = [{"role": "assistant", "content": "Hi, how can I help?"}]
        await client.respond_to_audio(
            _short_silence_wav(),
            system_prompt="You are a receptionist.",
            history=history,
            tools=tools,
        )
        return session, tools, history

    session, tools, history = asyncio.run(_run())
    assert session.last_json["tools"] == tools
    assert session.last_json["messages"][0]["content"] == "You are a receptionist."
    assert session.last_json["messages"][1] == history[0]


def test_non_200_raises_gemma_audio_error_with_body():
    async def _run():
        session = _FakeSession(response=_FakeResponse(500, '{"error":"boom"}'))
        client = gac.GemmaAudioClient(session=session)
        await client.transcribe_audio(_short_silence_wav())

    with pytest.raises(gac.GemmaAudioError, match="HTTP 500"):
        asyncio.run(_run())


def test_malformed_sse_lines_are_skipped_not_raised():
    """The client streams SSE now (see _chat_once) - a garbage/
    unparseable line is simply skipped per-line (matching real SSE
    resilience: one bad chunk shouldn't blow up the whole stream), not
    treated as a fatal error. A response with ONLY garbage lines and no
    valid data therefore comes back as an empty (not exceptional)
    result - genuinely different from the old non-streaming client's
    behavior, not a regression."""
    session = _FakeSession(response=_FakeResponse(200, sse_lines=["not json", "data: also not json", "data: [DONE]"]))
    client = gac.GemmaAudioClient(session=session)
    result = asyncio.run(client.transcribe_audio(_short_silence_wav()))
    assert result.text == ""
    assert result.completion_tokens is None


def test_connection_error_wrapped_as_gemma_audio_error():
    import aiohttp

    async def _run():
        session = _FakeSession(raise_on_post=aiohttp.ClientConnectionError("refused"))
        client = gac.GemmaAudioClient(session=session)
        await client.transcribe_audio(_short_silence_wav())

    with pytest.raises(gac.GemmaAudioError, match="could not reach vLLM"):
        asyncio.run(_run())


def test_audio_over_30s_rejected_before_any_network_call():
    async def _run():
        session = _FakeSession(response=_FakeResponse(200, "{}"))
        client = gac.GemmaAudioClient(session=session)
        too_long = _short_silence_wav(seconds=gac.MAX_AUDIO_SECONDS + 1.0)
        try:
            await client.transcribe_audio(too_long)
        finally:
            assert session.last_url is None  # never even attempted the call

    with pytest.raises(gac.GemmaAudioError, match="exceeds the 30s limit"):
        asyncio.run(_run())


def test_no_session_and_not_used_as_context_manager_raises():
    async def _run():
        client = gac.GemmaAudioClient()
        await client.transcribe_audio(_short_silence_wav())

    with pytest.raises(gac.GemmaAudioError, match="used outside 'async with'"):
        asyncio.run(_run())


def test_context_manager_opens_and_closes_owned_session():
    async def _run():
        async with gac.GemmaAudioClient() as client:
            assert client._session is not None
        return client

    client = asyncio.run(_run())
    assert client._session is None


# --------------------------------------------------------------------------
# Live integration - real server, real audio, skipped if unreachable
# --------------------------------------------------------------------------
def test_live_transcribe_matches_known_sentence():
    if not _vllm_reachable():
        pytest.skip("vLLM not reachable at 127.0.0.1:8000 - skipping live check")
    if not SAMPLE_WAV.exists():
        pytest.skip(f"fixture missing: {SAMPLE_WAV}")

    async def _run():
        wav_bytes = SAMPLE_WAV.read_bytes()
        async with gac.GemmaAudioClient() as client:
            return await client.transcribe_audio(wav_bytes)

    result = asyncio.run(_run())
    lowered = result.text.lower()
    missing = [w for w in SAMPLE_WAV_EXPECTED_WORDS if w not in lowered]
    assert not missing, f"transcript missing expected words {missing}: got {result.text!r}"
    assert result.latency_secs < 30.0


def test_live_respond_to_audio_produces_relevant_reply():
    if not _vllm_reachable():
        pytest.skip("vLLM not reachable at 127.0.0.1:8000 - skipping live check")
    if not SAMPLE_WAV.exists():
        pytest.skip(f"fixture missing: {SAMPLE_WAV}")

    async def _run():
        wav_bytes = SAMPLE_WAV.read_bytes()
        async with gac.GemmaAudioClient() as client:
            return await client.respond_to_audio(
                wav_bytes,
                system_prompt="You are a hospital receptionist. Reply briefly to the caller.",
            )

    result = asyncio.run(_run())
    assert len(result.text.strip()) > 0
    assert result.prompt_tokens and result.prompt_tokens > 0
