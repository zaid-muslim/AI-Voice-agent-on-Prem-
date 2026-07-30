"""
Tests for llm_plugin.py's GemmaDirectAudioLLM - the single-call,
audio-native reply path being validated as a faster replacement for the
current two-call (Gemma-transcribe + text-only-LLM) live pipeline.

This is the file that answers the real question at stake: does
tool-calling work reliably when the request is audio-conditioned instead
of text-conditioned? Unit tests cover the request/response plumbing
against a fake HTTP session; the live tests are the actual verification -
real audio, a real hospital-style tool schema, checking the model calls
the RIGHT tool with SENSIBLE arguments, and separately that it does NOT
call a tool when the caller's question doesn't warrant one.

Run:
    pytest direct_audio_agent/tests/test_llm_plugin.py -v
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemma_audio_client as gac  # noqa: E402
import llm_plugin  # noqa: E402

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectionError, llm  # noqa: E402
from dataclasses import replace as _dc_replace  # noqa: E402

NO_RETRY_CONN_OPTIONS = _dc_replace(DEFAULT_API_CONNECT_OPTIONS, max_retry=0)

FIXTURES = Path(__file__).parent / "fixtures"
TOOL_CALL_WAV = FIXTURES / "utt_tool_call_check_availability.wav"
GENERAL_QUESTION_WAV = FIXTURES / "utt_general_question.wav"


def _vllm_reachable(host: str = "127.0.0.1", port: int = 8000, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class _FakeAudioSource:
    def __init__(self, wav_bytes: bytes | None):
        self.last_turn_wav_bytes = wav_bytes


# --------------------------------------------------------------------------
# _attach_audio_to_last_user_message - pure function, no network
# --------------------------------------------------------------------------
def test_attach_audio_appends_to_existing_text_content():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello there"},
    ]
    llm_plugin._attach_audio_to_last_user_message(messages, b"FAKEWAV")

    last = messages[-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], list)
    assert last["content"][0] == {"type": "text", "text": "hello there"}
    assert last["content"][1]["type"] == "input_audio"
    assert last["content"][1]["input_audio"]["format"] == "wav"


def test_attach_audio_with_no_prior_text():
    messages = [{"role": "user", "content": ""}]
    llm_plugin._attach_audio_to_last_user_message(messages, b"FAKEWAV")
    assert messages[-1]["content"] == [
        {
            "type": "input_audio",
            "input_audio": {
                "data": gac.wav_bytes_to_base64(b"FAKEWAV"),
                "format": "wav",
            },
        }
    ]


def test_attach_audio_targets_the_last_user_message_not_the_first():
    messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "a reply"},
        {"role": "user", "content": "second turn"},
    ]
    llm_plugin._attach_audio_to_last_user_message(messages, b"X")
    assert messages[0]["content"] == "first turn"  # untouched
    assert isinstance(messages[2]["content"], list)


def test_strip_leaked_thinking_channel_removes_real_leaked_prefix():
    # Exact string captured live (2026-07-29) from the real vLLM server,
    # real RiversideReceptionist tools, real prompt, NO audio involved -
    # see llm_plugin.py's comment above _THINKING_CHANNEL_RE.
    leaked = (
        "<|channel>thought\n<channel|>I can help you with that, Sarah. "
        "Do you have the date and time of your appointment handy?"
    )
    cleaned = llm_plugin._strip_leaked_thinking_channel(leaked)
    assert cleaned == "I can help you with that, Sarah. Do you have the date and time of your appointment handy?"


def test_strip_leaked_thinking_channel_noop_on_clean_content():
    clean = "This is a perfectly normal reply."
    assert llm_plugin._strip_leaked_thinking_channel(clean) == clean


def test_strip_leaked_thinking_channel_handles_none():
    assert llm_plugin._strip_leaked_thinking_channel(None) is None


def test_strip_leaked_thinking_channel_falls_back_if_nothing_left():
    only_thought = "<|channel>thought\n<channel|>"
    assert llm_plugin._strip_leaked_thinking_channel(only_thought) == only_thought


def test_attach_audio_raises_without_any_user_message():
    messages = [{"role": "system", "content": "no users here"}]
    with pytest.raises(APIConnectionError, match="no user message"):
        llm_plugin._attach_audio_to_last_user_message(messages, b"X")


# --------------------------------------------------------------------------
# _ThinkingChannelStreamFilter - streaming-safe leak stripping
# --------------------------------------------------------------------------
def test_stream_filter_passes_clean_content_straight_through():
    f = llm_plugin._ThinkingChannelStreamFilter()
    assert f.push("Hello ") == "Hello "
    assert f.push("there!") == "there!"


def test_stream_filter_strips_leak_split_across_many_small_fragments():
    # The real leaked string, fed one/two characters at a time - the
    # worst case for a buffering filter, and the realistic case for a
    # real SSE stream (tokens arrive in small pieces).
    leaked_then_real = "<|channel>thought\n<channel|>Hello there!"
    f = llm_plugin._ThinkingChannelStreamFilter()
    out = ""
    for i in range(0, len(leaked_then_real), 2):
        out += f.push(leaked_then_real[i : i + 2])
    assert out == "Hello there!"


def test_stream_filter_leak_and_close_in_the_same_single_fragment():
    f = llm_plugin._ThinkingChannelStreamFilter()
    out = f.push("<|channel>thought\n<channel|>Hi")
    assert out == "Hi"


def test_stream_filter_never_emits_extra_or_missing_text_for_clean_stream():
    # A clean (non-leaked) stream that happens to start with characters
    # overlapping _OPEN's prefix ("<") must still come through byte-exact,
    # not get eaten by the ambiguous-prefix buffering.
    f = llm_plugin._ThinkingChannelStreamFilter()
    parts = ["<", "3 ", "you're", " great!"]
    out = "".join(f.push(p) for p in parts)
    assert out == "<3 you're great!"


# --------------------------------------------------------------------------
# _ToolCallAccumulator - reassembling fragmented tool-call deltas
# --------------------------------------------------------------------------
def test_accumulator_reassembles_arguments_across_fragments():
    acc = llm_plugin._ToolCallAccumulator()
    assert acc.push([{"index": 0, "id": "c1", "function": {"name": "book", "arguments": ""}}]) == []
    assert acc.push([{"index": 0, "function": {"arguments": '{"x": '}}]) == []
    assert acc.push([{"index": 0, "function": {"arguments": "1}"}}]) == []
    [call] = acc.finish()
    assert call.name == "book"
    assert call.call_id == "c1"
    assert call.arguments == '{"x": 1}'


def test_accumulator_flushes_previous_call_when_a_new_one_starts():
    acc = llm_plugin._ToolCallAccumulator()
    acc.push([{"index": 0, "id": "c1", "function": {"name": "first", "arguments": "{}"}}])
    completed = acc.push([{"index": 1, "id": "c2", "function": {"name": "second", "arguments": ""}}])
    assert len(completed) == 1
    assert completed[0].name == "first"
    [second] = acc.finish()
    assert second.name == "second"


def test_accumulator_finish_with_no_call_in_progress_returns_empty():
    acc = llm_plugin._ToolCallAccumulator()
    assert acc.finish() == []


# --------------------------------------------------------------------------
# GemmaDirectAudioLLM.chat() / _run() plumbing - fake HTTP, no network
# --------------------------------------------------------------------------
class _FakeStreamContent:
    """Stands in for aiohttp's resp.content (a StreamReader) - async
    iteration yields raw bytes lines, matching how GemmaDirectAudioLLM
    reads SSE events."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for line in self._lines:
            yield line.encode("utf-8")


def _sse(events: list[dict]) -> list[str]:
    """Builds SSE `data: ...` lines from a list of chunk dicts, ending
    with the standard [DONE] sentinel - the shape GemmaDirectAudioLLM's
    streaming _run() actually parses."""
    lines = [f"data: {json.dumps(e)}" for e in events]
    lines.append("data: [DONE]")
    return lines


class _FakeResponse:
    def __init__(self, status: int, body_text: str = "", sse_lines: list[str] | None = None):
        self.status = status
        self._body_text = body_text
        self.content = _FakeStreamContent(sse_lines or [])

    async def text(self) -> str:
        return self._body_text

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_json: dict | None = None
        self.closed = False

    def post(self, url: str, *, json: dict, timeout=None) -> _FakeResponse:
        self.last_json = json
        return self._response

    async def close(self) -> None:
        self.closed = True


def _chat_and_drain(
    gemma_llm: llm_plugin.GemmaDirectAudioLLM,
    ctx: llm.ChatContext,
    tools: list,
    conn_options=DEFAULT_API_CONNECT_OPTIONS,
) -> list[llm.ChatChunk]:
    """llm.LLMStream.__init__ schedules a background asyncio task the
    moment it's constructed, so .chat() itself must run inside a live
    event loop - not just the draining of its results."""

    async def _run():
        stream = gemma_llm.chat(chat_ctx=ctx, tools=tools, conn_options=conn_options)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        return chunks

    return asyncio.run(_run())


def test_run_sends_audio_and_parses_plain_text_reply():
    # Two separate content deltas, like a real SSE stream - proves this
    # is genuinely streamed (multiple chunks), not one big blocking call
    # dressed up as SSE.
    sse = _sse(
        [
            {"id": "x", "choices": [{"delta": {"content": "Sure, "}, "finish_reason": None}]},
            {
                "id": "x",
                "choices": [{"delta": {"content": "one moment."}, "finish_reason": "stop"}],
            },
            {"id": "x", "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}},
        ]
    )
    fake_session = _FakeSession(_FakeResponse(200, sse_lines=sse))
    audio_source = _FakeAudioSource(b"FAKEWAV")
    gemma_llm = llm_plugin.GemmaDirectAudioLLM(audio_source=audio_source, model="gemma-4-12b")
    gemma_llm._session = fake_session  # inject fake, bypass real aiohttp session creation

    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="hi")
    chunks = _chat_and_drain(gemma_llm, ctx, [])

    content_chunks = [c for c in chunks if c.delta is not None]
    assert len(content_chunks) == 2, "expected two separate streamed content chunks"
    assert "".join(c.delta.content for c in content_chunks) == "Sure, one moment."
    assert all(c.delta.tool_calls == [] for c in content_chunks)

    usage_chunks = [c for c in chunks if c.usage is not None]
    assert usage_chunks[0].usage.prompt_tokens == 100

    assert fake_session.last_json["model"] == "gemma-4-12b"
    assert fake_session.last_json["stream"] is True
    user_msg = fake_session.last_json["messages"][-1]
    assert any(p.get("type") == "input_audio" for p in user_msg["content"])


def test_run_parses_tool_calls_from_response():
    # Realistic fragmentation: name+id arrive once, arguments dribble in
    # across several deltas, finish_reason arrives on its own chunk -
    # this is the shape _ToolCallAccumulator exists to reassemble.
    sse = _sse(
        [
            {
                "id": "x",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "function": {"name": "check_availability", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "x",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"department": '}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "x",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"cardiology"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {"id": "x", "choices": [], "usage": {"prompt_tokens": 140, "completion_tokens": 12, "total_tokens": 152}},
        ]
    )
    fake_session = _FakeSession(_FakeResponse(200, sse_lines=sse))
    audio_source = _FakeAudioSource(b"FAKEWAV")
    gemma_llm = llm_plugin.GemmaDirectAudioLLM(audio_source=audio_source)
    gemma_llm._session = fake_session

    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="check cardiology")
    chunks = _chat_and_drain(gemma_llm, ctx, [])

    tool_call_chunks = [c for c in chunks if c.delta is not None and c.delta.tool_calls]
    assert len(tool_call_chunks) == 1, "fragments should reassemble into exactly one call"
    call = tool_call_chunks[0].delta.tool_calls[0]
    assert call.name == "check_availability"
    assert call.call_id == "call_abc"
    assert call.arguments == '{"department": "cardiology"}'


def test_run_raises_when_no_audio_available():
    audio_source = _FakeAudioSource(None)  # no turn audio stashed
    # audio_wait_secs=0 - keep this negative case fast; the real wait
    # (_AUDIO_WAIT_TIMEOUT_SECS) is covered by
    # test_run_waits_for_late_audio below.
    gemma_llm = llm_plugin.GemmaDirectAudioLLM(audio_source=audio_source, audio_wait_secs=0)
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="hi")

    with pytest.raises(Exception, match="no audio available"):
        _chat_and_drain(gemma_llm, ctx, [], conn_options=NO_RETRY_CONN_OPTIONS)


def test_run_waits_for_late_audio():
    """Reproduces the real bug found live (2026-07-29): STT can still be
    mid-flight when the LLM plugin's _run() is first invoked for the same
    turn - _run() should poll instead of raising immediately. Audio is
    stashed from a background thread (not an asyncio task) since
    _chat_and_drain owns its own event loop via asyncio.run()."""
    import threading
    import time as _time

    sse = _sse(
        [{"id": "x", "choices": [{"delta": {"content": "hi there"}, "finish_reason": "stop"}]}]
    )
    audio_source = _FakeAudioSource(None)
    fake_session = _FakeSession(_FakeResponse(200, sse_lines=sse))
    gemma_llm = llm_plugin.GemmaDirectAudioLLM(audio_source=audio_source, audio_wait_secs=2)
    gemma_llm._session = fake_session

    def _stash_after_delay() -> None:
        _time.sleep(0.05)
        audio_source.last_turn_wav_bytes = b"FAKEWAV"

    threading.Thread(target=_stash_after_delay, daemon=True).start()

    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="hi")
    chunks = _chat_and_drain(gemma_llm, ctx, [], conn_options=NO_RETRY_CONN_OPTIONS)

    assert any(c.delta is not None and c.delta.content for c in chunks)


def test_run_raises_gemma_audio_error_on_non_200():
    fake_session = _FakeSession(_FakeResponse(500, '{"error":"boom"}'))
    audio_source = _FakeAudioSource(b"FAKEWAV")
    gemma_llm = llm_plugin.GemmaDirectAudioLLM(audio_source=audio_source)
    gemma_llm._session = fake_session
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="hi")

    with pytest.raises(Exception, match="HTTP 500"):
        _chat_and_drain(gemma_llm, ctx, [], conn_options=NO_RETRY_CONN_OPTIONS)


# --------------------------------------------------------------------------
# Live: real tool schema, real audio, real vLLM server - the actual
# validation this module exists to provide before agent.py adopts it.
# --------------------------------------------------------------------------
def _make_check_availability_tool():
    @llm.function_tool
    async def check_availability(department: str, date: str | None = None) -> dict:
        """Check open appointment slots for a hospital department.

        Args:
            department: Department name as the caller said it.
            date: Optional day in YYYY-MM-DD. Omit to see all upcoming slots.
        """
        return {"department": department, "date": date, "slots": []}

    return check_availability


def test_live_tool_call_triggered_by_relevant_audio():
    if not _vllm_reachable():
        pytest.skip("vLLM not reachable at 127.0.0.1:8000 - skipping live check")
    if not TOOL_CALL_WAV.exists():
        pytest.skip(f"fixture missing: {TOOL_CALL_WAV}")

    audio_source = _FakeAudioSource(TOOL_CALL_WAV.read_bytes())
    gemma_llm = llm_plugin.GemmaDirectAudioLLM(audio_source=audio_source)
    ctx = llm.ChatContext()
    ctx.add_message(
        role="system",
        content=(
            "You are a hospital receptionist. Use check_availability whenever "
            "the caller asks about appointment slots for a department."
        ),
    )
    ctx.add_message(role="user", content="")  # audio replaces this at request time
    tool = _make_check_availability_tool()

    chunks = _chat_and_drain(gemma_llm, ctx, [tool])
    content_chunk = next(c for c in chunks if c.delta is not None)

    assert len(content_chunk.delta.tool_calls) == 1, (
        f"expected a check_availability tool call from audio asking about "
        f"cardiology availability, got content={content_chunk.delta.content!r} "
        f"tool_calls={content_chunk.delta.tool_calls}"
    )
    call = content_chunk.delta.tool_calls[0]
    assert call.name == "check_availability"
    assert "cardiolog" in call.arguments.lower()


def test_live_no_tool_call_for_unrelated_audio():
    if not _vllm_reachable():
        pytest.skip("vLLM not reachable at 127.0.0.1:8000 - skipping live check")
    if not GENERAL_QUESTION_WAV.exists():
        pytest.skip(f"fixture missing: {GENERAL_QUESTION_WAV}")

    audio_source = _FakeAudioSource(GENERAL_QUESTION_WAV.read_bytes())
    gemma_llm = llm_plugin.GemmaDirectAudioLLM(audio_source=audio_source)
    ctx = llm.ChatContext()
    ctx.add_message(
        role="system",
        content=(
            "You are a hospital receptionist. Use check_availability ONLY when "
            "the caller asks about appointment slots for a department. The "
            "caller here is asking about hospital hours and parking - answer "
            "briefly, do not call any tool."
        ),
    )
    ctx.add_message(role="user", content="")
    tool = _make_check_availability_tool()

    chunks = _chat_and_drain(gemma_llm, ctx, [tool])
    content_chunk = next(c for c in chunks if c.delta is not None)

    assert content_chunk.delta.tool_calls == [], (
        f"expected no tool call for an hours/parking question, got "
        f"{content_chunk.delta.tool_calls}"
    )
    assert content_chunk.delta.content
