"""
llm_plugin.py - GemmaDirectAudioLLM: a livekit-agents `llm.LLM` that
generates the REAL reply (including tool calls) from the caller's raw
audio directly, in ONE network call - not from the text transcript
GemmaDirectAudioSTT already produced.

WHY THIS EXISTS: the pipeline agent.py shipped first (STT swap only,
text-only llm_node unchanged) is CORRECT and safe, but measurably
slower than the existing Whisper+text-LLM cascade - see benchmark.py's
"direct_audio_stt" numbers (620ms vs cascade's 479ms). The reason: it
pays for a full audio-conditioned Gemma call to transcribe, THEN a full
separate text-only Gemma call to reply - two expensive calls where the
cascade only pays for one. gemma_audio_client.respond_to_audio() already
proved a single combined call is faster than the cascade
(324ms vs 479ms, see benchmark_results.json) - this module is what
wires that into the LIVE tool-calling response path, not just a
benchmark script.

WHY A SEPARATE, CHEAP TRANSCRIBE CALL STILL HAPPENS (in stt_plugin.py,
unchanged) EVEN THOUGH THIS MODULE COULD IN THEORY REPLACE IT: the
emergency safety gate (RiversideReceptionist.on_user_turn_completed,
app/main.py) runs BEFORE any LLM generation and can bypass the LLM
entirely on a match - "the LLM never sees this turn" is the whole point
of a deterministic, keyword-based safety net for a hospital receptionist.
That gate needs real transcript TEXT to run its regex match, and needs
it BEFORE deciding whether to call an LLM at all. There is no way to get
that text without SOME transcription step happening first, no matter how
the reply itself is generated - so the STT call is a floor, not
something this module tries to eliminate. What it DOES eliminate is
the SECOND, separate, full-cost generation call.

HOW THE AUDIO GETS HERE: LiveKit's ChatContext only carries the TEXT
GemmaDirectAudioSTT produced - `llm.AudioContent` exists on ChatContext,
but the standard OpenAI-compatible serialization path
(`livekit.agents.llm._provider_format.openai._to_chat_item`) silently
DROPS it (`elif isinstance(content, llm.AudioContent): pass` - checked
directly in the installed livekit-agents 1.6.6 source, not assumed).
That path is only wired up for LiveKit's Realtime API integration, not
plain chat completions. So this module reads the raw WAV bytes from a
side-channel instead: `audio_source.last_turn_wav_bytes`, set by
stt_plugin.GemmaDirectAudioSTT right before this LLM's chat() is called
for the same turn (see that file's `last_turn_wav_bytes` docstring for
why one attribute is enough - AgentSession processes STT then LLM
sequentially per turn, not concurrently).

REUSES, NOT REIMPLEMENTS, LiveKit's own OpenAI-compatible serialization:
`ChatContext.to_provider_format("openai")` for prior-turn history
(correctly threads tool_call/tool_output pairs) and
`ToolContext.parse_function_tools("openai")` for the tool schemas - the
exact same public helpers `livekit.plugins.openai.LLM` itself uses
internally. Only the LAST user message (the current turn) is touched,
to attach the input_audio content part before sending.

NOT YET LIVE in agent.py - see README.md's "Single-call live path" for
the validation plan (tool-calling-over-audio reliability) before it
replaces the current text-only llm_node there.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NotGivenOr,
    llm,
)
from livekit.agents.types import NOT_GIVEN

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma_audio_client as gac  # noqa: E402

DEFAULT_MAX_TOKENS = 400

# How long _run() will poll audio_source.last_turn_wav_bytes before giving
# up, when the paired GemmaDirectAudioSTT hasn't stashed it yet. Measured
# live (2026-07-29): on a cold first turn of a fresh job process, STT's own
# transcribe_audio() call didn't finish until ~9.3s after this LLM was first
# invoked for the same turn - well past DEFAULT_API_CONNECT_OPTIONS' ~4.1s
# total retry budget (max_retry=3, retry_interval=2.0s), so every attempt
# was raising instantly instead of actually waiting, and the turn failed
# outright ("failed to generate LLM completion after 4 attempts"). Polling
# here, inside each attempt, gives each one a real chance instead of
# wasting the framework's fixed retry cadence on immediate failures.
_AUDIO_WAIT_TIMEOUT_SECS = 5.0
_AUDIO_WAIT_POLL_INTERVAL_SECS = 0.2


class GemmaDirectAudioLLM(llm.LLM):
    def __init__(
        self,
        *,
        audio_source: Any,
        base_url: str = gac.DEFAULT_BASE_URL,
        model: str = gac.DEFAULT_MODEL,
        timeout_secs: float = gac.DEFAULT_TIMEOUT_SECS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        audio_wait_secs: float = _AUDIO_WAIT_TIMEOUT_SECS,
    ) -> None:
        """
        Args:
            audio_source: any object exposing `.last_turn_wav_bytes: bytes
                | None` - in practice the SAME GemmaDirectAudioSTT
                instance passed to AgentSession(stt=...), so both read/
                write the same turn's audio.
            audio_wait_secs: how long _run() polls audio_source before
                giving up when last_turn_wav_bytes isn't set yet - see
                _AUDIO_WAIT_TIMEOUT_SECS' module-level comment for why
                this exists. Overridable mainly so tests can keep the
                "no audio available" negative case fast.
        """
        super().__init__()
        self._audio_source = audio_source
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_secs = timeout_secs
        self._max_tokens = max_tokens
        self._audio_wait_secs = audio_wait_secs
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "vllm-gemma-direct-audio"

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "_GemmaDirectAudioLLMStream":
        return _GemmaDirectAudioLLMStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class _ToolCallAccumulator:
    """Accumulates streamed OpenAI-format tool_call delta fragments into
    complete llm.FunctionToolCall objects. A tool call's `arguments`
    string arrives split across many chunks; `name`/`id` arrive once, on
    the first fragment for that call. push() returns any call that just
    completed because a NEW call started (multi-tool-call responses);
    finish() returns whatever call was still in progress when the stream
    ended (finish_reason stop/tool_calls). Mirrors the accumulation
    pattern in livekit.agents.inference.llm.LLMStream._parse_choice
    (read as a reference, not a public API - that class isn't a
    sanctioned extension point, so this is a fresh implementation)."""

    def __init__(self) -> None:
        self._index: int | None = None
        self._call_id: str | None = None
        self._name: str | None = None
        self._arguments: str = ""

    def push(self, tool_call_deltas: list[dict]) -> list["llm.FunctionToolCall"]:
        completed = []
        for tc in tool_call_deltas:
            fn = tc.get("function") or {}
            index = tc.get("index")
            if fn.get("name"):
                if self._call_id is not None and index != self._index:
                    completed.append(self._flush())
                self._index = index
                self._call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                self._name = fn["name"]
                self._arguments = fn.get("arguments") or ""
            elif fn.get("arguments"):
                self._arguments += fn["arguments"]
        return completed

    def finish(self) -> list["llm.FunctionToolCall"]:
        return [self._flush()] if self._call_id is not None else []

    def _flush(self) -> "llm.FunctionToolCall":
        call = llm.FunctionToolCall(
            name=self._name or "", arguments=self._arguments, call_id=self._call_id or ""
        )
        self._index = self._call_id = self._name = None
        self._arguments = ""
        return call


class _ThinkingChannelStreamFilter:
    """Streaming-safe version of _strip_leaked_thinking_channel: content
    now arrives in small fragments, so a whole-string regex can't run
    once at the end - by the time the leak would be visible, it's
    already been sent to TTS one fragment at a time. Buffers only long
    enough to decide whether the stream is starting with the leaked
    `<|channel>thought...<channel|>` prefix (see
    _strip_leaked_thinking_channel's comment for the real, reproduced
    bug this defends against); once it's clear either way (leak or not),
    every subsequent push() passes content straight through with zero
    buffering overhead - the common case (no leak) costs nothing beyond
    the first ~10 characters of the very first chunk."""

    _OPEN = "<|channel>"
    _CLOSE = "<channel|>"

    def __init__(self) -> None:
        self._state = "checking"  # "checking" | "stripping" | "clean"
        self._buffer = ""

    def push(self, content: str) -> str:
        if self._state == "clean":
            return content

        self._buffer += content

        if self._state == "stripping":
            idx = self._buffer.find(self._CLOSE)
            if idx == -1:
                return ""  # still inside the leaked thought, keep buffering
            self._state = "clean"
            rest = self._buffer[idx + len(self._CLOSE) :]
            self._buffer = ""
            return rest

        # state == "checking"
        if self._OPEN in self._buffer:
            self._state = "stripping"
            self._buffer = self._buffer.split(self._OPEN, 1)[1]
            idx = self._buffer.find(self._CLOSE)
            if idx == -1:
                return ""
            self._state = "clean"
            rest = self._buffer[idx + len(self._CLOSE) :]
            self._buffer = ""
            return rest

        if len(self._buffer) < len(self._OPEN) and self._OPEN.startswith(self._buffer):
            return ""  # still ambiguous - could become a leak on the next fragment

        # Definitely not a leak (diverged from _OPEN) - flush everything
        # buffered so far and stop checking for the rest of this stream.
        self._state = "clean"
        out = self._buffer
        self._buffer = ""
        return out


class _GemmaDirectAudioLLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_instance: GemmaDirectAudioLLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._gemma_llm = llm_instance

    async def _run(self) -> None:
        """Streams the response (SSE, stream=True) instead of waiting for
        one full completion - see README.md's "Streaming the reply" for
        why: measurement showed RAG/booking lookups cost ~1-4ms (not the
        bottleneck), while a real tool-calling turn pays for TWO
        sequential LLM calls of 0.5-1.5s each. Waiting for each one's
        FULL text before TTS can start wastes exactly the token-generation
        time that's actually available to stream - current best practice
        (verified via search, not assumed) is flushing content to TTS as
        tokens arrive, typically saving 300-800ms of perceived latency.
        Tool-call argument fragments are accumulated across chunks the
        same way livekit.agents.inference.llm.LLMStream._parse_choice
        does for the standard OpenAI-compatible path (reused as a
        reference pattern, not copied verbatim, since that class isn't a
        public extension point)."""
        wav_bytes = self._gemma_llm._audio_source.last_turn_wav_bytes
        if wav_bytes is None:
            # Poll rather than fail immediately - see _AUDIO_WAIT_TIMEOUT_SECS'
            # definition above for why the framework's own retry-with-backoff
            # isn't enough on its own for a slow/cold-starting paired STT call.
            deadline = time.monotonic() + self._gemma_llm._audio_wait_secs
            while wav_bytes is None and time.monotonic() < deadline:
                await asyncio.sleep(_AUDIO_WAIT_POLL_INTERVAL_SECS)
                wav_bytes = self._gemma_llm._audio_source.last_turn_wav_bytes
        if wav_bytes is None:
            raise APIConnectionError(
                "GemmaDirectAudioLLM: no audio available for this turn - "
                "the paired GemmaDirectAudioSTT must run (and set "
                "last_turn_wav_bytes) before this LLM is called for the "
                "same turn."
            )

        # Reuse LiveKit's OWN OpenAI-compatible serialization (public API,
        # not a private internal) for prior-turn history and tool
        # schemas - identical to what livekit.plugins.openai.LLM uses.
        messages, _ = self._chat_ctx.to_provider_format(format="openai")
        tool_schemas: list[dict[str, Any]] = []
        if self._tools:
            tool_schemas = llm.ToolContext(self._tools).parse_function_tools(
                "openai", strict=False
            )

        _attach_audio_to_last_user_message(messages, wav_bytes)

        payload: dict[str, Any] = {
            "model": self._gemma_llm._model,
            "max_tokens": self._gemma_llm._max_tokens,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Ban the <|channel> token (id 100) outright - confirmed live
            # (2026-07-29) that without this, real direct-audio calls hit
            # completion_tokens=max_tokens on EVERY turn with ttft 5-7s:
            # the same reasoning-channel runaway bug fixed for the cascade
            # pipeline in app/main.py's openai.LLM(...) call, just never
            # ported to this separate client. _ThinkingChannelStreamFilter
            # below only hides the leaked tokens from playback - it does
            # NOT stop the model from spending the whole token budget
            # generating them, which is what was actually slow.
            "logit_bias": {"100": -100},
        }
        if tool_schemas:
            payload["tools"] = tool_schemas

        session = await self._gemma_llm._ensure_session()
        response_id = "gemma-direct-audio"
        acc = _ToolCallAccumulator()
        leak_filter = _ThinkingChannelStreamFilter()

        try:
            async with session.post(
                f"{self._gemma_llm._base_url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._gemma_llm._timeout_secs),
            ) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    raise APIConnectionError(
                        f"GemmaDirectAudioLLM: vLLM returned HTTP {resp.status}: "
                        f"{body_text[:500]}"
                    )

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    response_id = event.get("id", response_id)
                    usage = event.get("usage")
                    choices = event.get("choices") or []

                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        finish_reason = choice.get("finish_reason")

                        for finished_call in acc.push(delta.get("tool_calls") or []):
                            self._event_ch.send_nowait(
                                llm.ChatChunk(
                                    id=response_id,
                                    delta=llm.ChoiceDelta(
                                        role="assistant", tool_calls=[finished_call]
                                    ),
                                )
                            )

                        content = delta.get("content")
                        if content:
                            visible = leak_filter.push(content)
                            if visible:
                                self._event_ch.send_nowait(
                                    llm.ChatChunk(
                                        id=response_id,
                                        delta=llm.ChoiceDelta(
                                            role="assistant", content=visible
                                        ),
                                    )
                                )

                        if finish_reason in ("tool_calls", "stop"):
                            for finished_call in acc.finish():
                                self._event_ch.send_nowait(
                                    llm.ChatChunk(
                                        id=response_id,
                                        delta=llm.ChoiceDelta(
                                            role="assistant", tool_calls=[finished_call]
                                        ),
                                    )
                                )

                    if usage:
                        self._event_ch.send_nowait(
                            llm.ChatChunk(
                                id=response_id,
                                usage=llm.CompletionUsage(
                                    completion_tokens=usage.get("completion_tokens", 0),
                                    prompt_tokens=usage.get("prompt_tokens", 0),
                                    total_tokens=usage.get("total_tokens", 0),
                                ),
                            )
                        )
        except aiohttp.ClientError as exc:
            raise APIConnectionError(
                f"GemmaDirectAudioLLM: could not reach vLLM at "
                f"{self._gemma_llm._base_url}: {exc}"
            ) from exc


def _attach_audio_to_last_user_message(messages: list[dict], wav_bytes: bytes) -> None:
    """Mutates `messages` (OpenAI-format, from to_provider_format) in
    place: converts the LAST user message's content into a multi-part
    list carrying its existing text (if any) plus an input_audio part
    for the current turn's raw audio. Raises if there's no user message
    to attach to - that would mean this LLM was called without a real
    user turn, a caller bug worth surfacing loudly rather than silently
    sending audio-less requests."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            existing = msg.get("content")
            parts: list[dict[str, Any]] = []
            if isinstance(existing, str) and existing:
                parts.append({"type": "text", "text": existing})
            elif isinstance(existing, list):
                parts.extend(existing)
            parts.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": gac.wav_bytes_to_base64(wav_bytes),
                        "format": "wav",
                    },
                }
            )
            msg["content"] = parts
            return
    raise APIConnectionError(
        "GemmaDirectAudioLLM: no user message found in chat_ctx to attach audio to"
    )


# Real bug, confirmed 2026-07-29 with plain TEXT requests (no audio
# involved at all) using the REAL RiversideReceptionist tools + real
# system prompt against this exact vLLM server: whenever tools are
# offered and the model decides NOT to call one, its thinking-channel
# preamble leaks verbatim into `content` instead of being routed to the
# separate `reasoning` field the response schema already has for this.
# Root cause: no --reasoning-parser is configured in scripts/run_vllm.sh
# / docker/docker-compose.yml, and this vLLM install has ZERO reasoning
# parsers registered at all (checked directly:
# vllm.reasoning.ReasoningParserManager.reasoning_parsers == {}) - not
# something fixable from here without touching the shared vLLM config,
# which is out of scope for this module. This is a PRE-EXISTING gap in
# app/main.py's own production pipeline too (same server, same flags,
# same tools) - reported in README.md, not fixed there without explicit
# go-ahead. This strip is a defensive net for THIS module's output only.
_THINKING_CHANNEL_RE = re.compile(r"<\|channel>.*?<channel\|>\s*", re.DOTALL)


def _strip_leaked_thinking_channel(content: str | None) -> str | None:
    if not content:
        return content
    stripped = _THINKING_CHANNEL_RE.sub("", content)
    return stripped if stripped.strip() else content
