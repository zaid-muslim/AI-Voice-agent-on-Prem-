"""
gemma_audio_client.py - direct-audio HTTP client for the shared vLLM server.

Sends raw PCM audio straight into Gemma 4 12B Unified's native audio
understanding (an `input_audio` chat-completion content part, OpenAI
Realtime-API-compatible), talking to the SAME already-running vLLM
process the existing STT+LLM pipeline uses (app/main.py, port 8000).
This module never touches app/, models/, system_config.json, or
vllm_manager.py - it is purely a caller of the server that was already
there, using a capability that server was already flagged for
(`--limit-mm-per-prompt '{"audio": 1}'` in scripts/run_vllm.sh and
docker/docker-compose.yml) but that turned out not to actually work
until two missing Python packages were installed. See README.md's
"Accountability log" for the full verification trail.

CONTRACT, verified two ways - reading
models/gemma-4-12b-w4a16/processor_config.json AND a real call against
the live server (2026-07-29, see README.md):
  - sampling_rate: 16000 Hz, mono (Gemma4UnifiedAudioFeatureExtractor).
    audio pushed in at a different rate is resampled by this module
    BEFORE sending, not left for the server to guess about.
  - audio_ms_per_token=40, audio_seq_length=750 -> 750 * 0.040 = 30.0s
    is the hard ceiling for ONE input_audio clip.
  - `--limit-mm-per-prompt '{"audio": 1}'` means exactly ONE audio clip
    per prompt - multi-turn history must be carried as TEXT, never as
    a second audio clip in the same request.
  - requires the vllm[audio] extras (librosa, soundfile, av) - NOT
    present in the stock vllm/vllm-openai:latest image. Installed into
    the running container manually; see README.md if this env is ever
    rebuilt from the base image again, the same install (+ restart) is
    needed once more, ideally made permanent in docker/docker-compose.yml.
  - ~7% of audio-conditioned calls come back with an empty/null content
    despite nonzero completion_tokens (a real, measured server-side
    quirk - see EMPTY_CONTENT_MAX_RETRIES below). This module retries
    automatically when that specific signature is detected.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np

SAMPLE_RATE = 16_000
AUDIO_MS_PER_TOKEN = 40
AUDIO_SEQ_LENGTH = 750
MAX_AUDIO_SECONDS = AUDIO_SEQ_LENGTH * AUDIO_MS_PER_TOKEN / 1000.0  # 30.0s

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "gemma-4-12b"
DEFAULT_TIMEOUT_SECS = 30.0

# A real, measured quirk (2026-07-29, this server, vllm-0.26.0): ~7% of
# audio-conditioned calls come back with content=null and EVERY other
# message field (function_call, reasoning, refusal) also null, despite
# completion_tokens being nonzero (14 tokens for a normal reply, 18-19
# for the empty case) and finish_reason="stop" - i.e. the model
# genuinely generated tokens, but none of them surfaced anywhere in the
# response. 20/20 equivalent TEXT-ONLY calls on the same server, same
# turn, came back clean - this is specific to the audio-conditioned
# path, most likely an interaction between the audio decode and the
# server's --tool-call-parser gemma4 (enabled for tool-calling support
# even though these particular calls pass no tools). Root cause lives in
# vLLM/the parser, not in this client - retrying is a real, verified
# mitigation (a retried call has never reproduced the same empty result
# twice in testing here), not a guess.
EMPTY_CONTENT_MAX_RETRIES = 2

TRANSCRIBE_SYSTEM_PROMPT = (
    "Transcribe the user's audio exactly, verbatim, in the language it was "
    "spoken. Output ONLY the transcript text - no commentary, no preamble, "
    "no quotation marks."
)


class GemmaAudioError(Exception):
    """Any failure talking to the audio-capable vLLM endpoint: connection
    refused, timeout, non-200, audio too long, or a malformed response."""


@dataclass
class AudioTurnResult:
    text: str
    latency_secs: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    ttft_secs: float | None = None
    """Time to first streamed token - None only if the server returned
    zero content chunks (e.g. a pure tool-call-only response, or the
    empty-content quirk EMPTY_CONTENT_MAX_RETRIES defends against)."""


# --------------------------------------------------------------------------
# Audio prep: raw PCM -> 16kHz mono int16 -> WAV bytes -> base64
# --------------------------------------------------------------------------
def resample_pcm16_mono(
    pcm: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """int16 mono PCM at src_rate -> int16 mono PCM at dst_rate. Linear
    interpolation, same approach app/plugins/whisper_stt.py uses for its
    own 16kHz resample - phone-quality speech carries nothing useful
    above 8kHz, so this is not a fidelity compromise for this use case."""
    if pcm.dtype != np.int16:
        raise ValueError(f"expected int16 PCM, got {pcm.dtype}")
    if src_rate == dst_rate:
        return pcm
    src_len = pcm.shape[0]
    dst_len = int(round(src_len * dst_rate / src_rate))
    resampled = np.interp(
        np.linspace(0.0, src_len - 1, dst_len, dtype=np.float64),
        np.arange(src_len, dtype=np.float64),
        pcm.astype(np.float64),
    )
    return resampled.astype(np.int16)


def pcm16_to_wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """int16 mono PCM samples -> an in-memory WAV file (bytes)."""
    if pcm.dtype != np.int16:
        raise ValueError(f"expected int16 PCM, got {pcm.dtype}")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_duration_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def wav_bytes_to_base64(wav_bytes: bytes) -> str:
    return base64.b64encode(wav_bytes).decode("ascii")


def _merge_tool_call_fragment(accumulated: list[dict[str, Any]], fragment: dict[str, Any]) -> None:
    """Merges one streamed OpenAI-format tool_call delta fragment into
    `accumulated`, keyed by index - same fragmentation shape
    llm_plugin.py's _ToolCallAccumulator reassembles for the live
    tool-calling path, but producing a plain dict list here since this
    client returns one final AudioTurnResult rather than a stream of
    framework ChatChunk events (this client is used for the cheap
    transcribe-only call and for benchmark.py's comparisons, not for
    driving live tool execution - see llm_plugin.py for that)."""
    index = fragment.get("index", 0)
    while len(accumulated) <= index:
        accumulated.append(
            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
        )
    entry = accumulated[index]
    if fragment.get("id"):
        entry["id"] = fragment["id"]
    fn = fragment.get("function") or {}
    if fn.get("name"):
        entry["function"]["name"] = fn["name"]
    if fn.get("arguments"):
        entry["function"]["arguments"] += fn["arguments"]


# Real, measured, pre-existing server quirk (see llm_plugin.py's
# _strip_leaked_thinking_channel for the full accountability trail):
# Gemma 4's thinking-channel tokens leak into content whenever tools are
# offered but not called - not audio-specific, reproduced with plain
# text too. This client accumulates the full streamed text before
# returning (unlike llm_plugin.py, which must filter incrementally
# since it streams ChatChunks straight to the framework), so a
# whole-string regex is correct and sufficient here.
_THINKING_CHANNEL_RE = re.compile(r"<\|channel>.*?<channel\|>\s*", re.DOTALL)


def _strip_leaked_thinking_channel(content: str) -> str:
    if not content:
        return content
    stripped = _THINKING_CHANNEL_RE.sub("", content)
    return stripped if stripped.strip() else content


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class GemmaAudioClient:
    """Thin async client for direct-audio chat completions against the
    shared vLLM server. Owns no server lifecycle - purely a caller."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_secs: float = DEFAULT_TIMEOUT_SECS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_secs = timeout_secs
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "GemmaAudioClient":
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    def _require_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise GemmaAudioError(
                "GemmaAudioClient used outside 'async with' and no session "
                "was supplied at construction - either use it as a context "
                "manager or pass session=<your aiohttp.ClientSession>."
            )
        return self._session

    async def _chat(
        self,
        *,
        system: str,
        wav_bytes: bytes,
        max_tokens: int,
        extra_messages_before_audio: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_empty_retries: int = EMPTY_CONTENT_MAX_RETRIES,
    ) -> AudioTurnResult:
        """Retries on the empty-content quirk documented at
        EMPTY_CONTENT_MAX_RETRIES's definition - measured live at ~7% of
        audio-conditioned calls (0/20 for equivalent text-only calls on
        the same server), so silently returning an empty transcript/reply
        that often would be a real reliability regression, not a rare
        edge case worth ignoring."""
        attempt = 0
        while True:
            result = await self._chat_once(
                system=system,
                wav_bytes=wav_bytes,
                max_tokens=max_tokens,
                extra_messages_before_audio=extra_messages_before_audio,
                tools=tools,
                tool_choice=tool_choice,
            )
            is_suspicious_empty = (
                not result.text.strip()
                and (result.completion_tokens or 0) > 0
            )
            if not is_suspicious_empty or attempt >= max_empty_retries:
                return result
            attempt += 1

    async def _chat_once(
        self,
        *,
        system: str,
        wav_bytes: bytes,
        max_tokens: int,
        extra_messages_before_audio: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> AudioTurnResult:
        duration_s = wav_duration_seconds(wav_bytes)
        if duration_s > MAX_AUDIO_SECONDS:
            raise GemmaAudioError(
                f"audio is {duration_s:.1f}s, exceeds the "
                f"{MAX_AUDIO_SECONDS:.0f}s limit for one input_audio clip "
                f"(audio_seq_length={AUDIO_SEQ_LENGTH} * "
                f"audio_ms_per_token={AUDIO_MS_PER_TOKEN}ms, from "
                f"models/gemma-4-12b-w4a16/processor_config.json). Split "
                f"into shorter turns."
            )

        messages: list[dict] = [{"role": "system", "content": system}]
        if extra_messages_before_audio:
            messages.extend(extra_messages_before_audio)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": wav_bytes_to_base64(wav_bytes),
                            "format": "wav",
                        },
                    }
                ],
            }
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Same ban as llm_plugin.py's live payload - see that file's
            # comment. Without it this client can also run away into the
            # reasoning channel until max_tokens, on transcribe_audio()
            # and respond_to_audio() alike (and therefore in benchmark.py
            # too, silently inflating measured latency).
            "logit_bias": {"100": -100},
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        # Streamed (not one blocking call) so the caller's own
        # ttft_secs is real, not "however long the full completion
        # took" - GemmaDirectAudioSTT (stt_plugin.py) uses this for its
        # transcribe_audio() call specifically so the transcript is
        # available to the safety gate as soon as generation finishes
        # streaming, not after an extra request/response round trip's
        # worth of buffering. Same SSE parsing shape as
        # llm_plugin.py's _GemmaDirectAudioLLMStream (that file's
        # tool-call accumulation isn't needed here - transcribe_audio()
        # never passes tools, and respond_to_audio()'s tool_calls, when
        # present, arrive as one bounded response for this client's use
        # case - benchmarking/comparison, not live tool execution).
        session = self._require_session()
        t0 = time.monotonic()
        first_token_at: float | None = None
        text_parts: list[str] = []
        tool_calls_raw: list[dict[str, Any]] = []
        response_id = "gemma-direct-audio"
        usage: dict[str, Any] = {}
        finish_reason: str | None = None

        try:
            async with session.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._timeout_secs),
            ) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    raise GemmaAudioError(
                        f"vLLM returned HTTP {resp.status} for a direct-audio "
                        f"call: {body_text[:500]}"
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
                    if event.get("usage"):
                        usage = event["usage"]

                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]

                    content = delta.get("content")
                    if content:
                        if first_token_at is None:
                            first_token_at = time.monotonic() - t0
                        text_parts.append(content)

                    for tc in delta.get("tool_calls") or []:
                        _merge_tool_call_fragment(tool_calls_raw, tc)
        except aiohttp.ClientError as exc:
            raise GemmaAudioError(
                f"could not reach vLLM at {self._base_url}: {exc}"
            ) from exc
        except asyncio.TimeoutError as exc:
            raise GemmaAudioError(
                f"direct-audio call timed out after {self._timeout_secs}s"
            ) from exc
        latency = time.monotonic() - t0

        text = _strip_leaked_thinking_channel("".join(text_parts))
        raw_body = {
            "id": response_id,
            "choices": [
                {
                    "message": {"content": text, "tool_calls": tool_calls_raw or None},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        return AudioTurnResult(
            text=text,
            latency_secs=latency,
            ttft_secs=first_token_at,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw=raw_body,
        )

    async def transcribe_audio(
        self, wav_bytes: bytes, *, max_tokens: int = 200
    ) -> AudioTurnResult:
        """Cheap, transcript-only call - Gemma's native audio understanding
        used purely as an ASR replacement (no separate Whisper model in
        the loop at all). This is what GemmaDirectAudioSTT (stt_plugin.py)
        calls so Gemma's audio understanding can slot into LiveKit's
        existing, proven turn-detection/safety-gate machinery unchanged."""
        return await self._chat(
            system=TRANSCRIBE_SYSTEM_PROMPT, wav_bytes=wav_bytes, max_tokens=max_tokens
        )

    async def respond_to_audio(
        self,
        wav_bytes: bytes,
        *,
        system_prompt: str,
        history: list[dict] | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 300,
    ) -> AudioTurnResult:
        """The fully audio-native path: raw audio -> understanding -> reply
        in ONE inference call, no separate transcription pass at all. Not
        wired into agent.py's live turn loop yet - see README.md's
        "Scoped out for v1" section for why tool-calling-over-audio needs
        more validation before it drives real bookings. Used today by
        benchmark.py to measure what this path can offer."""
        return await self._chat(
            system=system_prompt,
            wav_bytes=wav_bytes,
            extra_messages_before_audio=history,
            tools=tools,
            max_tokens=max_tokens,
        )
