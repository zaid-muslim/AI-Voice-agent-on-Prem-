"""
Riverside General voice receptionist - LiveKit Agents port, SOTA revision.

Architecture (what changed vs the Pipecat version, and what didn't):

  Pipecat                                 LiveKit (this file)
  ---------------------------------------------------------------------------
  Pipeline([...]) frame graph          -> AgentSession(vad, stt, llm, tts,
                                          turn_detection)
  SafetyGateProcessor (FrameProcessor  -> on_user_turn_completed() hook:
    swallowing TranscriptionFrame)        runs run_safety_gate() on the final
                                          transcript, speaks the escalation
                                          via session.say(), and raises
                                          StopResponse() so the LLM NEVER
                                          generates for that turn.
  faster-whisper STT                   -> Any of 4 STT engines (whisper,
                                          Parakeet, Canary), chosen via
                                          system_config.json (edit through
                                          the dev console at :7871), not
                                          a fixed env var.
  tuned VAD stop_secs (0.3, risky)     -> Silero VAD (presence) + local
                                          inference.TurnDetector v1-mini
                                          (semantic end-of-turn), dynamic
                                          endpointing, preemptive LLM gen.
  register_direct_function(...)        -> @function_tool methods below,
                                          signatures matching the REAL
                                          hospital_core/booking.py exactly
                                          (incl. department/time narrowing
                                          for ambiguous cancels).
  tool_filler decorator                -> run_with_filler() inside each tool
  manual FastAPI signaling server      -> DELETED. livekit-server does all
                                          signaling; token_server.py only
                                          mints join tokens + serves the UI.
  vLLM/RAG/TTS warm-ups at startup     -> prewarm_fnc + entrypoint warm-ups

Run (after README setup):
    python agent.py dev        # hot-reload dev worker
    python agent.py console    # terminal mode: local mic/speaker, no server
    python agent.py start      # production mode
"""

from __future__ import annotations

import asyncio
import os
import socket
from urllib.parse import urlparse

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    RoomInputOptions,
    RunContext,
    StopResponse,
    function_tool,
    inference,
    llm,
    metrics,
)
from livekit.plugins import openai, silero

import compat
import latency_log
import system_config
from helpers import (
    DEFAULT_FILLERS,
    push_ui,
    require_real_livekit_credentials,
    run_with_filler,
)
from plugins.qwen_tts import QwenSubprocessTTS
from prompts import GREETING_INSTRUCTIONS, build_system_prompt

load_dotenv()

# Fail closed at import time: the LiveKit CLI runtime (agents.cli.run_app,
# invoked at the bottom of this file) reads these same env vars to connect
# this worker to livekit-server - better to refuse to start than to
# silently connect (or worse, run) with the public devkey/secret pair.
require_real_livekit_credentials()

# --- shared config -----------------------------------------------------
# INFRASTRUCTURE env vars: where things live / how to reach them. These
# rarely change and are set once in .env.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
PARAKEET_MODEL = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "distil-large-v3")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "300"))

# Shared faster-whisper STT service (stt_service/server.py) - ONE persistent,
# GPU-warm process serving every worker/room, instead of each worker process
# loading its own WhisperModel copy. Same "one shared server" split as
# QWEN_OMNI_BASE_URL below, just for STT. Defaults to localhost since it
# typically runs on the same box as the agent (PC1); point it at a dedicated
# STT box the same way QWEN_OMNI_BASE_URL points at PC2 if you split it out.
SHARED_STT_BASE_URL = os.environ.get("SHARED_STT_BASE_URL", "http://localhost:8020")

# Qwen3-TTS via vLLM-Omni, running persistently on PC2 (the "TTS box"),
# NOT on this machine. Every room/session's TTS calls go over the network
# to this one shared, already-warm server - no per-call subprocess spawn.
# Override via .env if PC2's IP or the served model path ever changes.
QWEN_OMNI_MODEL = os.environ.get(
    "QWEN_OMNI_MODEL",
    "/home/nauyan/voice-agent-pipeline/models/qwen3-tts/Qwen3-TTS-12Hz-0.6B-CustomVoice",
)
QWEN_OMNI_BASE_URL = os.environ.get(
    "QWEN_OMNI_BASE_URL", "http://192.168.18.56:8091/v1"
)

QWEN_OMNI_VOICE = os.environ.get("QWEN_OMNI_VOICE", "Aiden")

# Shared Qwen3-TTS 0.6B service (tts_service/server.py) - same "one warm
# shared server" idea as SHARED_STT_BASE_URL, but for the LOCAL Qwen model
# instead of PC2's remote 1.7B one. Measured on this project's own
# hardware: ~245ms to first audio vs ~500-700ms for the PC2 path, at the
# cost of using THIS machine's GPU budget instead of PC2's - see README
# §3/§4. Opt-in via system_config.json's tts.engine="qwen_shared", not the
# default - see _make_tts() below.
SHARED_TTS_BASE_URL = os.environ.get("SHARED_TTS_BASE_URL", "http://localhost:8021")


def _shared_stt_reachable(base_url: str, timeout: float = 2.0) -> bool:
    """Same fast TCP-level reachability check as _qwen_omni_reachable
    below, applied to the shared STT service - run ONCE per worker process
    in prewarm(), not per call. A real HTTP health check still happens
    right after via SharedWhisperSTT.load()."""
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as exc:
        logger.warning(
            f"Shared STT service ({host}:{port}) not reachable ({exc}) - "
            "this worker process will use in-process faster-whisper instead."
        )
        return False


def _qwen_omni_reachable(base_url: str, timeout: float = 2.0) -> bool:
    """Fast TCP-level reachability check for PC2 (the remote Qwen-Omni TTS
    server), run ONCE per worker process in prewarm() - not per call, since
    a worker process serves calls sequentially and PC2's up/down state
    isn't expected to flip mid-process. Deliberately just a socket connect,
    not a full HTTP request - we only need "is anything listening on that
    port" before deciding which TTS engine to construct. The real
    synthesis correctness is still verified afterwards by
    _warm_up_qwen_omni()'s actual HTTP call."""
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as exc:
        logger.warning(
            f"PC2 (Qwen-Omni TTS, {host}:{port}) not reachable ({exc}) - "
            "this worker process will use local Qwen subprocess TTS instead."
        )
        return False


def _shared_qwen_tts_reachable(base_url: str, timeout: float = 2.0) -> bool:
    """Same fast TCP-level reachability check as _shared_stt_reachable,
    applied to the shared local Qwen-TTS service. A real HTTP health check
    still happens right after via SharedQwenTTS.load()."""
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as exc:
        logger.warning(
            f"Shared Qwen-TTS service ({host}:{port}) not reachable ({exc}) - "
            "falling back to Qwen (remote, PC2)."
        )
        return False


# SELECTION config: WHICH of the available models/backends is active RIGHT
# NOW. This comes from system_config.py instead of a fixed env var, so a
# developer's choice from the dev UI takes effect - see system_config.py's
# docstring for the exact "takes effect on next call" contract. Read fresh
# in prewarm() and entrypoint() below, never cached at module level, so a
# change is never more than one prewarm-cycle / one call away.


def _make_stt(stt_cfg: dict):
    """Chosen by stt_cfg = {"engine": ..., "model": ...} from
    system_config (schema v2 - see that module's docstring for why this
    changed from a flat backend string).

    engine="parakeet" fallback chain, in order:
      1. PARAKEET_PYTHON is set -> subprocess plugin. This is the path that
         actually works when the main agent's venv is Python 3.12, since
         NeMo's ASR extras currently fail to build there (see README) -
         Parakeet runs isolated in its own 3.10/3.11 venv instead.
      2. PARAKEET_PYTHON unset but `nemo` importable HERE -> in-process
         plugin (only works if your main venv itself is 3.10/3.11).
      3. Neither -> faster-whisper, the proven fallback.

    engine="canary" needs CANARY_PYTHON/CANARY_WORKER (same NeMo venv as
    Parakeet typically works - see plugins/canary_worker.py). Falls back
    to whisper if that infra isn't set, same pattern as parakeet.

    engine="whisper" (or anything unrecognized) -> faster-whisper, using
    stt_cfg["model"] as the specific checkpoint name (e.g.
    "distil-large-v3" or "large-v3" - same plugin code, different size)."""
    engine = stt_cfg.get("engine", "whisper_shared")
    model = stt_cfg.get("model", WHISPER_MODEL)

    if engine == "whisper_shared":
        if _shared_stt_reachable(SHARED_STT_BASE_URL):
            from plugins.shared_whisper_stt import SharedWhisperSTT

            logger.info(f"STT: shared faster-whisper service ({SHARED_STT_BASE_URL})")
            return SharedWhisperSTT(base_url=SHARED_STT_BASE_URL)
        logger.warning(
            f"stt engine=whisper_shared but {SHARED_STT_BASE_URL} isn't reachable - "
            "falling back to in-process faster-whisper for this worker process."
        )

    elif engine == "parakeet":
        if os.environ.get("PARAKEET_PYTHON"):
            from plugins.parakeet_stt_subprocess import ParakeetSubprocessSTT

            logger.info(
                f"STT: Parakeet (subprocess, venv={os.environ['PARAKEET_PYTHON']}, model={model})"
            )
            return ParakeetSubprocessSTT()
        try:
            from plugins.parakeet_stt import ParakeetSTT

            logger.info(f"STT: Parakeet (in-process, {model})")
            return ParakeetSTT(model=model)
        except ImportError as exc:
            logger.warning(
                f"stt engine=parakeet but NeMo isn't importable here ({exc}) and "
                "PARAKEET_PYTHON isn't set - falling back to faster-whisper."
            )

    elif engine == "canary":
        if os.environ.get("CANARY_PYTHON") and os.environ.get("CANARY_WORKER"):
            from plugins.canary_stt_subprocess import CanarySubprocessSTT

            logger.info(
                f"STT: Canary (subprocess, venv={os.environ['CANARY_PYTHON']}, model={model})"
            )
            return CanarySubprocessSTT(model=model)
        logger.warning(
            "stt engine=canary but CANARY_PYTHON/CANARY_WORKER aren't set "
            "in .env - falling back to faster-whisper."
        )

    from plugins.whisper_stt import FasterWhisperSTT

    logger.info(f"STT: faster-whisper ({model})")
    return FasterWhisperSTT(model=model)


def _make_tts(tts_cfg: dict):
    """Chosen by tts_cfg = {"engine": ..., "model": ...} from
    system_config (schema v2). "model" means different things per engine:
    Qwen -> speaker name, Chatterbox -> unused (single default voice),
    Kokoro -> voice pack name, Piper -> .onnx file path.

    Each candidate engine falls back to Qwen (remote) if its required env
    vars aren't set - same graceful-degradation pattern used throughout
    this project rather than crashing the call. The "qwen_omni" / "qwen"
    (default) path is itself preceded by a PC2 reachability check up in
    entrypoint(), which may rewrite the engine to
    "qwen_local_subprocess" BEFORE this function ever sees it - see that
    call site's comment for why."""
    engine = tts_cfg.get("engine", "qwen_omni")
    model = tts_cfg.get("model", "")

    if engine == "chatterbox":
        if os.environ.get("CHATTERBOX_PYTHON") and os.environ.get("CHATTERBOX_WORKER"):
            from plugins.chatterbox_tts import ChatterboxSubprocessTTS

            logger.info(
                f"TTS: Chatterbox (subprocess, venv={os.environ['CHATTERBOX_PYTHON']}) "
                f"- NOTE: non-streaming synthesis, see plugins/chatterbox_tts.py"
            )
            return ChatterboxSubprocessTTS()
        logger.warning(
            "tts engine=chatterbox but CHATTERBOX_PYTHON/CHATTERBOX_WORKER "
            "aren't set in .env - falling back to Qwen (remote)."
        )

    elif engine == "kokoro":
        if os.environ.get("KOKORO_PYTHON") and os.environ.get("KOKORO_WORKER"):
            from plugins.kokoro_tts import KokoroSubprocessTTS

            voice = model or "af_heart"
            logger.info(
                f"TTS: Kokoro (subprocess, venv={os.environ['KOKORO_PYTHON']}, voice={voice})"
            )
            return KokoroSubprocessTTS(voice=voice)
        logger.warning(
            "tts engine=kokoro but KOKORO_PYTHON/KOKORO_WORKER aren't set "
            "in .env - falling back to Qwen (remote)."
        )

    elif engine == "piper":
        if os.environ.get("PIPER_PYTHON") and os.environ.get("PIPER_WORKER"):
            from plugins.piper_tts import PiperSubprocessTTS

            model_path = model or os.environ.get("PIPER_MODEL_PATH", "")
            logger.info(
                f"TTS: Piper (subprocess, venv={os.environ['PIPER_PYTHON']}, "
                f"voice={model_path}) - GPL-3.0 licensed, verify this fits "
                f"your deployment (see plugins/piper_worker.py)"
            )
            return PiperSubprocessTTS(model_path=model_path)
        logger.warning(
            "tts engine=piper but PIPER_PYTHON/PIPER_WORKER aren't set "
            "in .env - falling back to Qwen (remote)."
        )

    elif engine == "qwen_shared":
        # Shared, persistently-warm local Qwen3-TTS 0.6B service (see
        # tts_service/server.py) - the fast local alternative to the
        # default remote Qwen-Omni (PC2) path. ~245ms to first audio vs
        # ~500-700ms for PC2, measured on this project's own hardware
        # (README §3/§4), at the cost of GPU budget on THIS machine
        # instead of PC2's. Opt-in, not the default - the "3 concurrent
        # callers" capacity budget (README §10) assumed PC2 handling TTS.
        if _shared_qwen_tts_reachable(SHARED_TTS_BASE_URL):
            from plugins.shared_qwen_tts import SharedQwenTTS

            logger.info(f"TTS: shared Qwen3-TTS service ({SHARED_TTS_BASE_URL})")
            return SharedQwenTTS(base_url=SHARED_TTS_BASE_URL)
        logger.warning(
            f"tts engine=qwen_shared but {SHARED_TTS_BASE_URL} isn't "
            "reachable - falling back to Qwen (remote, PC2)."
        )

    elif engine == "qwen_local_subprocess":
        # Old behavior, preserved as an explicit opt-in fallback: spawns a
        # fresh Qwen TTS subprocess on THIS machine per session. Reached
        # either by explicit system_config choice, or automatically from
        # entrypoint() when PC2 was found unreachable at this worker's
        # prewarm time.
        logger.info("TTS: Qwen (LOCAL subprocess, legacy fallback)")
        return QwenSubprocessTTS()

    # DEFAULT: Qwen3-TTS via vLLM-Omni, served remotely on PC2.
    #
    # VERIFIED (constructor signature AND real streaming behavior, both
    # confirmed live against PC2, not just inferred from openai.LLM()'s
    # usage below): since QWEN_OMNI_MODEL is a custom model path - not
    # "tts-1"/"tts-1-hd" - the openai plugin's TTS.synthesize() picks
    # SSEChunkedStream (stream_format="sse"), not the older AudioChunkedStream
    # path. Confirmed this genuinely streams progressively, not "buffer the
    # whole reply, then dump every frame at once behind a streaming-shaped
    # API": a short filler sentence's audio frames arrive within ~80ms of
    # each other (not meaningfully streamed - too little audio to show it),
    # but a realistic multi-sentence reply's frames spread across ~3.4s
    # between first and last frame, tracking real synthesis progress. First
    # audio reliably arrives ~500-700ms after the request starts regardless
    # of reply length - see README §3/§4.
    logger.info(f"TTS: Qwen3-TTS via vLLM-Omni (remote, {QWEN_OMNI_BASE_URL})")
    return openai.TTS(
        model=QWEN_OMNI_MODEL,
        voice=tts_cfg.get("voice", QWEN_OMNI_VOICE),
        base_url=QWEN_OMNI_BASE_URL,
        api_key="not-needed",  # vLLM-Omni ignores it, same as the LLM connection
        response_format="wav",  # vLLM-Omni's /v1/audio/speech rejects streaming
        # unless response_format is 'pcm' or 'wav'.
    )


def _group_slots(flat_slots: list) -> list:
    """The real booking.py returns a FLAT list of {doctor,date,time} dicts;
    the frontend renders per-doctor cards with time chips. Group here so the
    UI payload shape is stable regardless of which layer answered."""
    grouped: dict[tuple, dict] = {}
    for s in flat_slots or []:
        if not isinstance(s, dict):
            continue
        key = (s.get("doctor"), s.get("date"))
        grouped.setdefault(
            key, {"doctor": s.get("doctor"), "date": s.get("date"), "times": []}
        )
        if s.get("time"):
            grouped[key]["times"].append(s["time"])
    return list(grouped.values())


# Turn-level immediate filler: measured live (README §4), a real
# tool-invoking turn (booking, availability, hospital-info lookups -
# nearly every real caller request) takes ~3.2-3.5s stop-talking to
# first-audio - two LLM round trips (decide to call the tool, then
# answer from its result) plus real TTS synthesis, none of which
# `helpers.run_with_filler()`'s existing 0.7s race can see, since that
# race only covers the TOOL's own latency (fast local SQLite/RAG lookups,
# rarely over a few ms) - not the LLM/TTS time around it. This covers the
# actual gap instead: if the agent hasn't started producing real reply
# audio within TURN_FILLER_DELAY_SECS of the caller's turn being
# confirmed, speak a short, immediate acknowledgment so the caller hears
# SOMETHING well inside normal human conversational turn-taking latency,
# instead of 3+ seconds of dead air. This does not reduce the real
# compute time - it makes the wait perceptible as responsive instead of
# broken, the same technique real call-center agents and voice assistants
# use for exactly this kind of unavoidable backend latency.
#
# WHY 1.0s, NOT THE ORIGINALLY-TRIED 0.45s - a real, live call surfaced an
# interaction the isolated latency numbers didn't: with preemptive
# generation + the fast local qwen_shared TTS, MANY real turns (not just
# simple ones) are actually ready to speak in well under a second - ttft
# is often ~0.07s and short completions (~20-45 tokens at ~55-60 tok/s)
# generate in well under a second. At 0.45s the filler was firing on
# turns that were about to answer anyway, AND - worse - since
# tts_service/server.py is deliberately single-lane (concurrent
# generation corrupts output, see that file's docstring), the filler's
# own synthesis request competed with the real reply's for the SAME
# lock, so speaking a filler could make the real reply's audio arrive
# LATER than if no filler had fired at all, not "for free." 1.0s gives
# genuinely fast turns room to finish before the filler fires at all -
# it now only fires for the tool-invoking turns actually measured at
# 2-3.5s total. If this project ever needs the filler to fire faster than
# real replies can reliably beat it, the real fix is decoupling the
# filler from the shared TTS engine entirely (e.g. pre-rendered static
# audio played directly), not just tuning this number down again.
TURN_FILLER_DELAY_SECS = 1.0
TURN_FILLER_PHRASES = (
    "Mm-hmm, one moment.",
    "Let me check that for you.",
    "Just a second.",
)


async def _prerender_filler_audio(tts_service) -> dict[str, rtc.AudioFrame]:
    """Synthesize each TURN_FILLER_PHRASES phrase ONCE per call, with the
    actual TTS engine/voice this call is using, so a fired filler can play
    via session.say(audio=...) - a real frame already in memory - instead
    of a live synthesis call. This is exactly the fix TURN_FILLER_DELAY_SECS's
    own comment history already named as "the real fix" if ever needed:
    with tts_service/server.py's single-lane shared TTS, a LIVE-synthesized
    filler competes for the same lock the real reply's synthesis needs,
    which could make the real reply's audio arrive LATER than if no filler
    had fired at all - not "for free" the way a filler is supposed to be.
    A pre-rendered frame has zero runtime synthesis cost, so it can never
    do that. Confirmed as a current (2026) industry-standard technique, not
    a one-off idea - LiveKit's own docs recommend exactly this pattern
    (session.say(text, audio=...) with pre-synthesized audio) for "fixed
    phrases like greetings, hold messages, and error prompts... works best
    when the set of phrases is known and stable" - exactly this call site.

    Runs at call-start (entrypoint(), alongside the other warm-ups, not
    prewarm()) because the TTS engine/voice can differ per call (the PC2-
    reachability fallback chain above) - prewarm() runs once per WORKER
    process and can't know that yet. Failure is non-fatal: returns
    whatever succeeded, and _speak_turn_filler_after_delay() falls back to
    live synthesis for any phrase missing from the cache."""
    cache: dict[str, rtc.AudioFrame] = {}
    for phrase in TURN_FILLER_PHRASES:
        try:
            cache[phrase] = await tts_service.synthesize(phrase).collect()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"turn filler: pre-render failed for {phrase!r} ({exc}) - "
                "this phrase will fall back to live synthesis if it fires."
            )
    return cache


async def _single_frame_aiter(frame: rtc.AudioFrame):
    """Wraps one pre-rendered rtc.AudioFrame as the AsyncIterable
    session.say(audio=...) expects."""
    yield frame

# Chat context truncation - a REAL bug, not a hypothetical: found live via a
# real call that ran long enough to hit vLLM's configured
# --max-model-len 4096. Nothing was bounding chat history growth, so once a
# conversation's accumulated tokens crossed 4096, vLLM started rejecting the
# request with HTTP 400 ("This model's maximum context length is 4096
# tokens...") - and since history never shrinks, EVERY subsequent turn for
# that call failed identically, forever (confirmed: the same 400 repeated on
# turn after turn until the caller gave up and disconnected). This isn't a
# per-device/browser issue - purely a function of how many turns a given
# call has accumulated, which is exactly why it looked device-specific
# (one call happened to run longer than another).
#
# ChatContext.truncate(max_items=N) is the framework's own sanctioned tool
# for this (see llm/chat_context.py - keeps the last N items, preserves the
# system/instructions message, never leaves a dangling function_call without
# its output) - not a hand-rolled truncation. 16 items is a deliberately
# conservative budget: this project's system prompt + tool schemas alone
# already cost ~1841 of the 4096-token ceiling before a single word of
# conversation, and observed per-turn growth in the real call that triggered
# this was ~150-300 tokens/turn - 16 items (roughly 5-6 exchanges) leaves
# comfortable headroom under 4096 even for verbose turns. Applied in
# on_user_turn_completed - the framework's own documented "good opportunity
# to update the chat context... before it is sent to the LLM" hook.
CHAT_CTX_MAX_ITEMS = 16


class _ThinkingChannelStreamFilter:
    """Real bug, found live 2026-07-29 (this deployment, not a
    hypothetical): gemma-4-12b sometimes opens a `<|channel>thought...`
    reasoning block on its OWN initiative mid-completion, even though
    the chat template already defaults to enable_thinking=false and
    pre-inserts a CLOSED thought marker in the prompt specifically to
    signal "skip thinking" - checked directly in
    chat_templates/tool_chat_template_gemma4.jinja. Root cause is a
    missing --reasoning-parser in scripts/run_vllm.sh /
    docker/docker-compose.yml (confirmed: this vLLM install has ZERO
    reasoning parsers registered,
    vllm.reasoning.ReasoningParserManager.reasoning_parsers == {}) - a
    shared-config fix out of scope here, same boundary as the
    librosa/av install. Left unhandled, this leaked text gets spoken
    aloud verbatim by TTS, and - the more serious real incident this
    defends against - can consume the ENTIRE max_completion_tokens
    budget on rumination with NO real answer ever generated, leaving
    the caller in dead silence for the whole turn.

    Streaming-safe (content arrives one token/fragment at a time, not
    as one final string) - same buffering approach as
    direct_audio_agent/llm_plugin.py's _ThinkingChannelStreamFilter,
    reimplemented here rather than imported since app/ and
    direct_audio_agent/ are deliberately independent (see that file's
    README for why)."""

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
                return ""
            self._state = "clean"
            rest = self._buffer[idx + len(self._CLOSE) :]
            self._buffer = ""
            return rest

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
            return ""

        self._state = "clean"
        out = self._buffer
        self._buffer = ""
        return out


# ---------------------------------------------------------------------------
# The agent: persona + safety gate + tools
# ---------------------------------------------------------------------------
class RiversideReceptionist(Agent):
    def __init__(self) -> None:
        self._base_instructions = build_system_prompt()
        super().__init__(instructions=self._base_instructions)
        self._filler_task: asyncio.Task | None = None
        self._filler_index = 0
        # REAL BUG, found live 2026-07-30: a caller heard a filler phrase
        # ("Mm-hmm, one moment.") fire AFTER the real reply had already
        # started speaking - not a timing-tuning issue, a genuine race.
        # cancel_pending_filler() calls Task.cancel() on the armed
        # filler task, but _speak_turn_filler_after_delay()'s only await
        # point is the delay sleep itself - once TURN_FILLER_DELAY_SECS
        # has genuinely elapsed, the task is no longer suspended at that
        # await, it's already scheduled to run its post-sleep body
        # (phrase pick + session.say(), both synchronous, no further
        # await for a cancellation to land at). If the real reply starts
        # "speaking" at close to the same instant the filler's own timer
        # elapses - exactly the common case this delay was tuned for,
        # see TURN_FILLER_DELAY_SECS's docstring - Task.cancel() can lose
        # that race and the filler fires anyway, right on top of/just
        # after real speech. This flag is an explicit, unambiguous
        # backstop checked AFTER the sleep, independent of asyncio's
        # cancellation-delivery timing: set the instant real speech
        # starts, reset the instant a new filler is armed for the next
        # turn.
        self._real_reply_started = False
        # Populated by entrypoint() right after tts_service is known, via
        # _prerender_filler_audio() - see that function's docstring for
        # why. None until then; _speak_turn_filler_after_delay() falls
        # back to live synthesis if it's still None/empty when a filler
        # actually needs to fire (e.g. pre-rendering itself failed).
        self.filler_audio: dict[str, rtc.AudioFrame] = {}
        # Survives CHAT_CTX_MAX_ITEMS truncation - see _remember()'s
        # docstring for why this exists instead of trusting raw history.
        self._call_facts: dict[str, str] = {}

    # ------------------------------------------------------------- lifecycle
    async def on_enter(self) -> None:
        # Light up the frontend directory board the moment the agent joins.
        await push_ui({"type": "roster", "doctors": compat.get_doctor_roster()})
        await push_ui({"type": "status", "state": "listening"})
        self.session.generate_reply(instructions=GREETING_INSTRUCTIONS)

    # ------------------------------------------------------------------ LLM
    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ):
        """Wraps the default llm_node to strip leaked
        `<|channel>thought...` content from streamed text before it can
        reach TTS - see _ThinkingChannelStreamFilter's docstring for the
        real incident this defends against (a caller left in dead
        silence for an entire turn because the leak consumed the whole
        token budget). Tool-call chunks and usage chunks pass through
        completely unmodified - only `delta.content` is filtered.

        DOES NOT also re-truncate chat_ctx here, despite a same-day
        attempt to do exactly that. REAL BUG, found live 2026-07-30,
        caught via the SAME test call that motivated it: truncating on
        EVERY completion (not just once per user turn, in
        on_user_turn_completed) did bound the mid-turn prompt-size growth
        it was built for - but it also made LiveKit's own chat-context
        code log "function output missing the corresponding function
        call, ignoring" repeatedly (see
        livekit.agents.llm._provider_format.utils.group_tool_calls /
        _ChatItemGroup.remove_invalid_tool_calls) - ChatContext.truncate()
        only protects against an orphaned function_call/function_call_output
        landing at the very FRONT of the truncated window; calling it
        this much more often, mid-tool-chain, hit that boundary far more
        often and started silently DROPPING tool results from what the
        LLM can see. For a booking agent, an LLM that can no longer see
        a tool's own result is a worse failure mode than a turn running
        a bit slower, so this reverts to the single per-turn truncation
        point only. The mid-turn prompt growth this was chasing is real
        (see CHAT_CTX_MAX_ITEMS's docstring) but accepted as a known,
        lesser cost unless/until a truncation approach that's provably
        call/output-pair-safe replaces it."""
        stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        if asyncio.iscoroutine(stream):
            stream = await stream
        if stream is None:
            return
        leak_filter = _ThinkingChannelStreamFilter()
        # yielded_anything tracks whether ANY visible text or tool_calls
        # made it downstream this turn. REAL INCIDENT, not hypothetical
        # (this deployment, 2026-07-30): a turn where the model's ENTIRE
        # completion was spent opening `<|channel>thought` and never
        # closing it (confirmed live: completion_tokens=3, i.e. the model
        # barely started before stopping) leaves leak_filter stuck in
        # "stripping" forever - every push() call returns "", so this
        # loop yields NOTHING at all. Before this fix that meant the turn
        # ended in total silence: no TTS call, no error, no log line -
        # indistinguishable from a hang to the caller, who has no signal
        # to do anything but wait and eventually hang up. That's a WORSE
        # outcome than the leaked-text-spoken-aloud case this filter was
        # built to prevent. The fallback below guarantees the caller
        # always hears SOMETHING instead of dead air.
        yielded_anything = False
        async for chunk in stream:
            if isinstance(chunk, str):
                visible = leak_filter.push(chunk)
                if visible:
                    yielded_anything = True
                    yield visible
                continue
            if chunk.delta is not None and chunk.delta.content:
                visible = leak_filter.push(chunk.delta.content)
                if not visible and not chunk.delta.tool_calls:
                    continue
                chunk.delta.content = visible
            if chunk.delta is not None and (chunk.delta.content or chunk.delta.tool_calls):
                yielded_anything = True
            yield chunk
        if not yielded_anything:
            logger.error(
                "llm_node: entire completion swallowed by the thinking-channel "
                "leak filter (model never closed `<|channel>thought...`) - "
                "the turn would otherwise end in total silence. Speaking a "
                "fallback so the caller always hears something."
            )
            yield "Sorry, could you say that again?"

    # ------------------------------------------------------------ TURN FILLER
    def cancel_pending_filler(self) -> None:
        """Called from entrypoint()'s agent_state_changed handler the
        moment real audio actually starts (whether that's this filler's
        own playback or the real reply beating it) - cancelling an
        already-fired/completed task is a harmless no-op, so this is safe
        to call unconditionally on every "speaking" transition. Also
        flips _real_reply_started - see that field's docstring for the
        real race this closes that Task.cancel() alone did not."""
        self._real_reply_started = True
        if self._filler_task is not None and not self._filler_task.done():
            self._filler_task.cancel()
        self._filler_task = None

    async def _speak_turn_filler_after_delay(self) -> None:
        self._real_reply_started = False
        logger.debug(f"turn filler: armed, sleeping {TURN_FILLER_DELAY_SECS}s")
        try:
            await asyncio.sleep(TURN_FILLER_DELAY_SECS)
        except asyncio.CancelledError:
            logger.debug("turn filler: cancelled before firing (real reply beat it)")
            return
        if self._real_reply_started:
            # Task.cancel() lost the race (see _real_reply_started's
            # docstring) - the sleep completed without raising
            # CancelledError, but real speech already started in the
            # meantime. Bail out explicitly instead of speaking over/
            # right after it.
            logger.debug(
                "turn filler: sleep completed but real reply already started "
                "(cancel() lost the race) - not firing"
            )
            return
        phrase = TURN_FILLER_PHRASES[self._filler_index % len(TURN_FILLER_PHRASES)]
        self._filler_index += 1
        frame = self.filler_audio.get(phrase)
        logger.debug(
            f"turn filler: firing now - {phrase!r} "
            f"({'pre-rendered' if frame is not None else 'LIVE SYNTHESIS - pre-render missing/failed'})"
        )
        try:
            if frame is not None:
                self.session.say(phrase, audio=_single_frame_aiter(frame), add_to_chat_ctx=False)
            else:
                self.session.say(phrase, add_to_chat_ctx=False)
        except TypeError:
            self.session.say(phrase)

    # -------------------------------------------------------- CALL-FACT RECAP
    _RECAP_LABELS = (
        ("patient_name", "patient name"),
        ("department", "department"),
        ("doctor", "doctor"),
        ("date", "date"),
        ("time", "time"),
    )

    async def _remember(self, **facts: str | None) -> None:
        """Call with whatever the caller just stated (patient_name,
        department, doctor, date, time - pass only the ones a tool call
        just received, None/omitted values are ignored). Folds them into
        the agent's INSTRUCTIONS, not the raw chat history - the point is
        that CHAT_CTX_MAX_ITEMS truncation only guarantees the *first*
        instructions message survives, not any particular chat turn. A
        caller who gives their name in turn 2 and is still going in turn
        20 would otherwise have that fact silently truncated away,
        producing exactly the "forgot who I'm talking to" failure a human
        receptionist would never make. update_instructions() is the
        framework's own sanctioned way to change this mid-session (see
        livekit.agents.voice.agent.Agent.update_instructions) - this
        rebuilds from the ORIGINAL base prompt each time rather than
        appending, so the recap reflects current knowledge, not a growing
        log of every value ever mentioned (a corrected department/date
        replaces the old one instead of both lingering)."""
        changed = False
        for key, _label in self._RECAP_LABELS:
            value = facts.get(key)
            if value and self._call_facts.get(key) != value:
                self._call_facts[key] = value
                changed = True
        if not changed:
            return

        recap_parts = [
            f"{label}: {self._call_facts[key]}"
            for key, label in self._RECAP_LABELS
            if key in self._call_facts
        ]
        recap = "Known so far this call (do not ask again for these): " + "; ".join(
            recap_parts
        )
        await self.update_instructions(f"{self._base_instructions}\n\n{recap}")

    # ------------------------------------------------------------ SAFETY GATE
    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Deterministic emergency bypass - the LiveKit equivalent of
        SafetyGateProcessor. Runs on the final transcript BEFORE any LLM
        inference for this turn. On a match: speak the escalation message
        directly (not via the LLM, not added to context) and StopResponse()
        so no generation happens.

        Acceptance test (never completed live in the old build): say
        "I'm having chest pain" -> escalation audio plays AND the logs show
        NO llm generation/TTFB entry for that turn."""
        # Bound context growth BEFORE anything else this turn - see
        # CHAT_CTX_MAX_ITEMS's docstring for the real outage this fixes.
        # Unconditional (applies to emergency turns too): a real call
        # doesn't need unlimited history, and doing this first means a
        # long-running call is protected regardless of what this hook
        # decides below.
        turn_ctx.truncate(max_items=CHAT_CTX_MAX_ITEMS)
        text = (getattr(new_message, "text_content", None) or "").strip()
        if not text:
            return
        hit = compat.run_safety_gate(text)
        if hit is None:
            # Not an emergency - a normal turn is starting, so arm the
            # turn-level filler (see cancel_pending_filler()'s docstring
            # for how/when it gets cancelled). Emergencies skip this
            # entirely: the escalation message below IS the immediate
            # response, no filler needed or wanted.
            self.cancel_pending_filler()  # replace any stale prior-turn timer
            self._filler_task = asyncio.create_task(self._speak_turn_filler_after_delay())
            logger.debug(f"turn filler: task created ({self._filler_task!r})")
            return

        logger.warning(
            "SAFETY GATE: emergency matched "
            f"(category={hit.get('category')}, kind={hit.get('kind')}, "
            f"matched_text={hit.get('matched_text')!r}) "
            "- bypassing LLM, speaking escalation directly"
        )
        await push_ui(
            {
                "type": "emergency",
                "category": hit.get("category"),
                "message": hit.get("message"),
            }
        )
        try:
            self.session.say(hit["message"], add_to_chat_ctx=False)
        except TypeError:
            self.session.say(hit["message"])
        raise StopResponse()  # the LLM never sees this turn

    # ------------------------------------------------------------------ tools
    # Signatures mirror hospital_core/booking.py EXACTLY. Docstrings ARE the
    # schemas the LLM sees - they encode the contract rules learned the
    # hard way.

    @function_tool
    async def check_availability(
        self,
        context: RunContext,
        department: str,
        date: str | None = None,
        doctor: str | None = None,
    ) -> dict:
        """Check open appointment slots. This is your MANDATORY first tool
        call whenever the caller names any department or doctor, even one
        you believe does not exist - this tool decides, not you.

        Args:
            department: Department name as the caller said it (e.g. cardiology).
            date: Optional day in YYYY-MM-DD. Omit to see all upcoming slots.
            doctor: Optional doctor name if the caller asked for one.
        """
        # Tool-call args weren't logged anywhere before 2026-07-30 - when a
        # caller reported "my name is Abdullah Amin" got answered with
        # "no doctor named Abdullah Amin" (the LLM misreading a caller's
        # self-introduction as a doctor-name lookup - see prompts.py's
        # fix for the actual behavior change), there was no log line
        # anywhere confirming what arguments the LLM actually passed -
        # only the DB and the caller's own report to go on. This makes
        # the next one directly diagnosable instead of inferred.
        logger.info(
            f"tool call: check_availability(department={department!r}, "
            f"date={date!r}, doctor={doctor!r})"
        )
        await self._remember(department=department, date=date, doctor=doctor)
        result = await run_with_filler(
            context.session,
            compat.check_availability(department=department, date=date, doctor=doctor),
            filler=DEFAULT_FILLERS["check_availability"],
        )
        logger.info(f"tool result: check_availability -> status={result.get('status')!r}")
        if isinstance(result, dict):
            slots = _group_slots(result.get("slots") or [])
            alternatives = _group_slots(result.get("alternatives") or [])
            if slots or alternatives:
                await push_ui(
                    {
                        "type": "availability",
                        "department": result.get("department") or department,
                        "date": result.get("date"),
                        "slots": slots,
                        "alternatives": alternatives,
                    }
                )
        return result

    @function_tool
    async def book_appointment(
        self,
        context: RunContext,
        patient_name: str,
        department: str,
        date: str,
        time: str,
        doctor: str | None = None,
    ) -> dict:
        """Book an appointment. Only call after check_availability confirmed
        the slot and the caller confirmed their details.

        Args:
            patient_name: The caller's full name, confirmed back to them.
            department: The department name.
            date: The confirmed day, in YYYY-MM-DD, EXACTLY as
                check_availability returned it.
            time: EXACTLY the time string check_availability returned,
                character for character (e.g. "11:00" - never "11:00 AM").
            doctor: Optional specific doctor; omit to take any open doctor.
        """
        logger.info(
            f"tool call: book_appointment(patient_name={patient_name!r}, "
            f"department={department!r}, date={date!r}, time={time!r}, "
            f"doctor={doctor!r})"
        )
        await self._remember(
            patient_name=patient_name, department=department, date=date, time=time, doctor=doctor
        )
        result = await run_with_filler(
            context.session,
            compat.book_appointment(
                patient_name=patient_name,
                department=department,
                date=date,
                time=time,
                doctor=doctor,
            ),
            filler=DEFAULT_FILLERS["book_appointment"],
        )
        logger.info(f"tool result: book_appointment -> status={result.get('status')!r}")
        if isinstance(result, dict):
            if result.get("status") == "booked":
                await push_ui({"type": "booking", "booking": result})
            elif result.get("alternatives"):
                await push_ui(
                    {
                        "type": "availability",
                        "department": department,
                        "date": date,
                        "slots": _group_slots(result["alternatives"]),
                    }
                )
        return result

    @function_tool
    async def cancel_appointment(
        self,
        context: RunContext,
        patient_name: str,
        date: str | None = None,
        department: str | None = None,
        time: str | None = None,
    ) -> dict:
        """Cancel an existing appointment, looked up by patient name. If the
        result is "ambiguous", ask the caller which department or date they
        mean and call again with those extra arguments.

        Args:
            patient_name: The caller's full name as used at booking.
            date: Optional day (YYYY-MM-DD) of the appointment to narrow down.
            department: Optional department to narrow down.
            time: Optional time (e.g. "11:00") to narrow down.
        """
        logger.info(
            f"tool call: cancel_appointment(patient_name={patient_name!r}, "
            f"date={date!r}, department={department!r}, time={time!r})"
        )
        await self._remember(patient_name=patient_name, department=department, date=date, time=time)
        result = await run_with_filler(
            context.session,
            compat.cancel_appointment(
                patient_name=patient_name, date=date, department=department, time=time
            ),
            filler=DEFAULT_FILLERS["cancel_appointment"],
        )
        logger.info(f"tool result: cancel_appointment -> status={result.get('status')!r}")
        if isinstance(result, dict) and result.get("status") == "cancelled":
            await push_ui({"type": "cancellation", "booking": result})
        return result

    @function_tool
    async def update_appointment(
        self,
        context: RunContext,
        patient_name: str,
        new_date: str | None = None,
        new_time: str | None = None,
        date: str | None = None,
        department: str | None = None,
        time: str | None = None,
    ) -> dict:
        """Reschedule an existing appointment - same doctor and department.
        Check the new slot with check_availability first, and pass new_time
        EXACTLY as that tool returned it. If the result is "ambiguous", ask
        which department or date they mean and call again.

        Args:
            patient_name: The caller's full name as used at booking.
            new_date: The new day (YYYY-MM-DD), if changing the day.
            new_time: The new time, exactly as check_availability returned it.
            date: The CURRENT appointment's day, to narrow down which one.
            department: The CURRENT appointment's department, to narrow down.
            time: The CURRENT appointment's time, to narrow down.
        """
        logger.info(
            f"tool call: update_appointment(patient_name={patient_name!r}, "
            f"new_date={new_date!r}, new_time={new_time!r}, date={date!r}, "
            f"department={department!r}, time={time!r})"
        )
        await self._remember(
            patient_name=patient_name,
            department=department,
            date=new_date or date,
            time=new_time or time,
        )
        result = await run_with_filler(
            context.session,
            compat.update_appointment(
                patient_name=patient_name,
                new_date=new_date,
                new_time=new_time,
                date=date,
                department=department,
                time=time,
            ),
            filler=DEFAULT_FILLERS["update_appointment"],
        )
        logger.info(f"tool result: update_appointment -> status={result.get('status')!r}")
        if isinstance(result, dict):
            if result.get("status") in ("booked", "updated"):
                await push_ui({"type": "booking", "booking": result})
            elif result.get("alternatives"):
                await push_ui(
                    {
                        "type": "availability",
                        "department": department or result.get("department"),
                        "date": new_date,
                        "slots": _group_slots(result["alternatives"]),
                    }
                )
        return result

    @function_tool
    async def search_hospital_info(self, context: RunContext, query: str) -> dict:
        """Look up general hospital information: hours, departments, doctor
        bios, insurance, billing, prescriptions, visiting policy, lab
        results, parking. Use this BEFORE ever saying you don't have some
        piece of hospital information. Do NOT use it for appointment
        availability or booking.

        Args:
            query: The caller's question, rephrased as a short search query.
        """
        return await run_with_filler(
            context.session,
            compat.search_hospital_info(query=query),
            filler=DEFAULT_FILLERS["search_hospital_info"],
        )


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------
def prewarm(proc: JobProcess) -> None:
    """Runs once per worker process, BEFORE any call is accepted. All the
    warm-up lessons from the Pipecat build live here or in entrypoint():
    a cold anything (VAD, STT, RAG embedder, vLLM, TTS CUDA graphs) must
    never be paid for by a real caller's first turn.

    CROSS-PROCESS NOTE on the dev-UI model switcher: this reads
    system_config fresh at the moment THIS worker process starts up. A
    developer's STT-backend switch takes effect for any NEWLY SPAWNED
    worker process - an already-warmed, already-pooled process keeps
    whatever it was warmed with until LiveKit recycles it. Same honest
    cross-process caveat as the hospital-data admin UI, documented there
    for the same underlying reason (this file, admin_server.py, and the
    dev UI are all separate processes with their own in-memory state).

    PC2 REACHABILITY: also checked once here, ONCE per worker process
    (not per call - see _qwen_omni_reachable's docstring for why), and
    the result is stashed in proc.userdata for entrypoint() to read on
    every call this worker handles."""
    cfg = system_config.get_config()
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=float(os.environ.get("VAD_MIN_SILENCE", "0.4")),
    )
    stt_service = _make_stt(cfg["stt"])
    stt_service.load()
    proc.userdata["stt"] = stt_service

    proc.userdata["qwen_omni_reachable"] = _qwen_omni_reachable(QWEN_OMNI_BASE_URL)
    # Checked here too (not just when tts.engine is explicitly
    # "qwen_shared") so entrypoint()'s PC2-down fallback chain below can
    # prefer the fast shared local service over the legacy per-worker
    # subprocess, if it happens to be up.
    proc.userdata["qwen_shared_reachable"] = _shared_qwen_tts_reachable(SHARED_TTS_BASE_URL)

    compat.warm_rag()


async def _warm_up_vllm(served_model_name: str) -> None:
    """One tiny completion so the first real turn doesn't eat vLLM's
    cold-start spike (~2s observed live in the old build)."""
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json={
                    "model": served_model_name,
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": "ping"}],
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    # Previously this just called resp.read() and logged
                    # "OK" unconditionally, which silently masked a wrong
                    # served_model_name (vLLM returns 404 "model does not
                    # exist" here, not a connection error) as a false
                    # success. Surface it loudly instead.
                    logger.error(
                        f"vLLM warm-up ping got HTTP {resp.status} - "
                        f"served_model_name='{served_model_name}' likely doesn't "
                        f"match what vLLM is actually serving. Check with "
                        f"`curl {VLLM_BASE_URL}/models` and fix system_config.json "
                        f"(or switch models again via the dev UI). Body: {body[:300]}"
                    )
                    return
        logger.info("vLLM warm-up ping OK.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"vLLM warm-up ping failed (continuing): {exc}")


async def _warm_up_qwen_omni() -> None:
    """One tiny synthesis request so the FIRST real caller doesn't eat
    PC2's cold-start cost. Mirrors _warm_up_vllm's shape and its "log
    loudly on non-200, don't just assume success" fix - a wrong
    QWEN_OMNI_BASE_URL or QWEN_OMNI_MODEL should be obvious in the logs,
    not silently swallowed. Only called when the reachability check in
    prewarm() already found PC2 up - see entrypoint()."""
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{QWEN_OMNI_BASE_URL}/audio/speech",
                json={
                    "model": QWEN_OMNI_MODEL,
                    "input": "warm up",
                    "voice": QWEN_OMNI_VOICE,
                    "response_format": "wav",
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        f"Qwen-Omni warm-up got HTTP {resp.status} from "
                        f"{QWEN_OMNI_BASE_URL} - check PC2's server is running, "
                        f"reachable, and QWEN_OMNI_MODEL matches what it's "
                        f"actually serving. Body: {body[:300]}"
                    )
                    return
                await resp.read()
        logger.info(f"Qwen-Omni (PC2, {QWEN_OMNI_BASE_URL}) warm-up OK.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"Qwen-Omni warm-up ping failed (continuing, but PC2 may be "
            f"unreachable): {exc}"
        )


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # Read fresh HERE, at the start of every call - this is what makes a
    # dev-UI model switch take effect on "the next call" rather than
    # needing a full agent restart. See system_config.py's docstring for
    # the full contract (and vllm_manager.py for why an LLM switch has a
    # real ~60s+ cost the TTS/STT switches don't).
    cfg = system_config.get_config()
    served_model_name = cfg["llm"]["served_model_name"]

    # PC2 REACHABILITY FALLBACK CHAIN: if this worker process found PC2
    # unreachable back in prewarm() (checked ONCE per process, not per
    # call - see _qwen_omni_reachable's docstring), and the configured
    # engine is the default remote Qwen-Omni path, fall back in order of
    # "closest to PC2's quality/latency": first the shared local
    # Qwen3-TTS service (qwen_shared, if it happens to be running),
    # THEN the legacy per-worker subprocess (qwen_local_subprocess, always
    # available as a last resort, but pays a ~15s cold-worker-spawn cost
    # THIS call would eat since it isn't prewarmed). Explicit non-Qwen
    # choices (chatterbox/kokoro/piper) are left alone - PC2 being down is
    # irrelevant to those engines.
    tts_cfg = dict(cfg["tts"])  # copy - don't mutate the shared config dict
    if tts_cfg.get("engine", "qwen_omni") in ("qwen_omni", "qwen") and not (
        ctx.proc.userdata.get("qwen_omni_reachable", True)
    ):
        if ctx.proc.userdata.get("qwen_shared_reachable", False):
            logger.warning(
                "TTS: PC2 unreachable at this worker's startup - using the "
                "shared local Qwen3-TTS service (qwen_shared) for this call."
            )
            tts_cfg["engine"] = "qwen_shared"
        else:
            logger.warning(
                "TTS: PC2 unreachable at this worker's startup (and the "
                "shared local Qwen3-TTS service isn't running either) - "
                "using local Qwen subprocess fallback for this call."
            )
            tts_cfg["engine"] = "qwen_local_subprocess"

    tts_service = _make_tts(tts_cfg)

    # Warm-up strategy differs by TTS path: the remote Qwen-Omni server on
    # PC2 is warmed with a tiny real HTTP request (_warm_up_qwen_omni),
    # same shape as the vLLM warm-up ping. The local-subprocess path
    # (whether explicitly configured or reached via the fallback above)
    # uses its own tts_service.prewarm() method instead, since that class
    # manages its own subprocess lifecycle.
    # filler_audio_cache: pre-rendered turn-filler phrases, alongside the
    # other warm-ups (not blocking on them) - see _prerender_filler_audio's
    # docstring for why this runs here (per-call, TTS engine known) rather
    # than in prewarm() (per-worker, TTS engine not yet known).
    if isinstance(tts_service, QwenSubprocessTTS):
        _, _, filler_audio_cache = await asyncio.gather(
            _warm_up_vllm(served_model_name),
            tts_service.prewarm(),
            _prerender_filler_audio(tts_service),
        )
    else:
        _, _, filler_audio_cache = await asyncio.gather(
            _warm_up_vllm(served_model_name),
            _warm_up_qwen_omni(),
            _prerender_filler_audio(tts_service),
        )

    # Semantic turn detection: the single highest-leverage latency change of
    # this whole migration. version="v1-mini" is pinned EXPLICITLY - if left
    # unset, inference.TurnDetector() auto-resolves to "v1", a CLOUD-hosted
    # model that needs LIVEKIT_INFERENCE_URL + LiveKit Cloud credentials and
    # a network round-trip per turn. This stack is fully self-hosted (own
    # livekit-server, own vLLM, own STT) with no LiveKit Cloud account, so
    # the cloud default would either fail outright or silently add a WAN
    # hop to the hottest path in the whole system. "v1-mini" runs the same
    # class of model locally (per-language calibrated thresholds), same as
    # the old EnglishModel it replaces, just via the non-deprecated API.
    # Falls back to plain VAD end-pointing if construction fails, so the
    # agent still runs.
    turn_detection = None
    try:
        # unlikely_threshold={"en": 0.15} - TUNED 2026-07-30, down from
        # 0.22 (2026-07-29), itself down from the shipped default (0.36
        # for English, confirmed by reading
        # livekit.agents.inference.eot.languages.LOCAL_LANGUAGES directly).
        # REAL DATA, not a guess: a real call on 2026-07-29 (room
        # reception-46069e5e) hit the FULL 2.5s max_delay ceiling on 3 of
        # 4 turns - end_of_utterance_delay was the single largest latency
        # contributor anywhere in the pipeline that call, dwarfing
        # STT/LLM/TTS combined (all of which stayed under ~0.85s). The
        # turn detector is a binary confident/uncertain classifier - when
        # uncertain, it ALWAYS pays the full max_delay, not a graded wait
        # - so a lower threshold means fewer turns get classified
        # "uncertain" in the first place. Paired with max_delay 1.5 -> 0.8
        # below to also bound the worst case when one still is.
        # TRADE-OFF, not a free win, and pushed further deliberately on
        # 2026-07-30: makes the agent more willing to jump in - a
        # genuinely slow/thinking-pause caller is more likely to get
        # interrupted than under 0.22/1.5, and more likely still than the
        # original 0.36/2.5. Chosen because raw responsiveness mattered
        # more than interruption risk for this deployment - re-tune back
        # up if real calls start showing the agent cutting callers off
        # mid-thought.
        turn_detection = inference.TurnDetector(
            version="v1-mini", unlikely_threshold={"en": 0.15}
        )
        logger.info(
            "Semantic turn detector: ENABLED (local, v1-mini, tuned: "
            "unlikely_threshold[en]=0.15, see comment above)."
        )
    except Exception as exc:  # noqa: BLE001
        # Most common cause: `python agent.py download-files` was never run
        # (or ran before this plugin was registered), so the model files
        # aren't on disk yet. This must NOT crash the whole job - fall back
        # to VAD-only endpointing.
        logger.warning(
            f"Turn detector model unavailable ({exc}) - falling back to "
            "VAD-only endpointing. Run `python agent.py download-files` "
            "once to fetch the model, then restart to re-enable it."
        )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=ctx.proc.userdata["stt"],
        llm=openai.LLM(
            model=served_model_name,
            base_url=VLLM_BASE_URL,
            api_key="not-needed",  # vLLM ignores it; the plugin requires one
            # REAL BUG FOUND 2026-07-29, not hypothetical: MAX_TOKENS (above)
            # was defined and read from the env but never actually passed
            # here - every completion was UNBOUNDED. Confirmed as the real
            # cause of a live "it just stops, caller hangs up" incident:
            # vLLM logged 200 OK for the request (it was accepted and
            # streaming started), but no LLM metrics or TTS ever followed -
            # consistent with generation running far longer than a normal
            # reply (thousands of tokens rather than tens) before the
            # caller gave up, not a crash or hang in the usual sense. This
            # is a real risk independent of the short-opener prompt change
            # made the same day - the leaked thinking-channel repetition
            # bug (accountability history, gemma-4-12b's tool-offered-
            # but-not-called quirk) is exactly the kind of degenerate
            # output that would run away without a cap.
            max_completion_tokens=MAX_TOKENS,
            # THE REAL FIX for the reasoning-channel bug, found 2026-07-29
            # by testing (not guessing): gemma-4-12b's `<|channel>` token
            # (confirmed via the model's own tokenizer: a SINGLE token,
            # id 100) opens a "thought" block the model sometimes enters
            # on its own initiative and then, in a real fraction of
            # cases, never exits - burning the ENTIRE max_completion_tokens
            # budget on rumination with zero real content produced (the
            # exact cause of the live "caller hears nothing, hangs up"
            # incident). Verified two things directly against this
            # server before deploying: (1) temperature=0 (greedy) hits
            # this runaway 12/12 times, always the same ~518 characters
            # of reasoning - it's the model's DEFAULT path, not rare bad
            # luck; (2) banning token 100 via logit_bias eliminates it
            # completely (0/16 true failures vs baseline's 4/16, or 25%)
            # and makes every reply ~4x faster on average (0.33s vs
            # 1.35s), with normal replies and tool-calling both verified
            # still working correctly. _ThinkingChannelStreamFilter
            # (llm_node override, above) stays in place as defense in
            # depth for any leak this doesn't fully suppress - it's now
            # a backstop, not the primary fix.
            extra_body={"logit_bias": {"100": -100}},
        ),
        tts=tts_service,
        # Non-deprecated shape (replaces the old turn_detection=/
        # min_endpointing_delay=/max_endpointing_delay= kwargs, which are
        # shimmed through _migrate_turn_handling() with a deprecation
        # warning in livekit-agents 1.6.6+).
        turn_handling={
            "turn_detection": turn_detection,  # None -> VAD-only endpointing
            # "dynamic" paces the endpointing wait to the caller's own
            # recent speaking cadence (adaptive EMA) instead of a fixed
            # delay - faster average turnaround for a normal-cadence
            # caller.
            #
            # max_delay=0.8 (down from 1.5 on 2026-07-29, itself down
            # from 2.5, itself down from an originally untested 6.0 - see
            # the history below). 2.5 was "LiveKit's own documented
            # default for this mode," but a real call on 2026-07-29
            # showed that default still isn't good enough for THIS
            # deployment: 3 of 4 turns hit the full 2.5s ceiling - see
            # the turn_detection comment above for the real numbers. 0.8s
            # bounds the worst case much further still, deliberately
            # prioritizing responsiveness over caller think-time on
            # 2026-07-30 - re-tune back up toward 1.5 if real calls start
            # showing the agent cutting callers off mid-thought. min_delay
            # stays 0.1: Whisper's own STT latency (~0.1-0.22s measured,
            # see direct_audio_agent/benchmark.py) is comfortably under
            # it, so - unlike direct_audio_agent's GemmaDirectAudioSTT,
            # which genuinely needed min_delay raised - nothing here
            # implicates min_delay as a problem for THIS STT engine.
            #
            # History: 6.0 was a deliberate but UNTESTED choice ("give a
            # caller reciting a date room to pause"). A real call
            # surfaced the real cost: HALF of that call's turns hit the
            # full 6.0s ceiling before the agent even started processing
            # - a 6-second silent gap that reads as broken, not patient,
            # dwarfing every other latency fix in this file at the time.
            "endpointing": {"mode": "dynamic", "min_delay": 0.1, "max_delay": 0.8},
            # Explicit, not just inherited: start LLM inference on stable
            # partial STT text before end-of-turn is even confirmed.
            # preemptive_tts stays False - speculatively running TTS too
            # would shave more latency but burns real GPU/PC2 compute on
            # every turn the speculation gets discarded, which isn't worth
            # it until preemptive_generation alone is measured in prod.
            "preemptive_generation": {"enabled": True, "preemptive_tts": False},
        },
    )

    # --- observability: per-stage latency (STT / LLM TTFT / TTS TTFB / EOU)
    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev) -> None:
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

        # LATENCY COMPARISON LOGGING (dev console feature): LiveKit emits
        # EOU / LLM / TTS metrics as SEPARATE events per turn, each
        # carrying its own field set (confirmed from this project's own
        # real logs: EOU events carry end_of_utterance_delay +
        # transcription_delay; LLM events carry ttft; TTS events carry
        # ttfb). Rather than assume a way to correlate multiple events
        # into one combined "turn" record (which would need verifying
        # LiveKit's internal event-correlation ID, not available to check
        # in this build environment), this logs ONE RECORD PER EVENT,
        # populated with whichever tracked fields that specific event
        # actually has. latency_log.get_summary() already averages each
        # field independently across all records for a combo, so these
        # sparse per-event records aggregate correctly regardless.
        fields = {}
        for raw_name, log_name in (
            ("end_of_utterance_delay", "end_of_utterance_delay"),
            ("transcription_delay", "transcription_delay"),
            ("ttft", "llm_ttft"),
            ("ttfb", "tts_ttfb"),
        ):
            value = getattr(ev.metrics, raw_name, None)
            if value is not None:
                fields[log_name] = value
        if fields:
            latency_log.record(cfg, fields)

    async def _log_usage() -> None:
        logger.info(f"Session usage summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(_log_usage)

    agent = RiversideReceptionist()
    agent.filler_audio = filler_audio_cache
    logger.info(
        f"turn filler: pre-rendered {len(filler_audio_cache)}/{len(TURN_FILLER_PHRASES)} "
        "phrases for this call."
    )

    # Cancels the turn-level filler (see TURN_FILLER_DELAY_SECS above) the
    # moment real audio actually starts - "speaking" fires both for the
    # filler's own playback and for the real reply, but cancelling an
    # already-fired/completed task is a harmless no-op either way, so no
    # need to distinguish which one triggered it here.
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev) -> None:
        if ev.new_state == "speaking":
            agent.cancel_pending_filler()

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            # NOTE: Krisp noise cancellation (noise_cancellation.BVC()) is a
            # LiveKit CLOUD feature - it will not run against a self-hosted
            # livekit-server, so it is deliberately not enabled here.
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # DEFAULT IS 10s. prewarm() chains Silero VAD -> STT (whisper,
            # in-process Parakeet, or the Parakeet SUBPROCESS which itself
            # needs ~15-20s to load in its own venv) -> the RAG
            # sentence-transformers embedder -> the PC2 reachability check
            # (bounded to ~2s by _qwen_omni_reachable's timeout). On a cold
            # cache, or under GPU contention from vLLM/Qwen already
            # running, that chain can exceed even a generous ceiling - and
            # LiveKit kills the whole process the instant it's exceeded,
            # sometimes mid-report (a BrokenPipeError from the Parakeet
            # worker trying to write "ready" to an already-closed parent
            # pipe is the signature of this exact race). 300s gives real
            # headroom for the worst case: first-time downloads AND three
            # GPU processes competing for the same card. Once everything is
            # warm/cached, prewarm actually finishes in a fraction of this -
            # the timeout only costs anything on a failure, never on the
            # happy path.
            initialize_process_timeout=300.0,
        )
    )
