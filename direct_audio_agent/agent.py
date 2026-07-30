"""
agent.py - LiveKit voice-agent entrypoint for the direct-audio pipeline.

Reuses app/main.py's RiversideReceptionist class - persona, hospital
tools, the emergency safety gate, call-fact recap, turn filler - by
IMPORTING it, not copying it. Nothing in app/, models/, or the docker
stack's config is written to by this file.

STT/LLM SHAPE (changed 2026-07-29 - see README.md's "Single-call live
path" section for the full validation trail before this was adopted):
  - STT: GemmaDirectAudioSTT (stt_plugin.py) - a CHEAP, transcript-only
    Gemma call. Exists ONLY so the emergency safety gate
    (on_user_turn_completed, app/main.py) has real transcript text to
    run its keyword match on BEFORE any reply is generated - that gate
    must be able to bypass the LLM entirely on a match, which requires
    text to exist first no matter how the reply itself is produced.
  - LLM: GemmaDirectAudioLLM (llm_plugin.py) - the REAL reply (including
    tool calls) generated from the SAME raw audio directly, in one
    call, NOT from GemmaDirectAudioSTT's transcript. This is what makes
    the total turn cost competitive with the Whisper+text-LLM cascade:
    the earlier design (STT transcript feeding a SEPARATE text-only LLM
    call) paid for two full-cost Gemma calls per turn and was measurably
    SLOWER than the cascade (620ms vs 479ms, see benchmark.py). One
    combined audio-conditioned call was measured faster (324ms) -
    validated for tool-calling reliability in tests/test_llm_plugin.py
    (10/10 clean across repeated real calls, using the REAL
    RiversideReceptionist tools + real system prompt) before being wired
    in here.

WHY IMPORT RATHER THAN COPY: RiversideReceptionist doesn't reference STT
at all - STT is a constructor arg to AgentSession, not to the Agent
class. Swapping STT engines never required touching persona/tools/
safety-gate code in the first place, so importing the class directly
guarantees this pipeline can never silently drift from the hard-won
behavior documented throughout app/main.py (the CHAT_CTX_MAX_ITEMS
bound, the two-layer safety gate, the exact tool-call contracts) the way
a hand-copied version eventually would.

WHY THIS WORKER IS SAFE TO RUN ALONGSIDE app/main.py'S WORKER:
agent_name="direct-audio-receptionist" below turns on LiveKit's explicit
dispatch mode for this worker. Without it, both this worker and
app/main.py's worker would auto-dispatch and could both try to join the
same room. With it, this worker claims NOTHING by default - a room only
reaches it if a caller's token explicitly requests this agent name (or
the AgentDispatch API is used), so starting this worker cannot change
what app/main.py's existing, already-running worker does.

Run (repo root, same venv app/main.py uses):
    venv/bin/python direct_audio_agent/agent.py console   # mic/speaker, no server
    venv/bin/python direct_audio_agent/agent.py dev        # hot-reload dev worker
    venv/bin/python direct_audio_agent/agent.py start      # production mode
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
APP_DIR = THIS_DIR.parent / "app"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(APP_DIR / ".env")

from loguru import logger  # noqa: E402
from livekit import agents  # noqa: E402
from livekit.agents import (  # noqa: E402
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    inference,
    metrics,
)
from livekit.plugins import silero  # noqa: E402

import compat  # noqa: E402 - app/compat.py, unmodified
import latency_log  # noqa: E402 - app/latency_log.py, unmodified
import system_config  # noqa: E402 - app/system_config.py, unmodified
import main as riverside_main  # noqa: E402 - app/main.py, THE EXISTING AGENT, UNMODIFIED
from plugins.qwen_tts import QwenSubprocessTTS  # noqa: E402 - app/plugins, unmodified

from stt_plugin import GemmaDirectAudioSTT  # this folder's own module
from llm_plugin import GemmaDirectAudioLLM  # this folder's own module

AGENT_NAME = "direct-audio-receptionist"


def prewarm(proc: JobProcess) -> None:
    """Narrower than app/main.py's prewarm(): there is no STT engine to
    pick/load/warm here (GemmaDirectAudioSTT has no local model - the
    shared vLLM server IS the warm state, and entrypoint() below pings it
    via riverside_main._warm_up_vllm same as app/main.py does). VAD is
    still needed for turn segmentation - identical Silero setup."""
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=float(os.environ.get("VAD_MIN_SILENCE", "0.4"))
    )
    proc.userdata["qwen_omni_reachable"] = riverside_main._qwen_omni_reachable(
        riverside_main.QWEN_OMNI_BASE_URL
    )
    proc.userdata["qwen_shared_reachable"] = riverside_main._shared_qwen_tts_reachable(
        riverside_main.SHARED_TTS_BASE_URL
    )
    compat.warm_rag()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    cfg = system_config.get_config()
    served_model_name = cfg["llm"]["served_model_name"]

    # Same PC2-reachability TTS fallback chain as app/main.py's
    # entrypoint() - read there for the full reasoning; reused verbatim
    # via the same helper functions, not reimplemented.
    tts_cfg = dict(cfg["tts"])
    if tts_cfg.get("engine", "qwen_omni") in ("qwen_omni", "qwen") and not (
        ctx.proc.userdata.get("qwen_omni_reachable", True)
    ):
        if ctx.proc.userdata.get("qwen_shared_reachable", False):
            logger.warning(
                "direct-audio agent: PC2 unreachable, using shared local "
                "Qwen3-TTS for this call."
            )
            tts_cfg["engine"] = "qwen_shared"
        else:
            logger.warning(
                "direct-audio agent: PC2 unreachable and shared Qwen3-TTS "
                "not running, using local Qwen subprocess fallback."
            )
            tts_cfg["engine"] = "qwen_local_subprocess"

    tts_service = riverside_main._make_tts(tts_cfg)

    if isinstance(tts_service, QwenSubprocessTTS):
        await asyncio.gather(
            riverside_main._warm_up_vllm(served_model_name), tts_service.prewarm()
        )
    else:
        await asyncio.gather(
            riverside_main._warm_up_vllm(served_model_name),
            riverside_main._warm_up_qwen_omni(),
        )

    turn_detection = None
    try:
        # TUNED, diverging from app/main.py's 0.36/2.5 - see agent.py's
        # module docstring "Turn-detector tuning" section for the real
        # call (2026-07-29) that motivated this: one turn hit the FULL
        # 2.5s max_delay ceiling because the model's confidence the
        # utterance was complete fell below the stock English threshold
        # (0.36, hardcoded in livekit.agents.inference.eot.languages).
        # v1-mini's ceiling behavior is a binary confident/uncertain
        # classifier, not a graded wait - when uncertain, it ALWAYS pays
        # the full max_delay, no matter how uncertain. Lowering both
        # numbers directly targets that: a lower threshold means fewer
        # utterances get classified "uncertain" in the first place, and
        # a lower ceiling bounds the worst case when one still does.
        # TRADE-OFF, not a free win: both changes make the agent more
        # willing to jump in - a genuinely slow/thinking-pause speaker is
        # now more likely to get cut off than under app/main.py's more
        # conservative values. NOT YET VALIDATED against a second real
        # call - see the README's tracking note for what to check next.
        turn_detection = inference.TurnDetector(
            version="v1-mini", unlikely_threshold={"en": 0.22}
        )
        logger.info(
            "direct-audio agent: semantic turn detector ENABLED (v1-mini, "
            "tuned: unlikely_threshold[en]=0.22, see agent.py comments)."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"direct-audio agent: turn detector unavailable ({exc}), falling "
            "back to VAD-only endpointing."
        )

    # ONE shared instance: GemmaDirectAudioLLM reads last_turn_wav_bytes
    # off this SAME object (set by GemmaDirectAudioSTT.recognize() right
    # before the LLM is called for the same turn) - see llm_plugin.py's
    # module docstring for why the audio has to travel through this
    # side-channel instead of LiveKit's normal ChatContext.
    direct_audio_stt = GemmaDirectAudioSTT(
        base_url=riverside_main.VLLM_BASE_URL, model=served_model_name
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=direct_audio_stt,
        llm=GemmaDirectAudioLLM(
            audio_source=direct_audio_stt,
            base_url=riverside_main.VLLM_BASE_URL,
            model=served_model_name,
        ),
        tts=tts_service,
        # DIVERGES from app/main.py: max_delay 2.5 -> 1.5 (see
        # turn_detection comment above - bounds the worst-case dead-air
        # ceiling, real accepted risk of more interruptions).
        #
        # min_delay 0.1 -> 0.7 - a SECOND real fix, found the same way as
        # the first: a real call on 2026-07-29 (via call_server.py, the
        # single-call live design's first end-to-end test) logged
        # "transcript arrives after turn has been committed. consider
        # raising `min_delay`..." repeatedly. Root cause: app/main.py's
        # 0.1 assumes a fast STT (Whisper: ~0.1-0.22s per
        # benchmark.py). GemmaDirectAudioSTT is NOT that fast - measured
        # 0.18-0.66s in that same real call (audio-conditioned Gemma
        # calls cost more than a purpose-built ASR model, see README's
        # "hard physical limit" note). With min_delay below the STT's
        # own floor, a turn could commit via VAD before the safety-gate
        # transcript even exists. 0.7s comfortably covers the observed
        # range with margin. NOT a full proof this can never race under
        # worse timing (e.g. very rapid back-to-back speech) - see
        # README's "Known gaps" for what "on_user_turn_completed DID fire
        # every turn in the test call" does and doesn't establish.
        #
        # preemptive_generation is a structural no-op here regardless of
        # its setting: it triggers on INTERIM STT results, and
        # GemmaDirectAudioSTT declares interim_results=False (one
        # non-streaming call per turn) - left enabled only for parity
        # with app/main.py, not because it does anything in this
        # pipeline.
        turn_handling={
            "turn_detection": turn_detection,
            "endpointing": {"mode": "dynamic", "min_delay": 0.7, "max_delay": 1.5},
            "preemptive_generation": {"enabled": True, "preemptive_tts": False},
        },
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev) -> None:
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)
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
            # Tagged distinctly from app/main.py's own latency_log entries
            # (engine="gemma_direct_audio") so the two pipelines' numbers
            # in latency_log.jsonl never get averaged together - see
            # benchmark.py, which reads this same log.
            tagged_cfg = {**cfg, "stt": {"engine": "gemma_direct_audio", "model": served_model_name}}
            latency_log.record(tagged_cfg, fields)

    async def _log_usage() -> None:
        logger.info(f"direct-audio agent: session usage summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(_log_usage)

    agent = riverside_main.RiversideReceptionist()

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev) -> None:
        if ev.new_state == "speaking":
            agent.cancel_pending_filler()

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,
            initialize_process_timeout=300.0,
        )
    )
