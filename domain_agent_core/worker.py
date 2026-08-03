"""domain_agent_core's generic LiveKit worker entrypoint.

ONE file, every domain: which pack runs is selected entirely by the
``DOMAIN_PACK`` environment variable (fail closed if unset), assembled via
``core.domain_loader.AgentAssembler``. This is the literal mechanism that
satisfies "no forking code per domain" - a hospital agent and a banking
agent are this same file, differing only in which manifest gets loaded.

Run (repo root, same venv app/main.py uses):
    DOMAIN_PACK=hospital venv/bin/python domain_agent_core/worker.py console
    DOMAIN_PACK=banking  venv/bin/python domain_agent_core/worker.py console
    DOMAIN_PACK=hospital venv/bin/python domain_agent_core/worker.py start
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from dotenv import load_dotenv

load_dotenv(_APP_DIR / ".env")

from livekit import agents
from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    inference,
)
from livekit.plugins import silero
from loguru import logger
from plugins.qwen_tts import QwenSubprocessTTS

from domain_agent_core.core import engine_factory
from domain_agent_core.core.agent_base import BaseDomainAgent
from domain_agent_core.core.domain_loader import (
    AgentAssembler,
    AssembledDomain,
)
from domain_agent_core.core.turn_filler import prerender_filler_audio

DOMAIN_PACK = os.environ.get("DOMAIN_PACK")
if not DOMAIN_PACK:
    raise RuntimeError(
        "DOMAIN_PACK environment variable is not set - which domain pack "
        "should this worker run? (e.g. DOMAIN_PACK=hospital or "
        "DOMAIN_PACK=banking). Fail closed rather than guessing."
    )

_assembled: AssembledDomain = AgentAssembler().assemble(DOMAIN_PACK)
logger.info(
    f"domain_agent_core: assembled domain {DOMAIN_PACK!r} "
    f"(agent_name={_assembled.pack.agent_name!r}, "
    f"compliance={_assembled.compliance.name!r})"
)


def prewarm(proc: JobProcess) -> None:
    """Runs once per worker process, before any call is accepted -
    ported from ``app/main.py``'s ``prewarm()``, generalized to this
    domain's own engine config and RAG engine."""
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=float(os.environ.get("VAD_MIN_SILENCE", "0.4"))
    )
    proc.userdata["stt"] = engine_factory.make_stt(_assembled.engines["stt"])
    proc.userdata["stt"].load() if hasattr(proc.userdata["stt"], "load") else None
    proc.userdata["qwen_omni_reachable"] = engine_factory.qwen_omni_reachable()
    proc.userdata["qwen_shared_reachable"] = engine_factory.shared_qwen_tts_reachable()
    if _assembled.rag_engine is not None:
        _assembled.rag_engine.try_load_embedder()


async def entrypoint(ctx: JobContext) -> None:
    """Assemble one call's ``AgentSession`` for the domain this worker
    process was started with - generalized from ``app/main.py``'s
    ``entrypoint()``."""
    await ctx.connect()

    served_model_name = _assembled.engines["llm"].get("served_model_name") or _assembled.engines[
        "llm"
    ].get("engine", "")

    tts_cfg = dict(_assembled.engines["tts"])
    if tts_cfg.get("engine", "qwen_omni") in ("qwen_omni", "qwen") and not (
        ctx.proc.userdata.get("qwen_omni_reachable", True)
    ):
        if ctx.proc.userdata.get("qwen_shared_reachable", False):
            tts_cfg["engine"] = "qwen_shared"
        else:
            tts_cfg["engine"] = "qwen_local_subprocess"

    tts_service = engine_factory.make_tts(tts_cfg)

    # Same warm-up shape as app/main.py's entrypoint(): the local-
    # subprocess TTS path manages its own async prewarm(); every other
    # path gets warmed via a real HTTP ping instead (qwen_omni). Checked
    # via isinstance, NOT hasattr(tts_service, "prewarm") - the base
    # livekit.agents.tts.TTS class already defines a synchronous no-op
    # prewarm() that every plugin inherits, so hasattr() is true for all
    # of them and silently breaks asyncio.gather() below (a sync None
    # isn't awaitable) for every engine except QwenSubprocessTTS. Filler-
    # phrase pre-rendering always runs alongside, since it needs the
    # actual TTS engine/voice for this call, not just-in-time at
    # prewarm() time.
    tts_warmup = (
        tts_service.prewarm()
        if isinstance(tts_service, QwenSubprocessTTS)
        else engine_factory.warm_up_qwen_omni()
    )
    _, _, filler_audio_cache = await asyncio.gather(
        engine_factory.warm_up_vllm(served_model_name),
        tts_warmup,
        prerender_filler_audio(tts_service, _assembled.pack.filler.phrases),
    )

    turn_detection = None
    try:
        turn_detection = inference.TurnDetector(version="v1-mini", unlikely_threshold={"en": 0.15})
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Turn detector model unavailable ({exc}) - VAD-only endpointing.")

    from livekit.plugins import openai

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=ctx.proc.userdata["stt"],
        llm=openai.LLM(
            model=served_model_name,
            base_url=engine_factory.VLLM_BASE_URL,
            api_key="not-needed",
            max_completion_tokens=int(os.environ.get("MAX_TOKENS", "300")),
            extra_body={"logit_bias": {"100": -100}},
        ),
        tts=tts_service,
        turn_handling={
            "turn_detection": turn_detection,
            "endpointing": {"mode": "dynamic", "min_delay": 0.1, "max_delay": 0.8},
            "preemptive_generation": {"enabled": True, "preemptive_tts": False},
        },
    )

    agent = BaseDomainAgent(_assembled)
    agent.filler.filler_audio = filler_audio_cache

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev) -> None:
        if ev.new_state == "speaking":
            agent.cancel_pending_filler()

    await session.start(room=ctx.room, agent=agent, room_input_options=RoomInputOptions())


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=_assembled.pack.agent_name,
            initialize_process_timeout=300.0,
        )
    )
