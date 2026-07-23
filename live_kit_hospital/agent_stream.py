"""
Riverside General voice receptionist - LiveKit Agents port, NEMOTRON STT VARIANT.

This file is agent.py with exactly one thing swapped: STT. Everything else
below (TTS engine selection, the vLLM-backed LLM, the safety gate, the four
booking/info tools, semantic turn detection, metrics/latency logging, and
worker plumbing) is copied verbatim from agent.py - see that file's module
docstring for the full Pipecat -> LiveKit migration notes if you need them.

What changed and why:

  agent.py                                agent_nemotron.py (this file)
  ---------------------------------------------------------------------------
  _make_stt(cfg["stt"]) - reads         -> NemotronSTT(base_url=...,
  system_config's stt.engine at the         language=...) instantiated
  top of prewarm() and picks whisper /      directly in prewarm(). No
  Parakeet(subprocess or in-process) /      system_config read for STT at
  Canary, each with its own env-var-        all - this file only ever runs
  gated fallback-to-whisper chain.          one engine, on purpose, "for now".
                                             See plugins/nemotron_stt.py for
                                             the plugin itself (talks to the
                                             standalone nemotron_stt_server.py
                                             process over its /v1/audio/stream
                                             WebSocket).

Everything downstream of prewarm() - entrypoint(), the AgentSession
construction, the Agent class - reads ctx.proc.userdata["stt"] exactly like
agent.py does, and has no idea which engine is behind it. That's the whole
reason this could be a one-function change instead of a rewrite.

Before running this: nemotron_stt_server.py must already be up and reachable
at NEMOTRON_STT_BASE_URL (default ws://localhost:8010). If it's running on a
different machine on your LAN (e.g. your friend's PC, alongside the vLLM
server), point the env var at that machine instead:

    NEMOTRON_STT_BASE_URL=ws://<that-machine's-LAN-IP>:8010

NemotronSTT.load() (called from prewarm(), below) only logs an error and
continues if the server isn't reachable at worker-startup time - it does NOT
crash the worker the way a missing Parakeet/Canary subprocess venv can. That
means a typo'd IP or a not-yet-started nemotron_stt_server.py will look like
a running worker that goes silent/errors on the first real call, not a
startup crash - check the worker's logs for the "Nemotron STT server
unreachable" line if calls aren't transcribing.

Run (identical to agent.py):
    python agent_nemotron.py dev        # hot-reload dev worker
    python agent_nemotron.py console    # terminal mode: local mic/speaker
    python agent_nemotron.py start      # production mode
"""

from __future__ import annotations

import asyncio
import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RunContext,
    StopResponse,
    function_tool,
    metrics,
)
from livekit.plugins import openai, silero

# Imported at module level (not lazily inside entrypoint) so this plugin
# REGISTERS ITSELF with `python agent_nemotron.py download-files` - a lazy
# import inside entrypoint() never runs during that CLI command, which is
# why only openai/silero showed up on the first download-files pass in the
# original agent.py before this comment was added there.
try:
    from livekit.plugins.turn_detector.english import EnglishModel

    _TURN_DETECTOR_INSTALLED = True
except ImportError:
    EnglishModel = None
    _TURN_DETECTOR_INSTALLED = False

import compat
import latency_log
import system_config
from helpers import DEFAULT_FILLERS, push_ui, run_with_filler
from plugins.nemotron_stt import NemotronSTT
from plugins.qwen_tts import QwenSubprocessTTS
from prompts import GREETING_INSTRUCTIONS, build_system_prompt

load_dotenv()

# --- shared config -----------------------------------------------------
# INFRASTRUCTURE env vars: where things live / how to reach them. These
# rarely change and are set once in .env.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "300"))

# STT is HARDCODED to Nemotron in this file - that's the whole point of it.
# No system_config.stt read, no engine switch. base_url defaults to
# nemotron_stt_server.py's own default port; override if that server runs
# on another machine (see module docstring above). language defaults to
# "en-US" to match the English-only semantic turn detector below; pass
# "auto" via env var if you need the model to detect the language itself.
NEMOTRON_STT_BASE_URL = os.environ.get("NEMOTRON_STT_BASE_URL", "ws://localhost:8010")
NEMOTRON_STT_LANGUAGE = os.environ.get("NEMOTRON_STT_LANGUAGE", "en-US")

# SELECTION config for TTS only: WHICH of the available TTS backends is
# active RIGHT NOW, same as agent.py. Still comes from system_config.py so
# a developer's choice from the dev UI takes effect - see that module's
# docstring for the exact "takes effect on next call" contract. Read fresh
# in entrypoint() below, never cached at module level.


def _make_tts(tts_cfg: dict):
    """Chosen by tts_cfg = {"engine": ..., "model": ...} from
    system_config (schema v2). "model" means different things per engine:
    Qwen -> speaker name, Chatterbox -> unused (single default voice),
    Kokoro -> voice pack name, Piper -> .onnx file path.

    Each candidate engine falls back to Qwen if its required env vars
    aren't set - same graceful-degradation pattern used throughout this
    project rather than crashing the call. Unchanged from agent.py."""
    engine = tts_cfg.get("engine", "qwen")
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
            "aren't set in .env - falling back to Qwen."
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
            "in .env - falling back to Qwen."
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
            "in .env - falling back to Qwen."
        )

    logger.info("TTS: Qwen (subprocess, proven)")
    return QwenSubprocessTTS()


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


# ---------------------------------------------------------------------------
# The agent: persona + safety gate + tools (unchanged from agent.py)
# ---------------------------------------------------------------------------
class RiversideReceptionist(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=build_system_prompt())

    # ------------------------------------------------------------- lifecycle
    async def on_enter(self) -> None:
        # Light up the frontend directory board the moment the agent joins.
        await push_ui({"type": "roster", "doctors": compat.get_doctor_roster()})
        await push_ui({"type": "status", "state": "listening"})
        self.session.generate_reply(instructions=GREETING_INSTRUCTIONS)

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
        text = (getattr(new_message, "text_content", None) or "").strip()
        if not text:
            return
        hit = compat.run_safety_gate(text)
        if hit is None:
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
        result = await run_with_filler(
            context.session,
            compat.check_availability(department=department, date=date, doctor=doctor),
            filler=DEFAULT_FILLERS["check_availability"],
        )
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
        result = await run_with_filler(
            context.session,
            compat.cancel_appointment(
                patient_name=patient_name, date=date, department=department, time=time
            ),
            filler=DEFAULT_FILLERS["cancel_appointment"],
        )
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
    """Runs once per worker process, BEFORE any call is accepted - same
    timing/purpose as agent.py's prewarm(). The only change: STT is built
    directly as NemotronSTT here instead of via _make_stt(cfg["stt"]), so
    there's no system_config read for STT in this file at all.

    stt_service.load() health-checks nemotron_stt_server.py right now, at
    worker startup - see the module docstring above for what happens if
    that server isn't reachable yet (it warns and continues, it does not
    crash prewarm).

    CROSS-PROCESS NOTE (same caveat as agent.py, for the TTS switch that
    still goes through system_config): a developer's dev-UI TTS-backend
    switch takes effect for any NEWLY SPAWNED worker process - an already-
    warmed, already-pooled process keeps whatever it was warmed with until
    LiveKit recycles it."""
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=float(os.environ.get("VAD_MIN_SILENCE", "0.4")),
    )
    stt_service = NemotronSTT(
        base_url=NEMOTRON_STT_BASE_URL, language=NEMOTRON_STT_LANGUAGE
    )
    stt_service.load()
    proc.userdata["stt"] = stt_service
    compat.warm_rag()


async def _warm_up_vllm(served_model_name: str) -> None:
    """One tiny completion so the first real turn doesn't eat vLLM's
    cold-start spike (~2s observed live in the old build). Unchanged from
    agent.py - works the same whether vLLM is on this machine or on your
    friend's PC, since VLLM_BASE_URL is just an env var either way."""
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


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # Read fresh HERE, at the start of every call - same contract as
    # agent.py. Only used for llm/tts now (STT is fixed for this file, so
    # it doesn't need cfg at all).
    cfg = system_config.get_config()
    served_model_name = cfg["llm"]["served_model_name"]

    tts_service = _make_tts(cfg["tts"])
    await asyncio.gather(_warm_up_vllm(served_model_name), tts_service.prewarm())

    # Semantic turn detection: the single highest-leverage latency change of
    # this whole migration. Falls back to plain VAD end-pointing if the
    # plugin isn't installed, so the agent still runs. Unchanged from
    # agent.py - NemotronSTT's interim_results=True capability feeds this
    # the same way any other streaming STT would.
    turn_detection = None
    if _TURN_DETECTOR_INSTALLED:
        try:
            turn_detection = EnglishModel()
            logger.info("Semantic turn detector: ENABLED (english).")
        except Exception as exc:  # noqa: BLE001
            # Most common cause: `python agent_nemotron.py download-files`
            # was never run (or ran before this plugin was registered), so
            # the model's languages.json etc. aren't on disk yet. This must
            # NOT crash the whole job - fall back to VAD-only endpointing.
            logger.warning(
                f"Turn detector model unavailable ({exc}) - falling back to "
                "VAD-only endpointing. Run "
                "`python agent_nemotron.py download-files` once to fetch "
                "the model, then restart to re-enable it."
            )
    else:
        logger.warning(
            "livekit-agents[turn-detector] not installed - falling back to "
            "VAD-only endpointing (works, but you lose the latency win)."
        )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=ctx.proc.userdata["stt"],
        llm=openai.LLM(
            model=served_model_name,
            base_url=VLLM_BASE_URL,
            api_key="not-needed",  # vLLM ignores it; the plugin requires one
        ),
        tts=tts_service,
        turn_detection=turn_detection,
        min_endpointing_delay=0.1,
        max_endpointing_delay=6.0,
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

    await session.start(
        room=ctx.room,
        agent=RiversideReceptionist(),
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
            # DEFAULT IS 10s. In agent.py this chain was Silero VAD -> STT
            # (whisper, in-process Parakeet, or the Parakeet SUBPROCESS
            # itself needing ~15-20s in its own venv) -> the RAG
            # sentence-transformers embedder. Here, STT is just an HTTP
            # health probe against an already-running nemotron_stt_server.py
            # (fast, typically well under a second) rather than an in-
            # process/subprocess model load - so this file's cold-start
            # budget is dominated by Silero VAD + the RAG embedder, not STT.
            # Keeping the same generous 300s anyway: it costs nothing on the
            # happy path, and VAD+RAG alone can still be slow under GPU
            # contention from vLLM/Qwen already running on this machine.
            initialize_process_timeout=300.0,
        )
    )
