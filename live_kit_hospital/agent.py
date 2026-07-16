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
  faster-whisper STT                   -> NVIDIA Parakeet TDT by default
                                          (STT_BACKEND=parakeet), whisper
                                          kept one env var away
                                          (STT_BACKEND=whisper) as the
                                          proven fallback.
  tuned VAD stop_secs (0.3, risky)     -> Silero VAD + the semantic turn-
                                          detector model.
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
# REGISTERS ITSELF with `python agent.py download-files` - a lazy import
# inside entrypoint() never runs during that CLI command, which is why
# only openai/silero showed up on the first download-files pass.
try:
    from livekit.plugins.turn_detector.english import EnglishModel

    _TURN_DETECTOR_INSTALLED = True
except ImportError:
    EnglishModel = None
    _TURN_DETECTOR_INSTALLED = False

import compat
from helpers import DEFAULT_FILLERS, push_ui, run_with_filler
from plugins.qwen_tts import QwenSubprocessTTS
from prompts import GREETING_INSTRUCTIONS, build_system_prompt

load_dotenv()

# --- shared config (env-overridable) ----------------------------------------
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
GEMMA_MODEL_NAME = os.environ.get("GEMMA_MODEL_NAME", "gemma-4-12b-w4a16")
STT_BACKEND = os.environ.get("STT_BACKEND", "parakeet").lower()
PARAKEET_MODEL = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "distil-large-v3")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "300"))


def _make_stt():
    """Parakeet TDT by default (SOTA local: leaderboard-topping accuracy,
    RTFx fast enough that a whole utterance transcribes in tens of ms).

    Three-way fallback chain, in order:
      1. PARAKEET_PYTHON is set -> subprocess plugin. This is the path that
         actually works when the main agent's venv is Python 3.12, since
         NeMo's ASR extras currently fail to build there (see README) -
         Parakeet runs isolated in its own 3.10/3.11 venv instead, same
         architecture as the Qwen TTS worker.
      2. PARAKEET_PYTHON unset but `nemo` importable HERE -> in-process
         plugin (only works if your main venv itself is 3.10/3.11).
      3. Neither -> faster-whisper, the proven fallback. Also handles Urdu
         code-switching better (Parakeet v2 is English-only)."""
    if STT_BACKEND == "parakeet":
        if os.environ.get("PARAKEET_PYTHON"):
            from plugins.parakeet_stt_subprocess import ParakeetSubprocessSTT

            logger.info(
                f"STT backend: Parakeet (subprocess, venv={os.environ['PARAKEET_PYTHON']})"
            )
            return ParakeetSubprocessSTT()
        try:
            from plugins.parakeet_stt import ParakeetSTT

            logger.info(f"STT backend: Parakeet (in-process, {PARAKEET_MODEL})")
            return ParakeetSTT(model=PARAKEET_MODEL)
        except ImportError as exc:
            logger.warning(
                f"STT_BACKEND=parakeet but NeMo isn't importable here ({exc}) and "
                "PARAKEET_PYTHON isn't set - falling back to faster-whisper. Set "
                "PARAKEET_PYTHON/PARAKEET_WORKER in .env to use the subprocess "
                "plugin instead (see README)."
            )
    from plugins.whisper_stt import FasterWhisperSTT

    logger.info(f"STT backend: faster-whisper ({WHISPER_MODEL})")
    return FasterWhisperSTT(model=WHISPER_MODEL)


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
# The agent: persona + safety gate + tools
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
    """Runs once per worker process, BEFORE any call is accepted. All the
    warm-up lessons from the Pipecat build live here or in entrypoint():
    a cold anything (VAD, STT, RAG embedder, vLLM, TTS CUDA graphs) must
    never be paid for by a real caller's first turn."""
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=float(os.environ.get("VAD_MIN_SILENCE", "0.4")),
    )
    stt_service = _make_stt()
    stt_service.load()
    proc.userdata["stt"] = stt_service
    compat.warm_rag()


async def _warm_up_vllm() -> None:
    """One tiny completion so the first real turn doesn't eat vLLM's
    cold-start spike (~2s observed live in the old build)."""
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json={
                    "model": GEMMA_MODEL_NAME,
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": "ping"}],
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                await resp.read()
        logger.info("vLLM warm-up ping OK.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"vLLM warm-up ping failed (continuing): {exc}")


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    tts_service = QwenSubprocessTTS()
    await asyncio.gather(_warm_up_vllm(), tts_service.prewarm())

    # Semantic turn detection: the single highest-leverage latency change of
    # this whole migration. Falls back to plain VAD end-pointing if the
    # plugin isn't installed, so the agent still runs.
    turn_detection = None
    if _TURN_DETECTOR_INSTALLED:
        try:
            turn_detection = EnglishModel()
            logger.info("Semantic turn detector: ENABLED (english).")
        except Exception as exc:  # noqa: BLE001
            # Most common cause: `python agent.py download-files` was never
            # run (or ran before this plugin was registered), so the
            # model's languages.json etc. aren't on disk yet. This must NOT
            # crash the whole job - fall back to VAD-only endpointing.
            logger.warning(
                f"Turn detector model unavailable ({exc}) - falling back to "
                "VAD-only endpointing. Run `python agent.py download-files` "
                "once to fetch the model, then restart to re-enable it."
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
            model=GEMMA_MODEL_NAME,
            base_url=VLLM_BASE_URL,
            api_key="not-needed",  # vLLM ignores it; the plugin requires one
        ),
        tts=tts_service,
        turn_detection=turn_detection,
    )

    # --- observability: per-stage latency (STT / LLM TTFT / TTS TTFB / EOU)
    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev) -> None:
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

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
            # DEFAULT IS 10s. prewarm() loads Silero VAD + STT (whisper or
            # Parakeet) + the RAG sentence-transformers embedder - on a
            # cold cache (first run, or first HF download) that combination
            # can easily exceed 10s, and LiveKit kills the process mid-load
            # with no useful error beyond "no process became available".
            # 120s gives real headroom for a first-time model download;
            # once everything is cached locally this returns in a few
            # seconds and the extra timeout costs nothing.
            initialize_process_timeout=120.0,
        )
    )
