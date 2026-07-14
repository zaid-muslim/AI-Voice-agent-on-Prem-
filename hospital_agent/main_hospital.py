"""
Hospital call agent - main pipeline (fictional test hospital, no real
patient data - see FICTIONAL TEST HOSPITAL note below).

    mic -> Whisper STT -> SafetyGateProcessor -> user_agg
        -> Gemma/vLLM (tools: check_availability, book_appointment,
           cancel_appointment, update_appointment, search_hospital_info)
        -> Qwen TTS -> speaker -> assistant_agg

Emergency transcripts are intercepted by SafetyGateProcessor BEFORE they
reach the LLM at all - see safety.py (pure logic, 15/15 self-test passing)
and safety_gate_processor.py (the pipecat wiring - flagged as not yet
end-to-end tested, see that file's docstring for the exact check to run
before trusting this on a real call).

RESERVATION FLOW: the system prompt instructs the LLM to call
check_availability, read the specific slot back to the caller, and only call
book_appointment after the caller confirms. book_appointment independently
re-validates the slot regardless (see booking.py - race-safety verified,
including the fix so an invalid DEPARTMENT is reported as such rather than
being confused with an invalid doctor).

CANCEL / RESCHEDULE FLOW: cancel_appointment and update_appointment look
the caller's appointment up by NAME (no confirmation code required) - see
booking.py's module docstring for the rationale and the caveat that name
matching has no fuzzy/typo tolerance yet. Both tools can return a "found
more than one appointment" message instead of acting; the system prompt
tells the LLM to relay that list and wait for the caller to narrow it down.

FUZZY MATCHING: booking.py returns a "Did you mean X?" suggestion when a
department or doctor name doesn't match anything closely enough to be a
typo/mishearing. The system prompt tells the LLM to relay that as a
question and wait for confirmation, never silently substituting it.

RAG (NEW): search_hospital_info (rag.py) answers general hospital
questions - hours, departments, doctors, insurance, billing, prescriptions,
visiting policy, lab results, parking - by retrieving from hospital_kb.py.
This REPLACES the old "I have no way to look up that information" stopgap:
the LLM now has a real tool for it and should use it instead of refusing.
Distinct from web_search/websearch.py (general internet search), which
remains intentionally NOT wired in - see the earlier conversation for why
open web search is the wrong tool for hospital-specific information.

NAME HANDLING FIX: the caller's name is conversational state, not a
database lookup - if they restate or correct it mid-call, the LLM should
just use whichever name was most recently given, with no tool call and no
"I don't have that information" response. (Previously, correcting a name
mid-conversation incorrectly triggered the "I can't look that up" stopgap.)

ESCALATION WHITELIST WIDENED: routine mentions of an existing condition
(e.g. "I have heart disease, I need a checkup") were previously at risk of
being misread as an emergency, the same false-positive class as "heart
checkup" being escalated to "call 911" in an earlier test. The prompt now
explicitly treats naming a known condition as routine care, not distress,
unless paired with acute symptom language.

ADAPTIVE FILLER: check_availability, book_appointment, cancel_appointment,
update_appointment, and search_hospital_info are all wrapped with
@with_adaptive_filler (see tool_filler.py, verified end-to-end) - a filler
phrase is only spoken if a call is still running past its threshold, never
on fast calls.

FICTIONAL TEST HOSPITAL: booking.py's seed data is synthetic
departments/doctors/slots for demo purposes only. No real patient data, not
HIPAA-reviewed, not connected to any real hospital system. Say so plainly if
this is ever shown to anyone as more than a prototype.

EMERGENCY_NUMBER: safety.py is set to 1122 (Pakistan's Rescue service,
confirmed operational in Islamabad via CARES 1122) - update this if
deployed elsewhere.

IMPORT PATHS: written as flat imports assuming this file sits alongside
safety.py, booking.py, rag.py, hospital_kb.py, tool_filler.py, and
qwen_bridge.py in one directory.
"""

import asyncio
from datetime import datetime

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_context import LLMContext, ToolsSchema
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

try:
    from .safety_gate_processor import SafetyGateProcessor
    from .booking import (
        check_availability,
        book_appointment,
        cancel_appointment,
        update_appointment,
    )
    from .rag import search_hospital_info
    from .qwen_bridge import QwenTTSService
except ImportError:
    from safety_gate_processor import SafetyGateProcessor
    from booking import (
        check_availability,
        book_appointment,
        cancel_appointment,
        update_appointment,
    )
    from rag import search_hospital_info
    from qwen_bridge import QwenTTSService

VLLM_BASE_URL = "http://localhost:8000/v1"
GEMMA_MODEL_NAME = "gemma-4-12b"

QWEN_TTS_MODEL_ID = "/home/nauyan/voice-agent-pipeline/models/Qwen3-TTS-0.6B-custom"
QWEN_TTS_SPEAKER = "aiden"
QWEN_TTS_LANGUAGE = "English"


def _build_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return (
        f"You are the phone receptionist for Riverside General, a test "
        f"hospital. Today's date is {today}. When the caller uses a "
        f"relative date like 'tomorrow', 'next Tuesday', or 'July 9th', "
        f"compute the actual YYYY-MM-DD date yourself using today's date - "
        f"never guess a date from an unrelated year. "
        "Answer in one or two short sentences unless more detail is "
        "genuinely needed. "
        "For NEW appointments: whenever a caller names ANY department or "
        "specialty for booking purposes, always call check_availability "
        "with that name immediately - even if you are not sure it is a "
        "real department. check_availability itself will tell you the "
        "correct list of departments if the name given is not valid, so "
        "you never have to judge that yourself. If the caller asks "
        "generally when a department has openings, without naming one "
        "specific date, call check_availability with just the department "
        "and leave the date empty - it will return whatever is open. "
        "Always call check_availability first, then read the specific "
        "doctor, date, and time back to the caller before booking - only "
        "call book_appointment after the caller clearly confirms that "
        "exact slot. Never invent a slot that check_availability did not "
        "actually return. "
        "If check_availability or book_appointment responds with "
        "'Did you mean X?' for a department or doctor name, treat that as "
        "a question to relay to the caller, word for word - ask them to "
        "confirm that specific name before you proceed. Do not silently "
        "substitute the suggested name yourself, and do not re-call the "
        "tool again until the caller has confirmed which name is correct. "
        "Before calling book_appointment, make sure you have the caller's "
        "full name - ask for it if they have not given it yet. The name is "
        "just information from the conversation, not something in any "
        "database yet - if the caller restates or corrects their name at "
        "any point before booking, simply use whatever name they most "
        "recently gave. This never needs a tool call or an 'I don't have "
        "that information' response - it is just tracking what they said. "
        "After book_appointment succeeds, clearly read back the full "
        "confirmation to the caller: the doctor, department, date, time, "
        "and the confirmation code, so they can write it down. "
        "When calling book_appointment or update_appointment, copy the "
        "time argument EXACTLY as check_availability returned it - "
        "24-hour HH:MM format like '11:00' or '15:30', with no AM/PM "
        "suffix and no reformatting - even though you should describe "
        "times in natural 12-hour format (like '11 AM') when speaking to "
        "the caller. These are two different things: speak naturally, "
        "call tools exactly. If book_appointment or update_appointment "
        "reports a slot as invalid, do NOT tell the caller there was a "
        "system error or that the slot is actually available - the "
        "tool's answer is correct, not a bug. Call check_availability "
        "again to get the exact current values, and retry with those "
        "values copied precisely. "
        "For CANCELLING an appointment: ask for the caller's full name, "
        "then call cancel_appointment with it (and the department, if they "
        "mention one). If it responds that it found more than one "
        "appointment under that name, read the list back to the caller "
        "exactly and ask them to tell you which department and/or date "
        "they mean, then call cancel_appointment again with that added "
        "detail - do not guess which one they meant. Before actually "
        "cancelling, confirm out loud with the caller which specific "
        "appointment (doctor, department, date, time) you are about to "
        "cancel, and only proceed once they confirm. After it succeeds, "
        "clearly confirm the cancellation was completed. "
        "For RESCHEDULING an appointment: ask for the caller's full name "
        "and the new date/time they want, then call update_appointment "
        "with the name and the new date/time (add department/date/time of "
        "the CURRENT appointment too if the caller mentions it, to help "
        "narrow things down). Same as cancelling - if it reports more than "
        "one appointment under that name, read the list back and ask which "
        "one before calling again. Before calling update_appointment, read "
        "the requested new date and time back to the caller and get their "
        "confirmation, the same way you would for a brand-new booking. "
        "update_appointment only changes the date/time of an existing "
        "appointment - it cannot change which doctor or department it is "
        "with. If a caller wants a different doctor or department, cancel "
        "the old appointment and book a completely new one instead. "
        "After update_appointment succeeds, clearly read back the new "
        "date and time to the caller. "
        "For general hospital information - hours, departments, doctors, "
        "insurance, billing, prescriptions, visiting policy, lab results, "
        "or parking - call search_hospital_info with the caller's "
        "question. Only say you don't have that information if "
        "search_hospital_info returns no matching information found; "
        "never guess or invent hospital hours, policies, or doctor "
        "details instead. "
        "This does NOT apply to appointment availability, dates, times, or "
        "department names - for those, always try check_availability "
        "instead of saying you don't know, even on a follow-up question "
        "like 'when is it open' - just call check_availability again, "
        "with no date, to get whatever is actually open. "
        "Routine appointment requests or mentions of an existing condition "
        "- like wanting a heart checkup, managing heart disease, a "
        "vaccination, or a general checkup - are NOT emergencies, even "
        "though they mention a body part, organ, or medical condition by "
        "name. Proceed normally with check_availability or "
        "search_hospital_info for these. Only escalate if the caller "
        "describes acute symptoms or distress happening right now - for "
        "example: chest pain, trouble breathing, severe bleeding, sudden "
        "numbness or slurred speech, or thoughts of self-harm. In those "
        "specific cases, gently but firmly tell them to hang up and call "
        "emergency services or go to the nearest ER yourself - do not "
        "attempt to handle it, even though a dedicated safety check also "
        "runs separately on every turn. When in doubt about a routine "
        "appointment request or a mention of a known condition, proceed "
        "normally rather than escalating."
    )


MAX_TOKENS = 150
VAD_STOP_SECS = 0.5


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        )
    )

    stt = WhisperSTTService(
        model="distil-medium.en",
        device="cuda",
        compute_type="int8_float16",
        ttfs_p99_latency=0.2,
    )

    safety_gate = SafetyGateProcessor()

    llm_text = OpenAILLMService(
        api_key="not-needed",
        base_url=VLLM_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=GEMMA_MODEL_NAME,
            max_tokens=MAX_TOKENS,
        ),
    )
    llm_text.register_direct_function(check_availability)
    llm_text.register_direct_function(book_appointment)
    llm_text.register_direct_function(cancel_appointment)
    llm_text.register_direct_function(update_appointment)
    llm_text.register_direct_function(search_hospital_info)

    tts = QwenTTSService(
        model_id=QWEN_TTS_MODEL_ID,
        speaker=QWEN_TTS_SPEAKER,
        language=QWEN_TTS_LANGUAGE,
        sample_rate=24000,
        chunk_size=8,
    )

    context = LLMContext(
        messages=[{"role": "system", "content": _build_system_prompt()}],
        tools=ToolsSchema(
            standard_tools=[
                check_availability,
                book_appointment,
                cancel_appointment,
                update_appointment,
                search_hospital_info,
            ]
        ),
    )
    aggregators = LLMContextAggregatorPair(context)
    user_aggregator = aggregators.user()
    assistant_aggregator = aggregators.assistant()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            safety_gate,  # emergency transcripts are intercepted here,
            # before the LLM ever sees them
            user_aggregator,
            llm_text,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
        ),
        idle_timeout_secs=1200,
    )

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
