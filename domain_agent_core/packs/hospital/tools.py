"""Hospital domain tools - Phase 0 regression baseline.

Ported from ``app/main.py:852-1072``'s five ``@function_tool`` bound
methods, now free functions (per ``core/tool_registry.py``'s design) that
reach the active ``BaseDomainAgent`` via ``context.session.current_agent``
for the recap/filler helpers. Business logic is NOT reimplemented here -
every call forwards straight into ``app/compat.py`` -> ``app/hospital_core/
*``, the exact real (or demo-fallback) modules the existing hospital
pipeline uses, so this pack's behavior is bit-for-bit identical to
``app/main.py``'s today.
"""

from __future__ import annotations

import sys
from pathlib import Path

from livekit.agents import RunContext, function_tool
from loguru import logger

_APP_DIR = Path(__file__).resolve().parents[3] / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import compat
from helpers import DEFAULT_FILLERS

from domain_agent_core.core.turn_filler import run_with_filler

# Every hospital tool is backed by real SQLite/RAG lookups (compat.py ->
# hospital_core/*), never an LLM-asserted fact - see
# core/tool_registry.check_deterministic_backing().
DETERMINISTIC = {
    "check_availability",
    "book_appointment",
    "cancel_appointment",
    "update_appointment",
    "search_hospital_info",
}


@function_tool
async def check_availability(
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
    logger.info(
        f"tool call: check_availability(department={department!r}, "
        f"date={date!r}, doctor={doctor!r})"
    )
    agent = context.session.current_agent
    await agent.remember(department=department, date=date, doctor=doctor)
    result = await run_with_filler(
        context.session,
        compat.check_availability(department=department, date=date, doctor=doctor),
        filler=DEFAULT_FILLERS["check_availability"],
    )
    logger.info(f"tool result: check_availability -> status={result.get('status')!r}")
    await agent.record_tool_call(
        "check_availability",
        {"department": department, "date": date, "doctor": doctor},
        result,
    )
    return result


@function_tool
async def book_appointment(
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
    agent = context.session.current_agent
    await agent.remember(
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
    await agent.record_tool_call(
        "book_appointment",
        {
            "patient_name": patient_name,
            "department": department,
            "date": date,
            "time": time,
            "doctor": doctor,
        },
        result,
    )
    return result


@function_tool
async def cancel_appointment(
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
    agent = context.session.current_agent
    await agent.remember(patient_name=patient_name, department=department, date=date, time=time)
    result = await run_with_filler(
        context.session,
        compat.cancel_appointment(
            patient_name=patient_name, date=date, department=department, time=time
        ),
        filler=DEFAULT_FILLERS["cancel_appointment"],
    )
    logger.info(f"tool result: cancel_appointment -> status={result.get('status')!r}")
    await agent.record_tool_call(
        "cancel_appointment",
        {"patient_name": patient_name, "date": date, "department": department, "time": time},
        result,
    )
    return result


@function_tool
async def update_appointment(
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
    agent = context.session.current_agent
    await agent.remember(
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
    await agent.record_tool_call(
        "update_appointment",
        {
            "patient_name": patient_name,
            "new_date": new_date,
            "new_time": new_time,
            "date": date,
            "department": department,
            "time": time,
        },
        result,
    )
    return result


@function_tool
async def search_hospital_info(context: RunContext, query: str) -> dict:
    """Look up general hospital information: hours, departments, doctor
    bios, insurance, billing, prescriptions, visiting policy, lab
    results, parking. Use this BEFORE ever saying you don't have some
    piece of hospital information. Do NOT use it for appointment
    availability or booking.

    Args:
        query: The caller's question, rephrased as a short search query.
    """
    result = await run_with_filler(
        context.session,
        compat.search_hospital_info(query=query),
        filler=DEFAULT_FILLERS["search_hospital_info"],
    )
    agent = context.session.current_agent
    await agent.record_tool_call("search_hospital_info", {"query": query}, result)
    return result


def get_doctor_roster() -> list[dict]:
    """Thin re-export of ``compat.get_doctor_roster()`` for
    ``worker.py``'s ``on_enter``-equivalent roster push."""
    return compat.get_doctor_roster()
