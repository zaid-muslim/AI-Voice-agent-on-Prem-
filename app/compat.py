
from __future__ import annotations

import asyncio
import difflib
import inspect
import os
import re
import sys
import uuid as _uuid
from datetime import date as _date
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

# Make ./hospital_core importable exactly as-is (the files import each
# other with bare names like `from hospital_kb import ...` in script mode).
_CORE_DIR = Path(__file__).parent / "hospital_core"
if _CORE_DIR.is_dir() and str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))


def _try_import(name: str):
    try:
        module = __import__(name)
        logger.info(f"compat: using the REAL module for '{name}'")
        return module
    except ImportError:
        logger.warning(
            f"compat: '{name}.py' not found in hospital_core/ - demo fallback active"
        )
        return None


_booking = _try_import("booking")
_rag = _try_import("rag")
_safety = _try_import("safety")
_kb = _try_import("hospital_kb")

# canonical_name -> real_name, if they ever diverge again.
_KWARG_ALIASES: dict[str, str] = {}


async def _call(fn: Callable, /, **kwargs) -> Any:
    """Call a target with only the kwargs it accepts; await if async."""
    renamed = {_KWARG_ALIASES.get(k, k): v for k, v in kwargs.items()}
    try:
        accepted = set(inspect.signature(fn).parameters)
        filtered = {k: v for k, v in renamed.items() if k in accepted}
    except (TypeError, ValueError):
        filtered = renamed
    result = fn(**filtered)
    if inspect.isawaitable(result):
        result = await result
    return result


# ---------------------------------------------------------------------------
# SAFETY GATE (real: safety.py, 15/15 self-test verified in this revision)
# ---------------------------------------------------------------------------

# Same env var + default as hospital_core/safety.py's EMERGENCY_NUMBER -
# this fallback only activates when hospital_core itself failed to import
# (see run_safety_gate() below), so it can't just import that module's
# constant directly, but it can still read the same env var instead of
# hardcoding an independent copy that could silently drift from it.
_EMERGENCY_NUMBER = os.environ.get("EMERGENCY_NUMBER", "1122")

_FALLBACK_PATTERNS: list[tuple[str, str]] = [
    (
        "cardiac",
        r"\b(chest pain|heart attack|crushing (pain|pressure) in (my|the) chest)\b",
    ),
    (
        "breathing",
        r"\b(can\W?t breathe|cannot breathe|trouble breathing|struggling breathing|choking)\b",
    ),
    (
        "stroke",
        r"\b(face is drooping|slurred speech|sudden numbness|worst headache of my life)\b",
    ),
    (
        "trauma_bleeding",
        r"\b(severe bleeding|bleeding (heavily|a lot)|been (shot|stabbed)|unconscious|unresponsive)\b",
    ),
    ("overdose", r"\b(overdos\w+|took too many (pills|tablets))\b"),
    (
        "self_harm",
        r"\b((want|going|planning) to kill myself|end my life|suicidal|hurt myself|harm myself)\b",
    ),
]

_FALLBACK_MESSAGE = (
    f"This sounds like it could be a medical emergency. Please hang up right "
    f"now and call {_EMERGENCY_NUMBER}, or go to your nearest emergency room "
    f"immediately."
)


def run_safety_gate(text: str) -> Optional[dict]:
    """Deterministic, zero-LLM emergency check. None on ordinary text, or a
    dict with at least a 'message' key on a match."""
    if _safety is not None and hasattr(_safety, "run_safety_gate"):
        return _safety.run_safety_gate(text)

    lowered = text.lower()
    for category, pattern in _FALLBACK_PATTERNS:
        if re.search(pattern, lowered):
            return {
                "emergency": True,
                "category": category,
                "kind": "fallback_regex",
                "matched_text": "",
                "message": _FALLBACK_MESSAGE,
            }
    return None


# ---------------------------------------------------------------------------
# DOCTOR ROSTER (real: hospital_kb.get_doctor_roster(), verified present)
# ---------------------------------------------------------------------------

_FALLBACK_DOCTORS: list[dict] = [
    {
        "name": "Dr. Imran Malik",
        "department": "Cardiology",
        "bio": "Cardiologist. General adult cardiology and preventive heart care.",
    },
    {
        "name": "Dr. Ayesha Siddiqui",
        "department": "Cardiology",
        "bio": "Cardiologist. Cardiac rehabilitation and follow-up after cardiac events.",
    },
    {
        "name": "Dr. Bilal Ahmed",
        "department": "General Medicine",
        "bio": "General physician. Routine checkups, illness, and referrals.",
    },
    {
        "name": "Dr. Sana Farooqi",
        "department": "Pediatrics",
        "bio": "Pediatrician. Infancy through age 17, vaccinations, wellness checks.",
    },
]


def get_doctor_roster() -> list[dict]:
    if _kb is not None:
        for attr in ("get_doctor_roster", "DOCTORS", "doctors"):
            source = getattr(_kb, attr, None)
            if callable(source):
                try:
                    return list(source())
                except Exception:
                    logger.exception("compat: hospital_kb roster hook failed")
            elif isinstance(source, list) and source:
                return list(source)
    return _FALLBACK_DOCTORS


# ---------------------------------------------------------------------------
# BOOKING fallback (real: booking.py on SQLite - the section below only runs
# if booking.py is removed). Mirrors the real module's shapes EXACTLY: flat
# slot dicts, patient_name key, booking_id, alternatives on unavailable.
# ---------------------------------------------------------------------------

_SLOT_TIMES = [
    "09:00",
    "09:30",
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
    "16:00",
]
_demo_bookings: dict[tuple, dict] = {}  # (doctor_lower, date, time) -> booking
_demo_lock = asyncio.Lock()

_FALLBACK_DEPARTMENTS = sorted({d["department"].lower() for d in _FALLBACK_DOCTORS})


def _fb_normalize_date(day: Optional[str]) -> Optional[str]:
    if day is None:
        return None
    lowered = day.strip().lower()
    if lowered == "today":
        return _date.today().isoformat()
    if lowered == "tomorrow":
        return (_date.today() + timedelta(days=1)).isoformat()
    return day.strip()


def _fb_resolve_dept(name: str) -> tuple[Optional[str], Optional[str]]:
    key = name.strip().lower()
    if key in _FALLBACK_DEPARTMENTS:
        return key, None
    close = difflib.get_close_matches(key, _FALLBACK_DEPARTMENTS, n=1, cutoff=0.6)
    return (None, close[0]) if close else (None, None)


def _fb_doctors_in(dept_key: str) -> list[str]:
    return [d["name"] for d in _FALLBACK_DOCTORS if d["department"].lower() == dept_key]


def _fb_open_slots(
    dept_key: str, day: Optional[str], doctor: Optional[str]
) -> list[dict]:
    days = (
        [day]
        if day
        else [
            _date.today().isoformat(),
            (_date.today() + timedelta(days=1)).isoformat(),
        ]
    )
    doctors = _fb_doctors_in(dept_key)
    if doctor:
        doctors = [n for n in doctors if doctor.lower() in n.lower()] or doctors
    out = []
    for d in days:
        for name in doctors:
            for t in _SLOT_TIMES:
                if (name.lower(), d, t) not in _demo_bookings:
                    out.append({"doctor": name, "date": d, "time": t})
    return out


async def check_availability(
    department: str, date: Optional[str] = None, doctor: Optional[str] = None
) -> dict:
    if _booking is not None and hasattr(_booking, "check_availability"):
        return await _call(
            _booking.check_availability, department=department, date=date, doctor=doctor
        )

    dept_key, suggestion = _fb_resolve_dept(department)
    if dept_key is None:
        if suggestion:
            return {
                "status": "clarify",
                "message": f"No department called '{department}'. Did you mean {suggestion}?",
            }
        return {
            "status": "not_found",
            "message": f"No department called '{department}'. Available departments: "
            + ", ".join(_FALLBACK_DEPARTMENTS),
        }

    day = _fb_normalize_date(date)
    async with _demo_lock:
        slots = _fb_open_slots(dept_key, day, doctor)[:20]
    if not slots and day:
        async with _demo_lock:
            alternatives = _fb_open_slots(dept_key, None, None)[:5]
        return {
            "status": "ok",
            "department": dept_key,
            "date": day,
            "slots": [],
            "alternatives": alternatives,
            "message": f"No open slots found for {dept_key} on {day}.",
        }
    return {
        "status": "ok",
        "department": dept_key,
        "date": day,
        "slots": slots,
        "note": "Pass 'date' and 'time' back to book_appointment exactly as "
        "shown for whichever slot the caller picks.",
    }


async def book_appointment(
    patient_name: str,
    department: str,
    date: str,
    time: str,
    doctor: Optional[str] = None,
) -> dict:
    if _booking is not None and hasattr(_booking, "book_appointment"):
        return await _call(
            _booking.book_appointment,
            patient_name=patient_name,
            department=department,
            date=date,
            time=time,
            doctor=doctor,
        )

    dept_key, suggestion = _fb_resolve_dept(department)
    if dept_key is None:
        msg = (
            f"No department called '{department}'. Did you mean {suggestion}?"
            if suggestion
            else f"'{department}' isn't a department we have."
        )
        return {"status": "error", "message": msg}
    day = _fb_normalize_date(date)
    if time not in _SLOT_TIMES:
        return {
            "status": "unavailable",
            "message": f"Sorry, {time} on {day} isn't a valid slot.",
            "alternatives": _fb_open_slots(dept_key, day, doctor)[:5],
        }

    candidates = _fb_doctors_in(dept_key)
    if doctor:
        candidates = [
            n for n in candidates if doctor.lower() in n.lower()
        ] or candidates

    async with _demo_lock:  # atomic check-and-set = the demo's UNIQUE constraint
        for name in candidates:
            key = (name.lower(), day, time)
            if key not in _demo_bookings:
                booking_id = _uuid.uuid4().hex[:8]
                bk = {
                    "booking_id": booking_id,
                    "patient_name": patient_name,
                    "doctor": name,
                    "department": dept_key,
                    "date": day,
                    "time": time,
                }
                _demo_bookings[key] = bk
                return {
                    "status": "booked",
                    **bk,
                    "message": f"Confirmed: {patient_name} with {name} "
                    f"({dept_key}) on {day} at {time}. "
                    f"Confirmation code {booking_id}.",
                }
        alternatives = _fb_open_slots(dept_key, day, None)[:5]
    return {
        "status": "unavailable",
        "message": f"Sorry, no doctor in {dept_key} is free at {time} on {day}.",
        "alternatives": alternatives,
    }


def _fb_find_bookings(
    patient_name, department=None, date=None, time=None
) -> list[dict]:
    day = _fb_normalize_date(date)
    out = []
    for bk in _demo_bookings.values():
        if bk["patient_name"].lower() != patient_name.strip().lower():
            continue
        if department and bk["department"] != department.strip().lower():
            continue
        if day and bk["date"] != day:
            continue
        if time and bk["time"] != time:
            continue
        out.append(bk)
    return sorted(out, key=lambda b: (b["date"], b["time"]))


async def cancel_appointment(
    patient_name: str,
    date: Optional[str] = None,
    department: Optional[str] = None,
    time: Optional[str] = None,
) -> dict:
    if _booking is not None and hasattr(_booking, "cancel_appointment"):
        return await _call(
            _booking.cancel_appointment,
            patient_name=patient_name,
            date=date,
            department=department,
            time=time,
        )

    async with _demo_lock:
        matches = _fb_find_bookings(patient_name, department, date, time)
        if not matches:
            return {
                "status": "not_found",
                "message": f"I don't see any appointment booked under the name "
                f"'{patient_name}'. Can you confirm the name and "
                f"which department it was booked under?",
            }
        if len(matches) > 1:
            return {
                "status": "ambiguous",
                "message": f"I found more than one appointment under "
                f"'{patient_name}'. Which one do you mean - can you "
                f"tell me the department and date?",
                "matches": [
                    {k: b[k] for k in ("department", "doctor", "date", "time")}
                    for b in matches
                ],
            }
        bk = matches[0]
        del _demo_bookings[(bk["doctor"].lower(), bk["date"], bk["time"])]
        return {
            "status": "cancelled",
            **bk,
            "message": f"Cancelled: {bk['patient_name']}'s appointment with "
            f"{bk['doctor']} ({bk['department']}) on {bk['date']} "
            f"at {bk['time']}.",
        }


async def update_appointment(
    patient_name: str,
    new_date: Optional[str] = None,
    new_time: Optional[str] = None,
    date: Optional[str] = None,
    department: Optional[str] = None,
    time: Optional[str] = None,
) -> dict:
    if _booking is not None and hasattr(_booking, "update_appointment"):
        return await _call(
            _booking.update_appointment,
            patient_name=patient_name,
            new_date=new_date,
            new_time=new_time,
            date=date,
            department=department,
            time=time,
        )

    cancelled = await cancel_appointment(
        patient_name, date=date, department=department, time=time
    )
    if cancelled.get("status") != "cancelled":
        return cancelled
    booked = await book_appointment(
        patient_name=patient_name,
        department=cancelled["department"],
        date=_fb_normalize_date(new_date) or cancelled["date"],
        time=new_time or cancelled["time"],
        doctor=cancelled["doctor"],
    )
    if booked.get("status") == "booked":
        booked["status"] = "updated"
    return booked


# ---------------------------------------------------------------------------
# RAG / HOSPITAL INFO (real: rag.py -> hospital_kb.py, verified present)
# ---------------------------------------------------------------------------

_FALLBACK_KB: dict[str, str] = {
    "hours open opening timings closed": "Outpatient clinics are open Monday through "
    "Saturday, 8 AM to 6 PM. The emergency department is open around the clock.",
    "departments doctors specialist": "We have Cardiology with Dr. Imran Malik and "
    "Dr. Ayesha Siddiqui, General Medicine with Dr. Bilal Ahmed, and "
    "Pediatrics with Dr. Sana Farooqi.",
    "insurance billing payment": "We accept most major insurance plans. Billing is "
    "handled at the office on the 1st floor, with payment plans on request.",
    "prescription refill medication": "Refills need a doctor's approval and typically "
    "take one to two business days. Call the pharmacy or your doctor's office.",
    "visiting policy visitors hours ward": "Visiting hours for admitted patients are "
    "10 AM to 8 PM daily, two visitors per patient.",
    "lab results test report": "Lab results are typically available within three to "
    "five business days via the patient portal or the ordering department.",
    "parking location address directions": "A visitor parking garage is attached to "
    "the main building; the first hour is free. Valet is available on weekdays.",
}


async def search_hospital_info(query: str) -> dict:
    if _rag is not None and hasattr(_rag, "search_hospital_info"):
        return await _call(_rag.search_hospital_info, query=query)

    words = set(re.findall(r"[a-z]+", query.lower()))
    best_key, best_score = None, 0
    for key in _FALLBACK_KB:
        key_words = key.split()
        # Prefix-tolerant: "park" hits "parking", "visit" hits "visiting".
        # Minimum 4 chars so "the"/"can" never match anything.
        score = sum(
            1
            for w in words
            if len(w) >= 4
            and any(kw.startswith(w) or w.startswith(kw) for kw in key_words)
        )
        if score > best_score:
            best_key, best_score = key, score
    if best_key is None:
        return {
            "status": "not_found",
            "message": "No information found for that. Tell the caller you don't "
            "have that information rather than guessing.",
        }
    return {"status": "ok", "answer": _FALLBACK_KB[best_key]}


def warm_rag() -> None:
    """Pre-warm the embedding model BEFORE any call is accepted. This exact
    warm-up prevented a real live-call failure in the Pipecat version: the
    embedder lazy-loaded mid-call, blew the function-call timeout, and the
    caller's question was silently discarded."""
    if _rag is None:
        return
    for attr in ("_try_load_embedder", "warm_up", "load_embedder"):
        fn = getattr(_rag, attr, None)
        if callable(fn):
            logger.info(f"compat: pre-warming RAG embedder via rag.{attr}() ...")
            ok = fn()
            logger.info(
                "compat: RAG backend ready "
                + (
                    "(semantic)."
                    if ok
                    else "(keyword fallback - install "
                    "sentence-transformers for semantic search)."
                )
            )
            return
    logger.warning("compat: rag.py present but no warm-up hook found")
