"""
Reservation tools: check_availability + book_appointment + cancel_appointment
+ update_appointment, on SQLite instead of in-memory dicts.

WHY SQLITE, NOT REDIS (and not "faster" - see below):
A plain Python dict lookup is already about as fast as anything gets - no
database beats that, and this file's queries won't be noticeably faster than
the old in-memory version. What SQLite actually buys is CORRECTNESS UNDER
REAL CONCURRENCY, which the old dict-based version did not have: it only
avoided double-booking because everything ran in one single-threaded
process. Two instances of this agent (two simultaneous calls) would each
have had their own separate copy of the schedule and could have both booked
the same slot with neither aware of the other. Here, the UNIQUE constraint
on bookings(department, doctor, date, time) makes that impossible at the
database level, regardless of how many processes or threads are hitting it.
Redis would need manual WATCH/MULTI or a Lua script to get the same
guarantee; a one-line SQL constraint gets it for free.

CONSIDERING POSTGRES INSTEAD: reasonable next step for a more
production-like deployment - the schema and query shapes below translate
directly (swap sqlite3 for asyncpg/psycopg, PRAGMA busy_timeout/journal_mode
for Postgres's native MVCC, which handles concurrent readers/writers
without those pragmas at all). Function signatures would stay identical.

SCHEMA:
  slots(department, doctor, date, time)   - PRIMARY KEY, the schedule
      template: every valid, bookable slot, whether taken or not.
  bookings(booking_id, department, doctor, date, time, patient_name,
      created_at) - UNIQUE(department, doctor, date, time) is what makes
      double-booking impossible: the second INSERT for an already-booked
      slot raises sqlite3.IntegrityError, full stop, no race window.

DOCTOR NAMES: kept consistent with hospital_kb.py (same names, same
departments - if you rename here, rename there too).

FUZZY MATCHING: doctor comparisons are case-insensitive at the SQL level
(lower(doctor) = lower(?)) so casing drift from STT/LLM ("Dr Malik" vs
"Dr. Malik") doesn't cause a false "invalid slot". When a department or
doctor name doesn't match at all, both check_availability and
book_appointment try a fuzzy match via difflib.get_close_matches and, if
one clears the cutoff, suggest it instead of a flat error.
  cutoff=0.72 was picked by measuring difflib ratios: real near-misses
  (casing drift, one-letter typos, "cardiologyy") score 0.94+; a genuinely
  different word in the same list ("neurology" vs "cardiology") scores
  0.632. Checked against hand-built examples, NOT real STT output - verify
  against actual transcriptions before trusting it live.

CANCEL / UPDATE: looked up by PATIENT NAME rather than confirmation code,
since callers won't reliably have the code on hand. _resolve_matches() is
the shared disambiguation step: zero matches -> not-found; more than one ->
lists them and asks the caller to narrow down; exactly one -> proceeds.
  NAME MATCHING CAVEAT: exact, case-insensitive - no fuzzy/typo tolerance
  for patient names yet (unlike department/doctor).
  update_appointment only supports RESCHEDULING (new date/time, same
  doctor/department) - changing doctor or department is cancel + fresh
  book_appointment instead.

SEED DATES (this revision - fixed a real staleness bug): the previous
version hardcoded every seed slot to fixed July 2026 calendar dates
("2026-07-21", etc.). That was fine on the day it was written, but those
dates silently age into the past the moment the calendar moves on - a
caller asking about availability "today" or "next week" would eventually
get nothing, or an LLM could try to book a slot months in the past. Seed
dates are now generated relative to date.today() at the moment the DB is
first created (see _generate_seed_slots() below), so a fresh DB always
starts with a near-future schedule regardless of when it's actually spun
up. NOTE: this only affects a BRAND NEW database - _init_db() only seeds
when the slots table is empty, so an already-seeded hospital_bookings.db
keeps whatever dates it originally got. Delete the db file (and its
-wal/-shm siblings) to force a fresh, correctly-dated reseed. This also
does NOT make the schedule keep rolling forward on its own over time -
that would be a separate "prune old slots / append new ones on a cadence"
feature, not something a one-time seed can do.

PAST-TIME FILTER (this revision - fixed a real staleness bug, same family
as the seed-date fix above): _find_slots_sync's query only ever excluded
slots that were already BOOKED (via the LEFT JOIN ... IS NULL check). It
never checked whether a slot's TIME had already passed TODAY - so a 9:00
AM slot for today's date stayed "available" all day long, right up until
midnight, since nothing ever compared it against the current clock time.
Confirmed live: asking for availability at 5:43 PM still returned that
morning's 9:00 AM and 10:30 AM slots as open. Fixed by filtering, in
Python after the query runs, any row where row.date == today AND
row.time < now (string comparison on "HH:MM" works correctly here since
both sides are zero-padded 24-hour). Only today's date needs this check -
a future date's 9:00 AM slot is correctly still open regardless of what
time it is right now.
  IMPORT NAMING NOTE: this function's own parameter is named `date`,
  which shadows the `date` class imported from the datetime module at
  the top of this file. `datetime.date` is imported under the alias
  `date_cls` specifically so this function can call `date_cls.today()`
  without colliding with its own `date` parameter - calling plain
  `date.today()` inside this function would instead try (and fail) to
  call `.today()` on whatever string was passed in as the slot-lookup
  date argument.

LIVEKIT COMPAT NOTE (this revision - converted from Pipecat):
The original file took a `params` object as the first argument to every
public function and spoke results through `params.result_callback(string)`
instead of returning anything; it was also wrapped in
`@with_adaptive_filler` from a `tool_filler.py` that has no equivalent in a
LiveKit-style agent (LiveKit tools are plain async functions the LLM calls
and gets a return value back from - no params object, no mid-call spoken
filler hook). Both are removed here. Specific changes:
  - `params` argument dropped from every function.
  - Every function now RETURNS a dict (status + message + structured
    fields) instead of calling a callback with a spoken sentence. The LLM
    composes what it actually says from these fields.
  - `name` -> `patient_name` throughout, to match compat.py's canonical
    kwarg name (no _KWARG_ALIASES entry needed anymore).
  - `book_appointment`'s `doctor` is now OPTIONAL: if omitted, the first
    open slot matching department/date/time (any doctor) is used.
  - `update_appointment`'s `new_date`/`new_time` are now OPTIONAL: if
    either is omitted, the existing value is kept (so you can change just
    the time, or just the date).
  - The SQLite schema, `_book_sync`, `_reschedule_sync`, the UNIQUE
    constraint, and the fuzzy-matching logic are UNCHANGED - that's the
    tested, valuable part and none of it depended on Pipecat.
  - The dept_not_found-before-doctor_not_found check ordering fix from the
    previous revision is preserved as-is.
"""

import asyncio
import os
import sqlite3
import uuid
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "hospital_bookings.db"


# Doctor names - MUST stay consistent with hospital_kb.py's doctor bios
# (same names, same departments).
def _generate_seed_slots():
    """Builds the same schedule SHAPE as the old hardcoded list (same
    doctors, same relative day-spread, same times) but anchored to
    date.today() so a freshly-created DB always starts with near-future
    slots instead of ones hardcoded to a specific past/future month.

    Only affects a BRAND NEW db - _init_db() only calls this path when the
    slots table is empty. An existing hospital_bookings.db is untouched;
    delete it (plus its -wal/-shm files) to force a fresh, re-dated seed.
    """
    today = date_cls.today()
    d0 = str(today)
    d1 = str(today + timedelta(days=1))
    d2 = str(today + timedelta(days=2))
    d3 = str(today + timedelta(days=3))

    return [
        # Cardiology - Dr. Imran Malik (Today + Upcoming)
        ("cardiology", "Dr. Imran Malik", d0, "09:00"),
        ("cardiology", "Dr. Imran Malik", d0, "10:30"),
        ("cardiology", "Dr. Imran Malik", d0, "14:00"),
        ("cardiology", "Dr. Imran Malik", d1, "11:00"),
        ("cardiology", "Dr. Imran Malik", d1, "15:30"),
        ("cardiology", "Dr. Imran Malik", d2, "09:30"),
        ("cardiology", "Dr. Imran Malik", d2, "11:30"),
        ("cardiology", "Dr. Ayesha Siddiqui", d0, "13:00"),
        ("cardiology", "Dr. Ayesha Siddiqui", d1, "10:00"),
        ("cardiology", "Dr. Ayesha Siddiqui", d2, "14:30"),
        ("general medicine", "Dr. Bilal Ahmed", d0, "08:30"),
        ("general medicine", "Dr. Bilal Ahmed", d0, "09:30"),
        ("general medicine", "Dr. Bilal Ahmed", d0, "13:00"),
        ("general medicine", "Dr. Bilal Ahmed", d2, "10:00"),
        ("general medicine", "Dr. Bilal Ahmed", d3, "09:00"),
        ("general medicine", "Dr. Bilal Ahmed", d3, "11:00"),
        ("pediatrics", "Dr. Sana Farooqi", d1, "09:00"),
        ("pediatrics", "Dr. Sana Farooqi", d1, "09:30"),
        ("pediatrics", "Dr. Sana Farooqi", d1, "10:00"),
        ("pediatrics", "Dr. Sana Farooqi", d2, "11:00"),
        ("pediatrics", "Dr. Sana Farooqi", d2, "12:30"),
    ]


_SEED_SLOTS = _generate_seed_slots()


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 2000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _init_db():
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS slots (
                department TEXT NOT NULL,
                doctor TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                PRIMARY KEY (department, doctor, date, time)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                booking_id TEXT PRIMARY KEY,
                department TEXT NOT NULL,
                doctor TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (department, doctor, date, time)
            )
        """)
        (count,) = conn.execute("SELECT COUNT(*) FROM slots").fetchone()
        if count == 0:
            conn.executemany(
                "INSERT INTO slots (department, doctor, date, time) VALUES (?,?,?,?)",
                _SEED_SLOTS,
            )
        conn.commit()
    finally:
        conn.close()


_init_db()


def _closest_match(name, candidates, cutoff: float = 0.72):
    if not name or not candidates:
        return None
    lowered = {c.lower(): c for c in candidates}
    hits = get_close_matches(name.strip().lower(), lowered.keys(), n=1, cutoff=cutoff)
    return lowered[hits[0]] if hits else None


def _find_slots_sync(department, date, limit):
    """Returns None if the department doesn't exist at all, else a list of
    (doctor, date, time) rows for OPEN slots (already excludes bookings
    AND, for today's date specifically, already-passed times - see the
    PAST-TIME FILTER note in this file's module docstring)."""
    conn = _get_conn()
    try:
        dept_row = conn.execute(
            "SELECT 1 FROM slots WHERE department = ? LIMIT 1",
            (department.strip().lower(),),
        ).fetchone()
        if dept_row is None:
            return None

        query = """
            SELECT s.doctor, s.date, s.time
            FROM slots s
            LEFT JOIN bookings b
                ON s.department = b.department AND lower(s.doctor) = lower(b.doctor)
               AND s.date = b.date AND s.time = b.time
            WHERE s.department = ? AND b.booking_id IS NULL
        """
        params = [department.strip().lower()]
        if date:
            query += " AND s.date = ?"
            params.append(date)
        query += " ORDER BY s.date, s.time LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()

        # PAST-TIME FILTER: the JOIN above only excludes slots that are
        # already BOOKED - it says nothing about whether a slot's time has
        # already passed TODAY. Use date_cls (not the `date` parameter
        # this function shadows) to get the real current date/time.
        today_str = str(date_cls.today())
        now_str = datetime.now().strftime("%H:%M")
        rows = [r for r in rows if not (r[1] == today_str and r[2] < now_str)]

        return rows
    finally:
        conn.close()


def _all_departments_sync():
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT department FROM slots ORDER BY department"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def _doctors_in_department_sync(department):
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT doctor FROM slots WHERE department = ? ORDER BY doctor",
            (department.strip().lower(),),
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def _slots_as_dicts(rows) -> list:
    return [
        {"doctor": doctor, "date": date, "time": time}
        for doctor, date, time in (rows or [])
    ]


async def check_availability(
    department: str, date: Optional[str] = None, doctor: Optional[str] = None
) -> dict:
    """Look up open appointment slots for a department, optionally filtered
    by date (YYYY-MM-DD) and/or doctor name."""
    rows = await asyncio.to_thread(_find_slots_sync, department, date, 20)

    if rows is None:
        available_depts = await asyncio.to_thread(_all_departments_sync)
        suggestion = _closest_match(department, available_depts)
        if suggestion:
            return {
                "status": "clarify",
                "message": f"No department called '{department}'. Did you mean {suggestion}?",
            }
        return {
            "status": "not_found",
            "message": f"No department called '{department}'. Available departments: "
            + ", ".join(available_depts),
        }

    if doctor and rows:
        doc_match = _closest_match(doctor, sorted({r[0] for r in rows}))
        if doc_match:
            rows = [r for r in rows if r[0] == doc_match]

    if not rows:
        if date:
            alternatives = await asyncio.to_thread(
                _find_slots_sync, department, None, 5
            )
            return {
                "status": "ok",
                "department": department,
                "date": date,
                "slots": [],
                "alternatives": _slots_as_dicts(alternatives),
                "message": f"No open slots found for {department} on {date}.",
            }
        return {
            "status": "ok",
            "department": department,
            "date": date,
            "slots": [],
            "message": f"No open slots found for {department}.",
        }

    return {
        "status": "ok",
        "department": department,
        "date": date,
        "slots": _slots_as_dicts(rows),
        "note": "Pass 'date' and 'time' back to book_appointment exactly as "
        "shown for whichever slot the caller picks.",
    }


def _book_sync(patient_name, department, doctor, date, time):
    """Returns ("booked", id) / ("taken", None) / ("invalid", None) /
    ("doctor_not_found", None) / ("dept_not_found", None).

    Department existence is checked BEFORE doctor existence - a bogus
    department falling through to the doctor-check used to produce a
    confusing "I don't see a doctor named X in neurology" instead of
    reporting the department itself as invalid."""
    dept_key = department.strip().lower()
    conn = _get_conn()
    try:
        dept_exists = conn.execute(
            "SELECT 1 FROM slots WHERE department=? LIMIT 1", (dept_key,)
        ).fetchone()
        if not dept_exists:
            return ("dept_not_found", None)

        doctor_exists = conn.execute(
            "SELECT 1 FROM slots WHERE department=? AND lower(doctor)=lower(?) LIMIT 1",
            (dept_key, doctor),
        ).fetchone()
        if not doctor_exists:
            return ("doctor_not_found", None)

        exists = conn.execute(
            "SELECT 1 FROM slots WHERE department=? AND lower(doctor)=lower(?) "
            "AND date=? AND time=?",
            (dept_key, doctor, date, time),
        ).fetchone()
        if not exists:
            return ("invalid", None)

        booking_id = uuid.uuid4().hex[:8]
        try:
            conn.execute(
                "INSERT INTO bookings "
                "(booking_id, department, doctor, date, time, patient_name, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    booking_id,
                    dept_key,
                    doctor,
                    date,
                    time,
                    patient_name,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return ("booked", booking_id)
        except sqlite3.IntegrityError:
            conn.rollback()
            return ("taken", None)
    finally:
        conn.close()


async def book_appointment(
    patient_name: str,
    department: str,
    date: str,
    time: str,
    doctor: Optional[str] = None,
) -> dict:
    """Book a specific appointment slot. Only call this AFTER reading the
    slot back to the caller and getting their confirmation. If doctor is
    omitted, the first open slot matching department/date/time (any
    doctor in that department) is used."""
    if doctor is None:
        rows = await asyncio.to_thread(_find_slots_sync, department, date, 20)
        if rows is None:
            available_depts = await asyncio.to_thread(_all_departments_sync)
            suggestion = _closest_match(department, available_depts)
            if suggestion:
                return {
                    "status": "error",
                    "message": f"No department called '{department}'. Did you mean {suggestion}?",
                }
            return {
                "status": "error",
                "message": f"'{department}' isn't a department we have. "
                f"Available departments: {', '.join(available_depts)}.",
            }
        exact = [r for r in rows if r[1] == date and r[2] == time]
        if not exact:
            alternatives = rows or await asyncio.to_thread(
                _find_slots_sync, department, None, 5
            )
            return {
                "status": "unavailable",
                "message": f"No doctor in {department} is free at {time} on {date}.",
                "alternatives": _slots_as_dicts(alternatives),
            }
        doctor = exact[0][0]

    status, booking_id = await asyncio.to_thread(
        _book_sync, patient_name, department, doctor, date, time
    )

    if status == "dept_not_found":
        available_depts = await asyncio.to_thread(_all_departments_sync)
        suggestion = _closest_match(department, available_depts)
        if suggestion:
            return {
                "status": "error",
                "message": f"No department called '{department}'. Did you mean {suggestion}?",
            }
        return {
            "status": "error",
            "message": f"'{department}' isn't a department we have. "
            f"Available departments: {', '.join(available_depts)}.",
        }

    if status == "doctor_not_found":
        doctors = await asyncio.to_thread(_doctors_in_department_sync, department)
        suggestion = _closest_match(doctor, doctors)
        if suggestion:
            return {
                "status": "error",
                "message": f"I don't see a doctor named '{doctor}' in {department}. "
                f"Did you mean {suggestion}?",
            }
        return {
            "status": "error",
            "message": f"I don't see a doctor named '{doctor}' in {department}. "
            f"Doctors there: {', '.join(doctors) if doctors else 'none listed'}.",
        }

    if status in ("invalid", "taken"):
        alternatives = await asyncio.to_thread(_find_slots_sync, department, date, 5)
        if not alternatives:
            alternatives = await asyncio.to_thread(
                _find_slots_sync, department, None, 5
            )
        reason = (
            "isn't a valid slot" if status == "invalid" else "is no longer available"
        )
        return {
            "status": "unavailable",
            "message": f"Sorry, {doctor} at {time} on {date} {reason}.",
            "alternatives": _slots_as_dicts(alternatives),
        }

    return {
        "status": "booked",
        "booking_id": booking_id,
        "patient_name": patient_name,
        "doctor": doctor,
        "department": department,
        "date": date,
        "time": time,
        "message": f"Confirmed: {patient_name} with {doctor} ({department}) on "
        f"{date} at {time}. Confirmation code {booking_id}.",
    }


def _find_bookings_by_name_sync(patient_name, department=None, date=None, time=None):
    conn = _get_conn()
    try:
        query = (
            "SELECT booking_id, department, doctor, date, time, patient_name "
            "FROM bookings WHERE lower(patient_name) = lower(?)"
        )
        query_params = [patient_name.strip()]
        if department:
            query += " AND department = ?"
            query_params.append(department.strip().lower())
        if date:
            query += " AND date = ?"
            query_params.append(date)
        if time:
            query += " AND time = ?"
            query_params.append(time)
        query += " ORDER BY date, time"
        return conn.execute(query, query_params).fetchall()
    finally:
        conn.close()


def _bookings_as_dicts(rows) -> list:
    return [
        {"department": department, "doctor": doctor, "date": date, "time": time}
        for _booking_id, department, doctor, date, time, _patient_name in rows
    ]


def _format_bookings(rows) -> str:
    return "; ".join(
        f"{department} with {doctor} on {date} at {time}"
        for _booking_id, department, doctor, date, time, _patient_name in rows
    )


async def _resolve_matches(patient_name, department, date, time):
    """Shared lookup for cancel/update. Returns (dict_or_None, single_row).
    If dict_or_None is not None, that's the final not_found/ambiguous
    result to return - caller should stop there. Otherwise single_row is
    the one unambiguous booking row to act on."""
    matches = await asyncio.to_thread(
        _find_bookings_by_name_sync, patient_name, department, date, time
    )
    if not matches:
        qualifier = f" for {department}" if department else ""
        return (
            {
                "status": "not_found",
                "message": f"I don't see any appointment booked under the name "
                f"'{patient_name}'{qualifier}. Can you confirm the name "
                f"and which department it was booked under?",
            },
            None,
        )
    if len(matches) > 1:
        return (
            {
                "status": "ambiguous",
                "message": f"I found more than one appointment under '{patient_name}': "
                f"{_format_bookings(matches)}. Which one do you mean - can "
                f"you tell me the department and date?",
                "matches": _bookings_as_dicts(matches),
            },
            None,
        )
    return (None, matches[0])


def _cancel_sync(booking_id):
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM bookings WHERE booking_id = ?", (booking_id,))
        conn.commit()
        return "cancelled" if cur.rowcount else "not_found"
    finally:
        conn.close()


async def cancel_appointment(
    patient_name: str,
    date: Optional[str] = None,
    department: Optional[str] = None,
    time: Optional[str] = None,
) -> dict:
    """Cancel an existing appointment, looked up by patient name (and
    optionally narrowed by department/date/time if the name alone is
    ambiguous)."""
    early, booking = await _resolve_matches(patient_name, department, date, time)
    if early is not None:
        return early
    booking_id, dept, doctor, bdate, btime, name = booking

    status = await asyncio.to_thread(_cancel_sync, booking_id)
    if status == "not_found":
        return {
            "status": "not_found",
            "message": "That appointment appears to have already been cancelled.",
        }

    return {
        "status": "cancelled",
        "patient_name": name,
        "doctor": doctor,
        "department": dept,
        "date": bdate,
        "time": btime,
        "message": f"Cancelled: {name}'s appointment with {doctor} ({dept}) "
        f"on {bdate} at {btime}.",
    }


def _reschedule_sync(booking_id, department, doctor, new_date, new_time):
    dept_key = department.strip().lower()
    conn = _get_conn()
    try:
        slot_exists = conn.execute(
            "SELECT 1 FROM slots WHERE department=? AND lower(doctor)=lower(?) AND date=? AND time=?",
            (dept_key, doctor, new_date, new_time),
        ).fetchone()
        if not slot_exists:
            return ("invalid", None)

        row = conn.execute(
            "SELECT patient_name, created_at FROM bookings WHERE booking_id=?",
            (booking_id,),
        ).fetchone()
        if row is None:
            return ("not_found", None)
        patient_name, created_at = row

        conn.execute("DELETE FROM bookings WHERE booking_id=?", (booking_id,))
        try:
            conn.execute(
                "INSERT INTO bookings "
                "(booking_id, department, doctor, date, time, patient_name, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    booking_id,
                    dept_key,
                    doctor,
                    new_date,
                    new_time,
                    patient_name,
                    created_at,
                ),
            )
            conn.commit()
            return ("rescheduled", booking_id)
        except sqlite3.IntegrityError:
            conn.rollback()
            return ("taken", None)
    finally:
        conn.close()


async def update_appointment(
    patient_name: str,
    new_date: Optional[str] = None,
    new_time: Optional[str] = None,
    date: Optional[str] = None,
    department: Optional[str] = None,
    time: Optional[str] = None,
) -> dict:
    """Reschedule an existing appointment - same doctor and department
    only. new_date / new_time are each optional; omit one to change only
    the other (e.g. keep the same day, just move the time)."""
    early, booking = await _resolve_matches(patient_name, department, date, time)
    if early is not None:
        return early
    booking_id, dept, doctor, old_date, old_time, name = booking

    target_date = new_date or old_date
    target_time = new_time or old_time

    if target_date == old_date and target_time == old_time:
        return {
            "status": "no_change",
            "message": f"That appointment is already set for {old_date} at {old_time}.",
        }

    status, _ = await asyncio.to_thread(
        _reschedule_sync, booking_id, dept, doctor, target_date, target_time
    )

    if status == "invalid":
        alternatives = await asyncio.to_thread(_find_slots_sync, dept, target_date, 5)
        if not alternatives:
            alternatives = await asyncio.to_thread(_find_slots_sync, dept, None, 5)
        return {
            "status": "unavailable",
            "message": f"{doctor} isn't available at {target_time} on {target_date}.",
            "alternatives": _slots_as_dicts(alternatives),
        }

    if status == "taken":
        alternatives = await asyncio.to_thread(_find_slots_sync, dept, target_date, 5)
        if not alternatives:
            alternatives = await asyncio.to_thread(_find_slots_sync, dept, None, 5)
        return {
            "status": "unavailable",
            "message": "Sorry, that slot was just taken by someone else.",
            "alternatives": _slots_as_dicts(alternatives),
        }

    if status == "not_found":
        return {
            "status": "not_found",
            "message": "That appointment appears to have already been cancelled, "
            "so I can't reschedule it. Would you like to book a new "
            "one instead?",
        }

    # status is "updated" (not "rescheduled") to match agent.py's UI-push
    # check: `result.get("status") in ("booked", "updated")`. The spoken
    # message can still say "Rescheduled" - that's just user-facing text.
    return {
        "status": "updated",
        "booking_id": booking_id,
        "patient_name": name,
        "doctor": doctor,
        "department": dept,
        "date": target_date,
        "time": target_time,
        "message": f"Rescheduled: {name}'s appointment with {doctor} ({dept}) "
        f"is now on {target_date} at {target_time}. Confirmation code "
        f"stays {booking_id}.",
    }


# ---------------------------------------------------------------------------
# ADMIN SCHEDULE MANAGEMENT (this revision - adds the missing "add new doctor
# timings" surface). Used by admin_server.py's /api/doctors* endpoints only -
# NOT exposed to the LLM as a function_tool. check_availability/
# book_appointment/cancel_appointment/update_appointment above remain the
# only caller-facing surface; these are staff-facing schedule edits.
# ---------------------------------------------------------------------------


def _list_doctors_sync():
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT department, doctor, COUNT(*) AS total, "
            "SUM(CASE WHEN date >= ? THEN 1 ELSE 0 END) AS upcoming "
            "FROM slots GROUP BY department, doctor ORDER BY department, doctor",
            (str(date_cls.today()),),
        ).fetchall()
        return [
            {
                "department": d,
                "doctor": doc,
                "total_slots": total,
                "upcoming_slots": upcoming or 0,
            }
            for d, doc, total, upcoming in rows
        ]
    finally:
        conn.close()


async def list_doctors() -> list:
    """All (department, doctor) pairs with a slots row, plus slot counts -
    the schedule side of the roster (hospital_kb.get_doctor_roster() is the
    bio side; admin_server merges the two for the admin UI)."""
    return await asyncio.to_thread(_list_doctors_sync)


def _list_doctor_slots_sync(department, doctor):
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT s.date, s.time, b.patient_name "
            "FROM slots s LEFT JOIN bookings b "
            "ON s.department = b.department AND lower(s.doctor) = lower(b.doctor) "
            "AND s.date = b.date AND s.time = b.time "
            "WHERE s.department = ? AND lower(s.doctor) = lower(?) "
            "ORDER BY s.date, s.time",
            (department.strip().lower(), doctor),
        ).fetchall()
        return [
            {"date": d, "time": t, "booked": patient is not None, "patient_name": patient}
            for d, t, patient in rows
        ]
    finally:
        conn.close()


async def list_doctor_slots(department: str, doctor: str) -> list:
    """Every slot (open or booked) for one doctor, for the admin schedule
    view - unlike check_availability(), this deliberately includes booked
    slots too so staff can see the full picture."""
    return await asyncio.to_thread(_list_doctor_slots_sync, department, doctor)


def _add_slots_sync(department, doctor, slots):
    """slots: iterable of (date, time) strings. INSERT OR IGNORE so
    re-adding a slot that already exists is a harmless no-op, not a crash -
    admins re-submitting a recurring pattern that overlaps existing weeks
    should not error."""
    dept_key = department.strip().lower()
    conn = _get_conn()
    try:
        added = 0
        for d, t in slots:
            cur = conn.execute(
                "INSERT OR IGNORE INTO slots (department, doctor, date, time) "
                "VALUES (?,?,?,?)",
                (dept_key, doctor, d, t),
            )
            added += cur.rowcount
        conn.commit()
        return added
    finally:
        conn.close()


async def add_doctor_slots(department: str, doctor: str, slots: list) -> dict:
    """Add explicit (date, time) appointment slots for a doctor - the
    "add new timings" primitive the admin UI calls directly for one-off
    slots, and that add_recurring_schedule() below builds on for patterns."""
    slots = [(d, t) for d, t in slots]
    added = await asyncio.to_thread(_add_slots_sync, department, doctor, slots)
    return {"status": "ok", "requested": len(slots), "added": added}


def _generate_recurring_slots(start_date, weeks, weekdays, start_time, end_time, slot_minutes):
    """weekdays: set of int, Mon=0 .. Sun=6. Returns a flat list of
    (date, time) strings covering `weeks` weeks starting from start_date's
    week, on each matching weekday, every slot_minutes between start_time
    and end_time (end_time exclusive, matching how the seed schedule reads
    - e.g. 09:00-13:00/30min yields 09:00..12:30, not 13:00)."""
    out = []
    start_h, start_m = (int(x) for x in start_time.split(":"))
    end_h, end_m = (int(x) for x in end_time.split(":"))
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m
    for week in range(weeks):
        for day_offset in range(7):
            day = start_date + timedelta(days=week * 7 + day_offset)
            if day.weekday() not in weekdays:
                continue
            minutes = start_minutes
            while minutes < end_minutes:
                out.append((str(day), f"{minutes // 60:02d}:{minutes % 60:02d}"))
                minutes += slot_minutes
    return out


async def add_recurring_schedule(
    department: str,
    doctor: str,
    weekdays: list,
    start_time: str,
    end_time: str,
    slot_minutes: int = 30,
    weeks: int = 4,
    start_date: Optional[str] = None,
) -> dict:
    """Generate + add a recurring weekly schedule for a doctor - e.g. every
    Mon/Wed/Fri 09:00-13:00 in 30-minute slots for the next 4 weeks. This is
    the main "add new doctor timings" entry point the admin UI's recurring-
    schedule form calls. weekdays are ints, Mon=0..Sun=6."""
    anchor = date_cls.fromisoformat(start_date) if start_date else date_cls.today()
    slots = _generate_recurring_slots(
        anchor, weeks, set(weekdays), start_time, end_time, slot_minutes
    )
    if not slots:
        return {"status": "error", "message": "No slots generated - check weekdays/time range."}
    added = await asyncio.to_thread(_add_slots_sync, department, doctor, slots)
    return {"status": "ok", "generated": len(slots), "added": added}


def _remove_slot_sync(department, doctor, date, time):
    conn = _get_conn()
    try:
        booked = conn.execute(
            "SELECT 1 FROM bookings WHERE department = ? AND lower(doctor) = lower(?) "
            "AND date = ? AND time = ?",
            (department.strip().lower(), doctor, date, time),
        ).fetchone()
        if booked:
            return "booked"
        cur = conn.execute(
            "DELETE FROM slots WHERE department = ? AND lower(doctor) = lower(?) "
            "AND date = ? AND time = ?",
            (department.strip().lower(), doctor, date, time),
        )
        conn.commit()
        return "removed" if cur.rowcount else "not_found"
    finally:
        conn.close()


async def remove_doctor_slot(department: str, doctor: str, date: str, time: str) -> dict:
    """Remove a single, not-yet-booked slot. Refuses if a patient already
    holds that slot (cancel the booking first, via cancel_appointment) -
    an admin should never be able to silently evict a real patient."""
    status = await asyncio.to_thread(_remove_slot_sync, department, doctor, date, time)
    if status == "booked":
        return {
            "status": "error",
            "message": "That slot is already booked by a patient - cancel the booking first.",
        }
    if status == "not_found":
        return {"status": "error", "message": "That slot doesn't exist."}
    return {"status": "ok", "message": "Slot removed."}


async def self_test() -> bool:
    """Rebuild hospital_bookings.db from scratch and run the real-
    scenario checks below (double-booking, fuzzy matching, reschedule,
    the past-time filter, etc). NEVER call this against a real
    deployment's database - see tests/test_booking.py for the pytest
    wrapper that isolates this in a temp DB."""
    if DB_PATH.exists():
        os.remove(DB_PATH)
    for ext in ("-wal", "-shm"):
        p = Path(str(DB_PATH) + ext)
        if p.exists():
            os.remove(p)
    _init_db()

    # Recompute the same relative dates the fresh seed just used, so the
    # self-test's assertions (which reference specific dates) line up
    # with whatever _generate_seed_slots() actually inserted this run.
    today = date_cls.today()
    d0 = str(today)
    d1 = str(today + timedelta(days=1))
    d2 = str(today + timedelta(days=2))
    d3 = str(today + timedelta(days=3))
    past_date = str(today - timedelta(days=365))

    results = []

    # d2 (not d0) is used for the Alice/Bob booking scenario below so this
    # self-test is never flaky depending on what time of day it happens to
    # run - a d0 slot earlier than the current wall-clock time is correctly
    # excluded by the PAST-TIME FILTER (see this module's docstring), which
    # would make an assertion pinned to "today, 09:00" fail after 9 AM for
    # reasons that have nothing to do with a real regression.
    r1 = await check_availability("cardiology", d2)
    ok1 = r1["status"] == "ok" and any(
        s["doctor"] == "Dr. Imran Malik" and s["time"] == "09:30"
        for s in r1["slots"]
    )
    results.append(("check_availability finds known slots", ok1, r1))

    r2 = await check_availability("neurology")
    ok2 = r2["status"] == "not_found" and "cardiology" in r2["message"]
    results.append(("unknown department handled gracefully", ok2, r2))

    r2b = await check_availability("cardiologyy")
    ok2b = r2b["status"] == "clarify" and "cardiology" in r2b["message"]
    results.append(("near-miss department suggests closest match", ok2b, r2b))

    r2c = await book_appointment(
        "Someone", "neurology", d0, "09:00", doctor="Dr. Imran Malik"
    )
    ok2c = r2c["status"] == "error" and "doctor" not in r2c["message"].lower()
    results.append(
        ("booking into invalid department reports THAT, not the doctor", ok2c, r2c)
    )

    r3 = await book_appointment(
        "Alice Kim", "cardiology", d2, "09:30", doctor="Dr. Imran Malik"
    )
    ok3 = r3["status"] == "booked" and r3["patient_name"] == "Alice Kim"
    results.append(("booking an open slot succeeds", ok3, r3))

    r3b = await book_appointment(
        "Zoe Kim", "cardiology", d0, "13:00", doctor="dr. ayesha siddiqui"
    )
    ok3b = r3b["status"] == "booked"
    results.append(("booking succeeds despite doctor-name casing drift", ok3b, r3b))

    r4 = await book_appointment(
        "Bob Lee", "cardiology", d2, "09:30", doctor="Dr. Imran Malik"
    )
    ok4 = r4["status"] == "unavailable" and "no longer available" in r4["message"]
    results.append(("sequential double-booking rejected", ok4, r4))

    r4b = await book_appointment(
        "Eve Chan", "cardiology", d0, "09:00", doctor="Dr. Imran Malikk"
    )
    ok4b = r4b["status"] == "error" and "Imran Malik" in r4b["message"]
    results.append(("near-miss doctor name suggests closest match", ok4b, r4b))

    r5a, r5b = await asyncio.gather(
        book_appointment(
            "Carol Diaz",
            "cardiology",
            d1,
            "11:00",
            doctor="Dr. Imran Malik",
        ),
        book_appointment(
            "Dave Osei",
            "cardiology",
            d1,
            "11:00",
            doctor="Dr. Imran Malik",
        ),
    )
    outcomes = {r5a["status"] == "booked", r5b["status"] == "booked"}
    ok5 = outcomes == {True, False}
    results.append(
        (
            "concurrent double-booking: exactly one wins",
            ok5,
            f"r5a={r5a} | r5b={r5b}",
        )
    )

    r6 = await check_availability("cardiology", past_date)
    ok6 = (
        r6["status"] == "ok"
        and not r6["slots"]
        and any(a["doctor"].startswith("Dr.") for a in r6.get("alternatives", []))
    )
    results.append(("wrong/empty date offers real alternatives", ok6, r6))

    r7 = await cancel_appointment("Alice Kim")
    ok7 = r7["status"] == "cancelled" and r7["doctor"] == "Dr. Imran Malik"
    results.append(("cancel_appointment cancels a real booking", ok7, r7))

    r7b = await check_availability("cardiology", d2)
    ok7b = any(s["time"] == "09:30" for s in r7b["slots"])
    results.append(("cancelled slot becomes available again", ok7b, r7b))

    r8 = await cancel_appointment("Nobody Here")
    ok8 = r8["status"] == "not_found"
    results.append(("cancel: unknown name reported gracefully", ok8, r8))

    r9 = await cancel_appointment("Alice Kim")
    ok9 = r9["status"] == "not_found"
    results.append(
        ("re-cancelling an already-cancelled booking is graceful", ok9, r9)
    )

    r10setup = await book_appointment(
        "Zoe Kim",
        "general medicine",
        d0,
        "08:30",
        doctor="Dr. Bilal Ahmed",
    )
    assert r10setup["status"] == "booked", r10setup

    r10 = await cancel_appointment("Zoe Kim")
    depts = {m["department"] for m in r10.get("matches", [])}
    ok10 = r10["status"] == "ambiguous" and depts == {
        "cardiology",
        "general medicine",
    }
    results.append(
        ("cancel: ambiguous name lists both and asks to narrow down", ok10, r10)
    )

    r10b = await cancel_appointment("Zoe Kim", department="general medicine")
    ok10b = (
        r10b["status"] == "cancelled" and r10b["department"] == "general medicine"
    )
    results.append(
        ("cancel: department filter disambiguates correctly", ok10b, r10b)
    )

    r11setup = await book_appointment(
        "Bob Lee", "cardiology", d0, "14:00", doctor="Dr. Imran Malik"
    )
    assert r11setup["status"] == "booked", r11setup

    r11 = await update_appointment("Bob Lee", new_date=d1, new_time="15:30")
    ok11 = (
        r11["status"] == "updated" and r11["date"] == d1 and r11["time"] == "15:30"
    )
    results.append(("update_appointment reschedules to an open slot", ok11, r11))

    # Bob Lee's OLD slot before the reschedule above was d0/"14:00" (booked
    # in r11setup) - NOT "09:00" (a pre-existing copy-paste bug in this
    # assertion: it happened to still pass whenever the self-test ran
    # before 9 AM, since d0's untouched "09:00" seed slot is only hidden by
    # the PAST-TIME FILTER after that time - it was never actually checking
    # what this test's name says it checks).
    r12a = await check_availability("cardiology", d0)
    ok12a = any(s["time"] == "14:00" for s in r12a["slots"])
    results.append(("reschedule frees the old slot", ok12a, r12a))

    r12b = await book_appointment(
        "Someone Else",
        "cardiology",
        d1,
        "15:30",
        doctor="Dr. Imran Malik",
    )
    ok12b = (
        r12b["status"] == "unavailable" and "no longer available" in r12b["message"]
    )
    results.append(("reschedule occupies the new slot", ok12b, r12b))

    r13 = await update_appointment("Bob Lee", new_date=d1, new_time="23:59")
    ok13 = r13["status"] == "unavailable"
    results.append(
        ("reschedule to invalid slot rejected with alternatives", ok13, r13)
    )

    r14 = await update_appointment("Bob Lee", new_date=d1, new_time="15:30")
    ok14 = r14["status"] == "no_change"
    results.append(("reschedule to identical slot is a graceful no-op", ok14, r14))

    r14b = await update_appointment("Bob Lee", new_time="15:30")
    ok14b = r14b["status"] == "no_change"
    results.append(
        ("reschedule to same time (date carried over) is a no-op", ok14b, r14b)
    )

    r15 = await update_appointment("Nobody Here", new_date=d1, new_time="09:00")
    ok15 = r15["status"] == "not_found"
    results.append(("update: unknown name reported gracefully", ok15, r15))

    r16 = await book_appointment("Auto Pick", "pediatrics", d1, "09:30")
    ok16 = r16["status"] == "booked" and r16["doctor"] == "Dr. Sana Farooqi"
    results.append(
        ("book_appointment with doctor omitted auto-picks one", ok16, r16)
    )

    # PAST-TIME FILTER regression test: today's slots that have already
    # passed the current clock time must NOT show up as available.
    # This directly encodes the 5:43 PM bug (morning slots wrongly
    # still offered) so it can never silently regress again.
    now_str = datetime.now().strftime("%H:%M")
    r17 = await check_availability("general medicine", d0)
    past_today_slots = [s for s in r17["slots"] if s["time"] < now_str]
    ok17 = len(past_today_slots) == 0
    results.append(
        (
            "today's already-passed times are excluded from availability",
            ok17,
            f"now={now_str} | slots_returned={r17['slots']}",
        )
    )

    for desc, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {desc}\n       -> {detail}")

    all_ok = all(ok for _, ok, _ in results)
    print(f"\n{'ALL PASS' if all_ok else 'FAILURES ABOVE'}")
    if not all_ok:
        print("DO NOT wire this into the pipeline until all cases pass.")
    return all_ok


if __name__ == "__main__":
    passed = asyncio.run(self_test())
    exit(0 if passed else 1)
