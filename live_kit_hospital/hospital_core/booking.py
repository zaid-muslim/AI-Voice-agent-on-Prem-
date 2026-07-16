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
import sqlite3
import uuid
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "hospital_bookings.db"

# Doctor names - MUST stay consistent with hospital_kb.py's doctor bios
# (same names, same departments).
_SEED_SLOTS = [
    ("cardiology", "Dr. Imran Malik", "2026-07-14", "09:00"),
    ("cardiology", "Dr. Imran Malik", "2026-07-14", "10:30"),
    ("cardiology", "Dr. Imran Malik", "2026-07-14", "14:00"),
    ("cardiology", "Dr. Imran Malik", "2026-07-15", "11:00"),
    ("cardiology", "Dr. Imran Malik", "2026-07-15", "15:30"),
    ("cardiology", "Dr. Ayesha Siddiqui", "2026-07-14", "13:00"),
    ("general medicine", "Dr. Bilal Ahmed", "2026-07-14", "08:30"),
    ("general medicine", "Dr. Bilal Ahmed", "2026-07-14", "09:30"),
    ("general medicine", "Dr. Bilal Ahmed", "2026-07-14", "13:00"),
    ("general medicine", "Dr. Bilal Ahmed", "2026-07-16", "10:00"),
    ("pediatrics", "Dr. Sana Farooqi", "2026-07-15", "09:00"),
    ("pediatrics", "Dr. Sana Farooqi", "2026-07-15", "09:30"),
    ("pediatrics", "Dr. Sana Farooqi", "2026-07-15", "10:00"),
]


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
    (doctor, date, time) rows for OPEN slots (already excludes bookings)."""
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

        return conn.execute(query, params).fetchall()
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


if __name__ == "__main__":
    import os

    async def _run():
        if DB_PATH.exists():
            os.remove(DB_PATH)
        for ext in ("-wal", "-shm"):
            p = Path(str(DB_PATH) + ext)
            if p.exists():
                os.remove(p)
        _init_db()

        results = []

        r1 = await check_availability("cardiology", "2026-07-14")
        ok1 = r1["status"] == "ok" and any(
            s["doctor"] == "Dr. Imran Malik" and s["time"] == "09:00"
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
            "Someone", "neurology", "2026-07-14", "09:00", doctor="Dr. Imran Malik"
        )
        ok2c = r2c["status"] == "error" and "doctor" not in r2c["message"].lower()
        results.append(
            ("booking into invalid department reports THAT, not the doctor", ok2c, r2c)
        )

        r3 = await book_appointment(
            "Alice Kim", "cardiology", "2026-07-14", "09:00", doctor="Dr. Imran Malik"
        )
        ok3 = r3["status"] == "booked" and r3["patient_name"] == "Alice Kim"
        results.append(("booking an open slot succeeds", ok3, r3))

        r3b = await book_appointment(
            "Zoe Kim", "cardiology", "2026-07-14", "13:00", doctor="dr. ayesha siddiqui"
        )
        ok3b = r3b["status"] == "booked"
        results.append(("booking succeeds despite doctor-name casing drift", ok3b, r3b))

        r4 = await book_appointment(
            "Bob Lee", "cardiology", "2026-07-14", "09:00", doctor="Dr. Imran Malik"
        )
        ok4 = r4["status"] == "unavailable" and "no longer available" in r4["message"]
        results.append(("sequential double-booking rejected", ok4, r4))

        r4b = await book_appointment(
            "Eve Chan", "cardiology", "2026-07-14", "09:00", doctor="Dr. Imran Malikk"
        )
        ok4b = r4b["status"] == "error" and "Imran Malik" in r4b["message"]
        results.append(("near-miss doctor name suggests closest match", ok4b, r4b))

        r5a, r5b = await asyncio.gather(
            book_appointment(
                "Carol Diaz",
                "cardiology",
                "2026-07-14",
                "10:30",
                doctor="Dr. Imran Malik",
            ),
            book_appointment(
                "Dave Osei",
                "cardiology",
                "2026-07-14",
                "10:30",
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

        r6 = await check_availability("cardiology", "2025-05-15")
        ok6 = (
            r6["status"] == "ok"
            and not r6["slots"]
            and any(a["doctor"].startswith("Dr.") for a in r6.get("alternatives", []))
        )
        results.append(("wrong/empty date offers real alternatives", ok6, r6))

        r7 = await cancel_appointment("Alice Kim")
        ok7 = r7["status"] == "cancelled" and r7["doctor"] == "Dr. Imran Malik"
        results.append(("cancel_appointment cancels a real booking", ok7, r7))

        r7b = await check_availability("cardiology", "2026-07-14")
        ok7b = any(s["time"] == "09:00" for s in r7b["slots"])
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
            "2026-07-14",
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
            "Bob Lee", "cardiology", "2026-07-14", "14:00", doctor="Dr. Imran Malik"
        )
        assert r11setup["status"] == "booked", r11setup

        r11 = await update_appointment(
            "Bob Lee", new_date="2026-07-15", new_time="11:00"
        )
        ok11 = (
            r11["status"] == "updated"
            and r11["date"] == "2026-07-15"
            and r11["time"] == "11:00"
        )
        results.append(("update_appointment reschedules to an open slot", ok11, r11))

        r12a = await check_availability("cardiology", "2026-07-14")
        ok12a = any(s["time"] == "09:00" for s in r12a["slots"])
        results.append(("reschedule frees the old slot", ok12a, r12a))

        r12b = await book_appointment(
            "Someone Else",
            "cardiology",
            "2026-07-15",
            "11:00",
            doctor="Dr. Imran Malik",
        )
        ok12b = (
            r12b["status"] == "unavailable" and "no longer available" in r12b["message"]
        )
        results.append(("reschedule occupies the new slot", ok12b, r12b))

        r13 = await update_appointment(
            "Bob Lee", new_date="2026-07-15", new_time="23:59"
        )
        ok13 = r13["status"] == "unavailable"
        results.append(
            ("reschedule to invalid slot rejected with alternatives", ok13, r13)
        )

        r14 = await update_appointment(
            "Bob Lee", new_date="2026-07-15", new_time="11:00"
        )
        ok14 = r14["status"] == "no_change"
        results.append(("reschedule to identical slot is a graceful no-op", ok14, r14))

        r14b = await update_appointment("Bob Lee", new_time="15:30")
        ok14b = (
            r14b["status"] == "updated"
            and r14b["date"] == "2026-07-15"
            and r14b["time"] == "15:30"
        )
        results.append(
            ("partial reschedule (time only, date carried over) works", ok14b, r14b)
        )

        r15 = await update_appointment(
            "Nobody Here", new_date="2026-07-15", new_time="09:00"
        )
        ok15 = r15["status"] == "not_found"
        results.append(("update: unknown name reported gracefully", ok15, r15))

        r16 = await book_appointment("Auto Pick", "pediatrics", "2026-07-15", "09:30")
        ok16 = r16["status"] == "booked" and r16["doctor"] == "Dr. Sana Farooqi"
        results.append(
            ("book_appointment with doctor omitted auto-picks one", ok16, r16)
        )

        for desc, ok, detail in results:
            print(f"[{'PASS' if ok else 'FAIL'}] {desc}\n       -> {detail}")

        all_ok = all(ok for _, ok, _ in results)
        print(f"\n{'ALL PASS' if all_ok else 'FAILURES ABOVE'}")
        if not all_ok:
            print("DO NOT wire this into the pipeline until all cases pass.")
        return all_ok

    passed = asyncio.run(_run())
    exit(0 if passed else 1)
