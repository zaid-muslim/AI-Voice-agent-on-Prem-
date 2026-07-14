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
production-like deployment (see the separate architecture discussion this
was built alongside) - the schema and query shapes below translate directly
(swap sqlite3 for asyncpg/psycopg, PRAGMA busy_timeout/journal_mode for
Postgres's native MVCC, which handles concurrent readers/writers without
those pragmas at all). Function signatures would stay identical, so
main_hospital.py would not need to change.

SCHEMA:
  slots(department, doctor, date, time)   - PRIMARY KEY, the schedule
      template: every valid, bookable slot, whether taken or not.
  bookings(booking_id, department, doctor, date, time, patient_name,
      created_at) - UNIQUE(department, doctor, date, time) is what makes
      double-booking impossible: the second INSERT for an already-booked
      slot raises sqlite3.IntegrityError, full stop, no race window.

FIXED IN THIS REVISION - dept_not_found bug:
book_appointment previously checked doctor-existence WITHOUT first checking
whether the department itself was even real. Booking into a bogus
department (e.g. "neurology", which doesn't exist at all) would report
"I don't see a doctor named X in neurology" - confusing, since the real
problem is the department, not the doctor. _book_sync now checks department
existence FIRST (mirroring check_availability's own fuzzy-suggest logic),
and only then checks the doctor within that now-confirmed-valid department.

DOCTOR NAMES: renamed to Riverside General's actual roster - kept
consistent with hospital_kb.py (same names, same departments - if you
rename here, rename there too, per that file's own consistency note).

FUZZY MATCHING:
  Doctor name comparisons are case-insensitive at the SQL level
  (lower(doctor) = lower(?)) instead of exact-string, which fixes a real
  bug: department was always normalized with .strip().lower() but doctor
  was not, so a caller/LLM echoing back slightly different casing
  ("Dr Malik" vs "Dr. Malik") could get a false "invalid slot" even though
  the slot existed.

  On top of that, when a department or doctor name doesn't match at all
  (e.g. mis-heard STT, mispronunciation, minor typo from the LLM), both
  check_availability and book_appointment now try a fuzzy match via
  difflib.get_close_matches and, if one clears the cutoff, return a
  "Did you mean X?" suggestion instead of a flat error.

  cutoff=0.72 was picked by actually measuring difflib ratios: real
  near-misses (casing drift, one-letter typos, "cardiologyy") score 0.94+;
  a genuinely different word in the same list ("neurology" vs "cardiology")
  scores 0.632. This was checked against hand-built examples, NOT real STT
  output - test it against actual Whisper transcriptions before trusting
  it live.

CANCEL / UPDATE:
  Looked up by PATIENT NAME rather than confirmation code, since callers
  won't reliably have the code on hand. _resolve_single_booking() is the
  shared disambiguation step: zero matches -> not-found; more than one ->
  lists them and asks the caller to narrow down; exactly one -> proceeds.

  NAME MATCHING CAVEAT: exact, case-insensitive - no fuzzy/typo tolerance
  for patient names yet (unlike department/doctor).

  update_appointment only supports RESCHEDULING (new date/time, same
  doctor/department) - changing doctor or department is cancel + fresh
  book_appointment instead.

Expects tool_filler.py in the same directory/package.
"""

import asyncio
import sqlite3
import uuid
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path

try:
    from .tool_filler import with_adaptive_filler
except ImportError:
    from tool_filler import with_adaptive_filler

DB_PATH = Path(__file__).parent / "hospital_bookings.db"

# Doctor names updated - MUST stay consistent with hospital_kb.py's doctor
# bios (same names, same departments).
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


def _format_slots(slots) -> str:
    if not slots:
        return "no open slots found"
    return "; ".join(f"{doctor} on {date} at {time}" for doctor, date, time in slots)


@with_adaptive_filler(threshold_secs=0.5, filler_text="Let me check what's open.")
async def check_availability(params, department: str, date=None):
    """Look up open appointment slots for a department, optionally on a
    specific date (YYYY-MM-DD)."""
    slots = await asyncio.to_thread(_find_slots_sync, department, date, 5)

    if slots is None:
        available_depts = await asyncio.to_thread(_all_departments_sync)
        suggestion = _closest_match(department, available_depts)
        if suggestion:
            await params.result_callback(
                f"I don't see a department called '{department}'. "
                f"Did you mean '{suggestion}'? Full list: {', '.join(available_depts)}."
            )
        else:
            await params.result_callback(
                f"'{department}' isn't a department we have. "
                f"Available departments: {', '.join(available_depts)}."
            )
        return
    if not slots:
        if date:
            alternatives = await asyncio.to_thread(
                _find_slots_sync, department, None, 5
            )
            if alternatives:
                await params.result_callback(
                    f"No open slots found for {department} on {date}. "
                    f"Other available times: {_format_slots(alternatives)}."
                )
            else:
                await params.result_callback(
                    f"No open slots found for {department} at all right now."
                )
        else:
            await params.result_callback(f"No open slots found for {department}.")
        return
    await params.result_callback(
        f"Open slots for {department}: {_format_slots(slots)}."
    )


def _book_sync(name, department, doctor, date, time):
    """Returns ("booked", id) / ("taken", None) / ("invalid", None) /
    ("doctor_not_found", None) / ("dept_not_found", None).

    FIXED: department existence is checked BEFORE doctor existence - a
    bogus department used to fall through to the doctor-check and produce
    a confusing "I don't see a doctor named X in neurology" message
    instead of reporting the department itself as invalid."""
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
                    name,
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


@with_adaptive_filler(threshold_secs=0.8, filler_text="Booking that now, one moment.")
async def book_appointment(
    params, name: str, department: str, doctor: str, date: str, time: str
):
    """Book a specific slot. Only call this AFTER reading the slot back to
    the caller and getting their confirmation."""
    status, booking_id = await asyncio.to_thread(
        _book_sync, name, department, doctor, date, time
    )

    if status == "dept_not_found":
        available_depts = await asyncio.to_thread(_all_departments_sync)
        suggestion = _closest_match(department, available_depts)
        if suggestion:
            await params.result_callback(
                f"I don't see a department called '{department}'. "
                f"Did you mean '{suggestion}'? Full list: {', '.join(available_depts)}."
            )
        else:
            await params.result_callback(
                f"'{department}' isn't a department we have. "
                f"Available departments: {', '.join(available_depts)}."
            )
        return

    if status == "doctor_not_found":
        doctors = await asyncio.to_thread(_doctors_in_department_sync, department)
        suggestion = _closest_match(doctor, doctors)
        if suggestion:
            await params.result_callback(
                f"I don't see a doctor named '{doctor}' in {department}. "
                f"Did you mean {suggestion}?"
            )
        else:
            await params.result_callback(
                f"I don't see a doctor named '{doctor}' in {department}. "
                f"Doctors there: {', '.join(doctors) if doctors else 'none listed'}."
            )
        return

    if status in ("invalid", "taken"):
        alternatives = await asyncio.to_thread(_find_slots_sync, department, date, 5)
        if not alternatives:
            alternatives = await asyncio.to_thread(
                _find_slots_sync, department, None, 5
            )
        reason = (
            "isn't a valid slot" if status == "invalid" else "is no longer available"
        )
        await params.result_callback(
            f"Sorry, {doctor} at {time} on {date} {reason}. "
            f"Other options: {_format_slots(alternatives)}."
        )
        return

    await params.result_callback(
        f"Confirmed: {name} with {doctor} ({department}) on {date} at {time}. "
        f"Confirmation code {booking_id}."
    )


def _find_bookings_by_name_sync(name, department=None, date=None, time=None):
    conn = _get_conn()
    try:
        query = (
            "SELECT booking_id, department, doctor, date, time, patient_name "
            "FROM bookings WHERE lower(patient_name) = lower(?)"
        )
        query_params = [name.strip()]
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


def _format_bookings(bookings) -> str:
    return "; ".join(
        f"{department} with {doctor} on {date} at {time}"
        for _booking_id, department, doctor, date, time, _patient_name in bookings
    )


async def _resolve_single_booking(params, name, department, date, time):
    matches = await asyncio.to_thread(
        _find_bookings_by_name_sync, name, department, date, time
    )
    if not matches:
        qualifier = f" for {department}" if department else ""
        await params.result_callback(
            f"I don't see any appointment booked under the name '{name}'"
            f"{qualifier}. Can you confirm the name and which department it was booked under?"
        )
        return None
    if len(matches) > 1:
        await params.result_callback(
            f"I found more than one appointment under '{name}': "
            f"{_format_bookings(matches)}. Which one do you mean - can you tell me the department and date?"
        )
        return None
    return matches[0]


def _cancel_sync(booking_id):
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM bookings WHERE booking_id = ?", (booking_id,))
        conn.commit()
        return "cancelled" if cur.rowcount else "not_found"
    finally:
        conn.close()


@with_adaptive_filler(threshold_secs=0.5, filler_text="One moment while I cancel that.")
async def cancel_appointment(params, name: str, department=None, date=None, time=None):
    """Cancel an existing appointment, looked up by patient name."""
    booking = await _resolve_single_booking(params, name, department, date, time)
    if booking is None:
        return
    booking_id, dept, doctor, bdate, btime, patient_name = booking

    status = await asyncio.to_thread(_cancel_sync, booking_id)
    if status == "not_found":
        await params.result_callback(
            "That appointment appears to have already been cancelled."
        )
        return

    await params.result_callback(
        f"Cancelled: {patient_name}'s appointment with {doctor} ({dept}) on {bdate} at {btime}."
    )


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


@with_adaptive_filler(threshold_secs=0.8, filler_text="Updating that appointment now.")
async def update_appointment(
    params,
    name: str,
    new_date: str,
    new_time: str,
    department=None,
    date=None,
    time=None,
):
    """Reschedule an existing appointment to a new date/time - same doctor
    and department only."""
    booking = await _resolve_single_booking(params, name, department, date, time)
    if booking is None:
        return
    booking_id, dept, doctor, old_date, old_time, patient_name = booking

    if old_date == new_date and old_time == new_time:
        await params.result_callback(
            f"That appointment is already set for {new_date} at {new_time}."
        )
        return

    status, _ = await asyncio.to_thread(
        _reschedule_sync, booking_id, dept, doctor, new_date, new_time
    )

    if status == "invalid":
        alternatives = await asyncio.to_thread(_find_slots_sync, dept, new_date, 5)
        if not alternatives:
            alternatives = await asyncio.to_thread(_find_slots_sync, dept, None, 5)
        await params.result_callback(
            f"{doctor} isn't available at {new_time} on {new_date}. "
            f"Other options: {_format_slots(alternatives)}."
        )
        return

    if status == "taken":
        alternatives = await asyncio.to_thread(_find_slots_sync, dept, new_date, 5)
        if not alternatives:
            alternatives = await asyncio.to_thread(_find_slots_sync, dept, None, 5)
        await params.result_callback(
            f"Sorry, that slot was just taken by someone else. Other options: {_format_slots(alternatives)}."
        )
        return

    if status == "not_found":
        await params.result_callback(
            "That appointment appears to have already been cancelled, so I can't reschedule it. "
            "Would you like to book a new one instead?"
        )
        return

    await params.result_callback(
        f"Rescheduled: {patient_name}'s appointment with {doctor} ({dept}) "
        f"is now on {new_date} at {new_time}. Confirmation code stays {booking_id}."
    )


if __name__ == "__main__":
    import os

    class FakeParams:
        def __init__(self):
            self.result = None

        async def result_callback(self, result):
            self.result = result

    async def _run():
        if DB_PATH.exists():
            os.remove(DB_PATH)
        for ext in ("-wal", "-shm"):
            p = Path(str(DB_PATH) + ext)
            if p.exists():
                os.remove(p)
        _init_db()

        results = []

        p1 = FakeParams()
        await check_availability(p1, "cardiology", "2026-07-14")
        ok1 = "Dr. Imran Malik" in p1.result and "09:00" in p1.result
        results.append(("check_availability finds known slots", ok1, p1.result))

        p2 = FakeParams()
        await check_availability(p2, "neurology")
        ok2 = "isn't a department" in p2.result and "cardiology" in p2.result
        results.append(("unknown department handled gracefully", ok2, p2.result))

        p2b = FakeParams()
        await check_availability(p2b, "cardiologyy")
        ok2b = "Did you mean" in p2b.result and "cardiology" in p2b.result
        results.append(
            ("near-miss department suggests closest match", ok2b, p2b.result)
        )

        p2c = FakeParams()
        await book_appointment(
            p2c, "Someone", "neurology", "Dr. Imran Malik", "2026-07-14", "09:00"
        )
        ok2c = "isn't a department" in p2c.result and "doctor" not in p2c.result.lower()
        results.append(
            (
                "booking into invalid department reports THAT, not the doctor",
                ok2c,
                p2c.result,
            )
        )

        p3 = FakeParams()
        await book_appointment(
            p3, "Alice Kim", "cardiology", "Dr. Imran Malik", "2026-07-14", "09:00"
        )
        ok3 = "Confirmed" in p3.result and "Alice Kim" in p3.result
        results.append(("booking an open slot succeeds", ok3, p3.result))

        p3b = FakeParams()
        await book_appointment(
            p3b, "Zoe Kim", "cardiology", "dr. ayesha siddiqui", "2026-07-14", "13:00"
        )
        ok3b = "Confirmed" in p3b.result
        results.append(
            ("booking succeeds despite doctor-name casing drift", ok3b, p3b.result)
        )

        p4 = FakeParams()
        await book_appointment(
            p4, "Bob Lee", "cardiology", "Dr. Imran Malik", "2026-07-14", "09:00"
        )
        ok4 = "no longer available" in p4.result
        results.append(("sequential double-booking rejected", ok4, p4.result))

        p4b = FakeParams()
        await book_appointment(
            p4b, "Eve Chan", "cardiology", "Dr. Imran Malikk", "2026-07-14", "09:00"
        )
        ok4b = "Did you mean" in p4b.result and "Imran Malik" in p4b.result
        results.append(
            ("near-miss doctor name suggests closest match", ok4b, p4b.result)
        )

        p5a, p5b = FakeParams(), FakeParams()
        await asyncio.gather(
            book_appointment(
                p5a,
                "Carol Diaz",
                "cardiology",
                "Dr. Imran Malik",
                "2026-07-14",
                "10:30",
            ),
            book_appointment(
                p5b, "Dave Osei", "cardiology", "Dr. Imran Malik", "2026-07-14", "10:30"
            ),
        )
        outcomes = {("Confirmed" in p5a.result), ("Confirmed" in p5b.result)}
        ok5 = outcomes == {True, False}
        results.append(
            (
                "concurrent double-booking: exactly one wins",
                ok5,
                f"p5a={p5a.result!r} | p5b={p5b.result!r}",
            )
        )

        p6 = FakeParams()
        await check_availability(p6, "cardiology", "2025-05-15")
        ok6 = (
            "No open slots found for cardiology on 2025-05-15" in p6.result
            and "Dr." in p6.result
        )
        results.append(("wrong/empty date offers real alternatives", ok6, p6.result))

        p7 = FakeParams()
        await cancel_appointment(p7, "Alice Kim")
        ok7 = "Cancelled" in p7.result and "Imran Malik" in p7.result
        results.append(("cancel_appointment cancels a real booking", ok7, p7.result))

        p7b = FakeParams()
        await check_availability(p7b, "cardiology", "2026-07-14")
        ok7b = "09:00" in p7b.result
        results.append(("cancelled slot becomes available again", ok7b, p7b.result))

        p8 = FakeParams()
        await cancel_appointment(p8, "Nobody Here")
        ok8 = "don't see any appointment" in p8.result
        results.append(("cancel: unknown name reported gracefully", ok8, p8.result))

        p9 = FakeParams()
        await cancel_appointment(p9, "Alice Kim")
        ok9 = "don't see any appointment" in p9.result
        results.append(
            ("re-cancelling an already-cancelled booking is graceful", ok9, p9.result)
        )

        p10setup = FakeParams()
        await book_appointment(
            p10setup,
            "Zoe Kim",
            "general medicine",
            "Dr. Bilal Ahmed",
            "2026-07-14",
            "08:30",
        )
        p10 = FakeParams()
        await cancel_appointment(p10, "Zoe Kim")
        ok10 = (
            "more than one appointment" in p10.result
            and "cardiology" in p10.result
            and "general medicine" in p10.result
        )
        results.append(
            (
                "cancel: ambiguous name lists both and asks to narrow down",
                ok10,
                p10.result,
            )
        )

        p10b = FakeParams()
        await cancel_appointment(p10b, "Zoe Kim", department="general medicine")
        ok10b = "Cancelled" in p10b.result and "general medicine" in p10b.result
        results.append(
            ("cancel: department filter disambiguates correctly", ok10b, p10b.result)
        )

        p11setup = FakeParams()
        await book_appointment(
            p11setup, "Bob Lee", "cardiology", "Dr. Imran Malik", "2026-07-14", "14:00"
        )
        assert "Confirmed" in p11setup.result, p11setup.result

        p11 = FakeParams()
        await update_appointment(
            p11, "Bob Lee", new_date="2026-07-15", new_time="11:00"
        )
        ok11 = (
            "Rescheduled" in p11.result
            and "2026-07-15" in p11.result
            and "11:00" in p11.result
        )
        results.append(
            ("update_appointment reschedules to an open slot", ok11, p11.result)
        )

        p12a = FakeParams()
        await check_availability(p12a, "cardiology", "2026-07-14")
        ok12a = "09:00" in p12a.result
        results.append(("reschedule frees the old slot", ok12a, p12a.result))

        p12b = FakeParams()
        await book_appointment(
            p12b, "Someone Else", "cardiology", "Dr. Imran Malik", "2026-07-15", "11:00"
        )
        ok12b = "no longer available" in p12b.result
        results.append(("reschedule occupies the new slot", ok12b, p12b.result))

        p13 = FakeParams()
        await update_appointment(
            p13, "Bob Lee", new_date="2026-07-15", new_time="23:59"
        )
        ok13 = "isn't available" in p13.result and "Dr." in p13.result
        results.append(
            ("reschedule to invalid slot rejected with alternatives", ok13, p13.result)
        )

        p14 = FakeParams()
        await update_appointment(
            p14, "Bob Lee", new_date="2026-07-15", new_time="11:00"
        )
        ok14 = "already set for" in p14.result
        results.append(
            ("reschedule to identical slot is a graceful no-op", ok14, p14.result)
        )

        p15 = FakeParams()
        await update_appointment(
            p15, "Nobody Here", new_date="2026-07-15", new_time="09:00"
        )
        ok15 = "don't see any appointment" in p15.result
        results.append(("update: unknown name reported gracefully", ok15, p15.result))

        for desc, ok, detail in results:
            print(f"[{'PASS' if ok else 'FAIL'}] {desc}\n       -> {detail}")

        all_ok = all(ok for _, ok, _ in results)
        print(f"\n{'ALL PASS' if all_ok else 'FAILURES ABOVE'}")
        if not all_ok:
            print("DO NOT wire this into the pipeline until all cases pass.")
        return all_ok

    passed = asyncio.run(_run())
    exit(0 if passed else 1)
