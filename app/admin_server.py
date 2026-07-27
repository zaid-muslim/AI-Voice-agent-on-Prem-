"""
Hospital data admin server - a SEPARATE, password-protected internal web
app for hospital admin staff to edit doctors, departments, hours, and other
informational content WITHOUT editing Python code.

WHAT IT DOES:
  - Serves the admin UI (admin_ui.html) at /
  - GET  /api/entries        -> current hospital data (all entries)
  - POST /api/entries        -> save edited data (validates, writes JSON,
                                reloads, and re-embeds RAG so changes take
                                effect on the NEXT call immediately)
  - All endpoints require a password (set via ADMIN_PASSWORD env var).

WHY SEPARATE FROM token_server.py: token_server mints join tokens for
callers and must stay simple and public-facing. This admin app changes
production behaviour and is password-gated - keeping them separate means a
bug or exposure in one doesn't affect the other. Run it on its own port.

THE "IMMEDIATE" GUARANTEE (per the requirement): when an admin saves,
this calls hospital_kb.save_kb() (writes + reloads the data) and then
rag.reindex() (rebuilds semantic-search embeddings). Any call that STARTS
after the save sees the new data. A call already in progress keeps the data
it started with - which is the correct, safe behaviour (you don't want a
doctor's name changing mid-conversation).

IMPORTANT CROSS-PROCESS NOTE: the live agent runs as its OWN separate
worker process(es) with their OWN in-memory copy of the data and
embeddings. This admin server reloading/re-embedding updates THIS process.
For the agent workers to pick up the change, they re-read the JSON file
when each new call/session starts (hospital_kb loads from the file at
import, and warm_rag re-embeds per worker). In practice, for a hospital
admin editing between calls, this is effectively immediate. If you later
run long-lived pre-warmed workers that never re-read, you'd add a
file-watch or a small "reload" signal - noted here so it's a known,
deliberate design point, not a surprise.

RUN:
    ADMIN_PASSWORD=your-password uvicorn admin_server:app --host 0.0.0.0 --port 7870

Then open http://<LAN-IP>:7870/ and log in with that password.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

import re
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

# Make hospital_core importable (same trick compat.py uses).
_CORE_DIR = Path(__file__).parent / "hospital_core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

import booking  # noqa: E402
import hospital_kb  # noqa: E402

try:
    import rag  # noqa: E402
except Exception:  # noqa: BLE001
    rag = None  # RAG optional - admin editing still works without it

app = FastAPI(title="Hospital Data Admin")
security = HTTPBasic()

# Password from env. If unset, generate a random one and print it, so the
# server is NEVER accidentally left open with no/default password.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(12)
    print("=" * 68)
    print("  ADMIN_PASSWORD was not set. Generated a random one for this run:")
    print(f"      {ADMIN_PASSWORD}")
    print("  Set ADMIN_PASSWORD in the environment to choose your own.")
    print("=" * 68)

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")


def _check_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Constant-time credential check (secrets.compare_digest avoids timing
    attacks that a plain == would allow)."""
    user_ok = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    pass_ok = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class Entry(BaseModel):
    id: str
    category: str
    title: str
    text: str


class SavePayload(BaseModel):
    entries: list[Entry]


class DoctorCreate(BaseModel):
    name: str
    department: str
    bio: Optional[str] = None
    # Optional one-shot initial recurring schedule, so staff can add a
    # doctor and their timings in a single form submission.
    weekdays: Optional[list[int]] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    slot_minutes: int = 30
    weeks: int = 4


class SlotEntry(BaseModel):
    date: str
    time: str


class SlotsAddPayload(BaseModel):
    department: str
    doctor: str
    mode: str  # "explicit" or "recurring"
    slots: Optional[list[SlotEntry]] = None  # mode="explicit"
    weekdays: Optional[list[int]] = None  # mode="recurring", below
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    slot_minutes: int = 30
    weeks: int = 4
    start_date: Optional[str] = None


class SlotRemove(BaseModel):
    department: str
    doctor: str
    date: str
    time: str


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


@app.get("/api/entries")
def get_entries(_: str = Depends(_check_auth)) -> dict:
    """Return the current hospital data plus the list of known categories
    (so the UI can offer them as a dropdown)."""
    hospital_kb.reload_kb()  # always serve the freshest from disk
    categories = sorted({e["category"] for e in hospital_kb.HOSPITAL_KB})
    return {"entries": hospital_kb.HOSPITAL_KB, "categories": categories}


@app.post("/api/entries")
def save_entries(payload: SavePayload, _: str = Depends(_check_auth)) -> dict:
    """Validate + save edited data, then re-embed RAG so semantic search
    reflects it on the next call. Returns any non-blocking warnings and
    whether RAG was re-indexed."""
    entries = [e.model_dump() for e in payload.entries]
    try:
        warnings = hospital_kb.save_kb(
            entries
        )  # raises ValueError on blocking problems
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    reindexed = False
    reindex_note = "RAG module not loaded in this process"
    if rag is not None:
        try:
            reindexed = rag.reindex()
            reindex_note = (
                "semantic embeddings rebuilt"
                if reindexed
                else "running on keyword fallback (no embeddings to rebuild)"
            )
        except Exception as exc:  # noqa: BLE001
            reindex_note = f"reindex failed (data still saved): {exc}"

    return {
        "saved": True,
        "count": len(entries),
        "warnings": warnings,
        "reindexed": reindexed,
        "reindex_note": reindex_note,
    }


@app.get("/api/doctors")
async def get_doctors(_: str = Depends(_check_auth)) -> dict:
    """Merge the bio side (hospital_kb roster) with the schedule side
    (booking.list_doctors' slot counts) into one list for the admin UI's
    doctor picker. A doctor with a bio but zero slots (freshly added, no
    timings yet) still appears, with total_slots/upcoming_slots at 0."""
    hospital_kb.reload_kb()
    bios = {d["name"]: d for d in hospital_kb.get_doctor_roster()}
    scheduled = {d["doctor"]: d for d in await booking.list_doctors()}

    merged = {}
    for name, bio in bios.items():
        merged[name] = {**bio, "total_slots": 0, "upcoming_slots": 0}
    for name, sched in scheduled.items():
        entry = merged.setdefault(
            name, {"name": name, "department": sched["department"].title(), "bio": ""}
        )
        entry["total_slots"] = sched["total_slots"]
        entry["upcoming_slots"] = sched["upcoming_slots"]

    departments = sorted({e["department"] for e in merged.values()})
    return {"doctors": sorted(merged.values(), key=lambda d: d["name"]), "departments": departments}


@app.get("/api/doctors/slots")
async def get_doctor_slots(
    department: str, doctor: str, _: str = Depends(_check_auth)
) -> dict:
    """Every slot (open or booked) for one doctor - the admin schedule
    view, deliberately including booked slots so staff see the full
    picture, not just what's still available to callers."""
    slots = await booking.list_doctor_slots(department, doctor)
    return {"department": department, "doctor": doctor, "slots": slots}


@app.post("/api/doctors")
async def create_doctor(payload: DoctorCreate, _: str = Depends(_check_auth)) -> dict:
    """Register a new doctor: writes a hospital_kb bio entry (so the AI can
    talk about them) AND, if an initial schedule was given, seeds their
    first batch of bookable slots in the same call - closing the "two
    sources of truth must be kept in sync manually" gap the hospital_kb/
    booking modules' own docstrings warn about.
    """
    name = payload.name.strip()
    department = payload.department.strip()
    if not name or not department:
        raise HTTPException(status_code=400, detail="name and department are required")

    doctor_id = f"doctor_{_slugify(name)}"
    dept_id = f"dept_{_slugify(department)}"

    hospital_kb.reload_kb()
    entries = [dict(e) for e in hospital_kb.HOSPITAL_KB]

    if any(e["id"] == doctor_id for e in entries):
        raise HTTPException(
            status_code=409,
            detail=f"A doctor entry for '{name}' already exists - edit it from the "
            "hospital data admin page instead.",
        )

    short_name = name if name.lower().startswith("dr") else f"Dr. {name}"
    bio_text = payload.bio or (
        f"{short_name} practices in {department} at Riverside General."
    )
    entries.append(
        {
            "id": doctor_id,
            "category": "doctors",
            "title": f"{short_name} - {department}",
            "text": bio_text,
        }
    )

    dept_entry = next(
        (e for e in entries if e["id"] == dept_id and e["category"] == "departments"),
        None,
    )
    if dept_entry is None:
        entries.append(
            {
                "id": dept_id,
                "category": "departments",
                "title": f"{department} department",
                "text": f"The {department} department at Riverside General. Doctors: {short_name}.",
            }
        )
    elif short_name.replace("Dr. ", "") not in dept_entry["text"]:
        dept_entry["text"] = dept_entry["text"].rstrip() + f" {short_name} also practices here."

    try:
        warnings = hospital_kb.save_kb(entries)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    reindexed = False
    if rag is not None:
        try:
            reindexed = rag.reindex()
        except Exception:  # noqa: BLE001
            pass

    schedule_result = None
    if payload.weekdays and payload.start_time and payload.end_time:
        schedule_result = await booking.add_recurring_schedule(
            department=department,
            doctor=short_name,
            weekdays=payload.weekdays,
            start_time=payload.start_time,
            end_time=payload.end_time,
            slot_minutes=payload.slot_minutes,
            weeks=payload.weeks,
        )

    return {
        "saved": True,
        "doctor": short_name,
        "department": department,
        "warnings": warnings,
        "reindexed": reindexed,
        "schedule": schedule_result,
    }


@app.post("/api/doctors/slots")
async def add_slots(payload: SlotsAddPayload, _: str = Depends(_check_auth)) -> dict:
    """Add new timings for an existing doctor - either an explicit list of
    (date, time) slots (mode="explicit"), or a recurring weekly pattern
    (mode="recurring", e.g. every Mon/Wed/Fri 09:00-13:00 for 4 weeks).
    This is the actual "add doctor timings" action the admin UI's schedule
    form calls."""
    if payload.mode == "recurring":
        if not (payload.weekdays and payload.start_time and payload.end_time):
            raise HTTPException(
                status_code=400,
                detail="recurring mode requires weekdays, start_time, end_time",
            )
        return await booking.add_recurring_schedule(
            department=payload.department,
            doctor=payload.doctor,
            weekdays=payload.weekdays,
            start_time=payload.start_time,
            end_time=payload.end_time,
            slot_minutes=payload.slot_minutes,
            weeks=payload.weeks,
            start_date=payload.start_date,
        )
    if payload.mode == "explicit":
        if not payload.slots:
            raise HTTPException(status_code=400, detail="explicit mode requires slots")
        slots = [(s.date, s.time) for s in payload.slots]
        return await booking.add_doctor_slots(payload.department, payload.doctor, slots)
    raise HTTPException(status_code=400, detail="mode must be 'explicit' or 'recurring'")


@app.delete("/api/doctors/slots")
async def remove_slot(payload: SlotRemove, _: str = Depends(_check_auth)) -> dict:
    """Remove a single, not-yet-booked slot (mistakes happen - staff need a
    way to undo an added timing without touching the database directly)."""
    return await booking.remove_doctor_slot(
        payload.department, payload.doctor, payload.date, payload.time
    )


@app.get("/", response_class=HTMLResponse)
def admin_ui() -> str:
    """Serve the admin UI page. The page itself prompts for the password
    via the browser's basic-auth dialog on its first /api call."""
    ui_path = Path(__file__).parent / "admin_ui.html"
    return ui_path.read_text(encoding="utf-8")
