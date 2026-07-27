"""
Real automated coverage for hospital_core/booking.py, wrapping its own
self_test() (double-booking, fuzzy matching, reschedule, the past-time
filter, etc - see that function's docstring) in a temp database so this
never touches a real deployment's hospital_bookings.db.

Run:
    pytest tests/test_booking.py -v
"""

import asyncio
import importlib
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
CORE_DIR = APP_DIR / "hospital_core"


def _import_booking_against(tmp_db_path: Path):
    """Import hospital_core/booking.py fresh, pointed at an isolated DB
    file - booking.py resolves DB_PATH once at import time relative to its
    own file location, so we monkeypatch that attribute after import
    rather than trying to parameterize the module itself."""
    if str(CORE_DIR) not in sys.path:
        sys.path.insert(0, str(CORE_DIR))
    if "booking" in sys.modules:
        booking = importlib.reload(sys.modules["booking"])
    else:
        import booking  # noqa: PLC0415
    booking.DB_PATH = tmp_db_path
    # booking.py's module-level `_init_db()` call already ran once against
    # the path DB_PATH had AT IMPORT TIME, before we reassigned it above -
    # re-run it now so the isolated tmp DB actually gets its tables/seed
    # data (self_test() also does this itself, but the other tests below
    # call booking functions directly without going through self_test()).
    booking._init_db()
    return booking


def test_booking_self_test(tmp_path):
    booking = _import_booking_against(tmp_path / "hospital_bookings_test.db")
    assert asyncio.run(booking.self_test()) is True


def test_add_recurring_schedule_and_remove_slot(tmp_path):
    booking = _import_booking_against(tmp_path / "hospital_bookings_test2.db")

    result = asyncio.run(
        booking.add_recurring_schedule(
            department="Neurology",
            doctor="Dr. Test Person",
            weekdays=[0, 2, 4],  # Mon/Wed/Fri
            start_time="09:00",
            end_time="10:00",
            slot_minutes=30,
            weeks=1,
        )
    )
    assert result["status"] == "ok"
    assert result["generated"] > 0
    assert result["added"] == result["generated"]

    slots = asyncio.run(booking.list_doctor_slots("Neurology", "Dr. Test Person"))
    assert len(slots) == result["added"]
    assert all(not s["booked"] for s in slots)

    one = slots[0]
    removed = asyncio.run(
        booking.remove_doctor_slot("Neurology", "Dr. Test Person", one["date"], one["time"])
    )
    assert removed["status"] == "ok"

    slots_after = asyncio.run(booking.list_doctor_slots("Neurology", "Dr. Test Person"))
    assert len(slots_after) == result["added"] - 1


def test_remove_slot_refuses_when_booked(tmp_path):
    booking = _import_booking_against(tmp_path / "hospital_bookings_test3.db")

    asyncio.run(
        booking.add_doctor_slots(
            "cardiology", "Dr. Imran Malik", [("2099-01-01", "09:00")]
        )
    )
    booked = asyncio.run(
        booking.book_appointment(
            "Test Patient", "cardiology", "2099-01-01", "09:00", doctor="Dr. Imran Malik"
        )
    )
    assert booked["status"] == "booked"

    refused = asyncio.run(
        booking.remove_doctor_slot("cardiology", "Dr. Imran Malik", "2099-01-01", "09:00")
    )
    assert refused["status"] == "error"
    assert "booked" in refused["message"].lower()
