"""Phase 0 regression test: the hospital pack's tools.py reaches the SAME
booking.py business logic as app/main.py, via the same compat.py shim -
proven by re-running booking.py's own self_test() through this pack's
import path, isolated to a tmp DB (never touches a real deployment's
hospital_bookings.db). Mirrors tests/test_booking.py's exact isolation
pattern.

Run:
    venv/bin/python -m pytest domain_agent_core/tests/test_hospital_pack_regression.py -v
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _REPO_ROOT / "app"
_CORE_DIR = _APP_DIR / "hospital_core"


def _import_booking_against(tmp_db_path: Path):
    """Same isolation trick as tests/test_booking.py: booking.py resolves
    DB_PATH once at import time, so reload then monkeypatch DB_PATH, then
    re-run _init_db() against the new path."""
    if str(_CORE_DIR) not in sys.path:
        sys.path.insert(0, str(_CORE_DIR))
    if "booking" in sys.modules:
        booking = importlib.reload(sys.modules["booking"])
    else:
        import booking
    booking.DB_PATH = tmp_db_path
    booking._init_db()
    return booking


def test_hospital_pack_tools_reach_the_real_booking_layer(tmp_path) -> None:
    """Importing domain_agent_core.packs.hospital.tools must resolve to
    the SAME compat.py -> hospital_core.booking chain app/main.py uses -
    proven by running booking.py's own self_test() against an isolated
    tmp DB, reached through this pack's own sys.path setup."""
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))

    booking = _import_booking_against(tmp_path / "hospital_bookings_test.db")
    assert asyncio.run(booking.self_test()) is True

    # Now confirm the pack's own tools module resolves to this exact
    # module object via compat.py, not a re-authored copy.
    from domain_agent_core.packs.hospital import tools as hospital_tools

    assert hospital_tools.compat._booking is sys.modules["booking"]
