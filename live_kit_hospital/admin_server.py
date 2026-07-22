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

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

# Make hospital_core importable (same trick compat.py uses).
_CORE_DIR = Path(__file__).parent / "hospital_core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

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


@app.get("/", response_class=HTMLResponse)
def admin_ui() -> str:
    """Serve the admin UI page. The page itself prompts for the password
    via the browser's basic-auth dialog on its first /api call."""
    ui_path = Path(__file__).parent / "admin_ui.html"
    return ui_path.read_text(encoding="utf-8")
