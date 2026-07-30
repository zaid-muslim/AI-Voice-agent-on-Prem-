"""
call_server.py - lets you actually CALL the direct-audio agent from a
browser, the same way app/token_server.py lets you call the existing
one - but on its own port, minting tokens with an EXPLICIT agent
dispatch (RoomAgentDispatch(agent_name="direct-audio-receptionist"))
instead of relying on auto-dispatch.

WHY A SEPARATE SERVER INSTEAD OF REUSING app/token_server.py: that file
mints plain room-join tokens with no dispatch metadata, which only works
because the existing agent worker (app/main.py) auto-dispatches into any
new room. agent.py's worker deliberately does NOT auto-dispatch (see its
docstring - agent_name="direct-audio-receptionist" turns that off so it
can't interfere with the existing worker). A token from THIS server is
what tells LiveKit "dispatch the direct-audio agent to this room, not
the default one." app/token_server.py itself is not imported, read, or
modified - this is a new, independent file.

Reuses app/frontend/index.html AS-IS (served straight from its real
path, not copied) so the calling UI is identical to what you already
know - only which agent gets dispatched differs.

Run:
    venv/bin/python direct_audio_agent/call_server.py
Then open http://<this-machine>:7862/ (same LAN-IP caveats as
app/token_server.py's docstring - mic access needs a secure context).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
APP_DIR = THIS_DIR.parent / "app"
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(APP_DIR / ".env")

import os  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from livekit import api  # noqa: E402

from helpers import require_real_livekit_credentials  # noqa: E402 - app/helpers.py, unmodified
from agent import AGENT_NAME  # noqa: E402 - this folder's agent.py

LIVEKIT_API_KEY, LIVEKIT_API_SECRET = require_real_livekit_credentials()
LIVEKIT_WS_URL = os.environ.get("LIVEKIT_WS_URL", "ws://localhost:7880")
CALL_SERVER_PORT = int(os.environ.get("DIRECT_AUDIO_CALL_SERVER_PORT", "7862"))

FRONTEND_DIR = APP_DIR / "frontend"

app = FastAPI(title="Direct-Audio Gemma - reception console")


@app.get("/api/token")
async def token(identity: str | None = None, room: str | None = None) -> dict:
    identity = identity or f"caller-{uuid.uuid4().hex[:8]}"
    room = room or f"direct-audio-{uuid.uuid4().hex[:8]}"
    jwt = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name("Caller")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .with_room_config(
            api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)])
        )
        .to_jwt()
    )
    return {"token": jwt, "url": LIVEKIT_WS_URL, "room": room, "identity": identity}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=CALL_SERVER_PORT)
