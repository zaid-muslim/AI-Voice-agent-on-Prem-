"""call_server.py - lets you call a domain_agent_core worker from a
browser, generalizing ``direct_audio_agent/call_server.py``'s pattern:
mints tokens with an EXPLICIT agent dispatch (``RoomAgentDispatch
(agent_name=...)``), generalized to whichever domain's ``agent_name`` the
``DOMAIN_PACK`` environment variable selects, instead of one hardcoded
constant. This is what lets ``hospital-agent`` and ``banking-agent``
workers run concurrently without either ever claiming the other's room -
same explicit-dispatch mechanism, reused verbatim, not reinvented.

Reuses ``app/frontend/index.html`` AS-IS (served straight from its real
path, not copied).

Run:
    DOMAIN_PACK=hospital venv/bin/python domain_agent_core/call_server.py
    DOMAIN_PACK=banking  venv/bin/python domain_agent_core/call_server.py
Then open http://<this-machine>:<port>/ - port defaults per domain, see
``_DEFAULT_PORTS`` below, or override with ``CALL_SERVER_PORT``.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from dotenv import load_dotenv

load_dotenv(_APP_DIR / ".env")

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from helpers import (
    require_real_livekit_credentials,
)
from livekit import api

from domain_agent_core.core.domain_loader import AgentAssembler

DOMAIN_PACK = os.environ.get("DOMAIN_PACK")
if not DOMAIN_PACK:
    raise RuntimeError(
        "DOMAIN_PACK environment variable is not set - which domain's "
        "call server is this? (e.g. DOMAIN_PACK=hospital)."
    )

_assembled = AgentAssembler().assemble(DOMAIN_PACK)
AGENT_NAME = _assembled.pack.agent_name

# Every domain gets its own default port so hospital/banking call servers
# can run side by side without a collision - override per-deployment with
# CALL_SERVER_PORT if needed.
_DEFAULT_PORTS = {"hospital": 7863, "banking": 7864}

LIVEKIT_API_KEY, LIVEKIT_API_SECRET = require_real_livekit_credentials()
LIVEKIT_WS_URL = os.environ.get("LIVEKIT_WS_URL", "ws://localhost:7880")
CALL_SERVER_PORT = int(
    os.environ.get("CALL_SERVER_PORT", str(_DEFAULT_PORTS.get(DOMAIN_PACK, 7865)))
)

FRONTEND_DIR = _APP_DIR / "frontend"

app = FastAPI(title=f"{_assembled.pack.display_name} - reception console")


@app.get("/api/token")
async def token(identity: str | None = None, room: str | None = None) -> dict:
    """Mint a short-lived, explicitly-dispatched LiveKit join token.

    Args:
        identity: Caller identity. Auto-generated if omitted.
        room: Room name. Auto-generated (``<domain_id>-<8 hex>``) if
            omitted, which is the normal case.

    Returns:
        A dict with the signed ``token``, the ``url`` to dial, and the
        resolved ``room``/``identity``.
    """
    identity = identity or f"caller-{uuid.uuid4().hex[:8]}"
    room = room or f"{DOMAIN_PACK}-{uuid.uuid4().hex[:8]}"
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
    """Serve the reception console's single HTML entry point."""
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=CALL_SERVER_PORT)
