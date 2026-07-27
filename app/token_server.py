"""
Join-token server + static host for the browser frontend.

This file is the ENTIRE replacement for main_hospital_lan.py's manual
signaling server. There are no /api/offer or ICE-patch routes to get wrong
anymore - livekit-server owns all WebRTC signaling. All this does:

  GET /api/token   -> mint a room-join JWT for the browser
  GET /            -> serve frontend/index.html (the reception console)

Defaults match `livekit-server --dev` exactly (API key "devkey", secret
"secret"), so the whole stack runs on one machine with zero config.

Run:
    uvicorn token_server:app --host 0.0.0.0 --port 7860

Phone on the same WiFi:
    http://<this-machine's-LAN-IP>:7860/
    Same secure-context caveat as before: browsers block the mic on plain
    http:// LAN IPs. Android Chrome: chrome://flags/
    #unsafely-treat-insecure-origin-as-secure -> add http://<ip>:7860.
    iPhone Safari needs a local HTTPS cert (mkcert) instead.
    IMPORTANT: LIVEKIT_WS_URL below must ALSO use the LAN IP (not
    localhost) or the phone will mint a token and then dial your phone's
    own loopback.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from livekit import api

load_dotenv()

LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
# What the BROWSER dials. For phone testing this must be ws://<LAN-IP>:7880.
LIVEKIT_WS_URL = os.environ.get("LIVEKIT_WS_URL", "ws://localhost:7880")

FRONTEND_DIR = Path(__file__).parent / "frontend"

app = FastAPI(title="Riverside General - reception console")


@app.get("/api/token")
async def token(identity: str | None = None, room: str | None = None) -> dict:
    """Mint a short-lived join token. One room per call keeps sessions
    isolated; the agent worker auto-dispatches into any new room."""
    identity = identity or f"caller-{uuid.uuid4().hex[:8]}"
    room = room or f"reception-{uuid.uuid4().hex[:8]}"
    jwt = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name("Caller")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return {"token": jwt, "url": LIVEKIT_WS_URL, "room": room, "identity": identity}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
