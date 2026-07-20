#!/usr/bin/env python3
"""Token server for the web frontend (Phase 1 web-surface migration).

Today's WS-based server.py has zero auth on its raw socket — this is the first place real auth
gets added: the browser must hit this endpoint to get a signed LiveKit access token before it can
join a room at all. Also serves web/ so the whole frontend + auth flow is one process/origin for
local dev, matching run.sh's single-`python3 -m http.server` simplicity in the original Pipeline.
"""
import os
import secrets
from contextlib import asynccontextmanager
from datetime import timedelta

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from livekit import api
from pydantic import BaseModel

import orchestrator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
# The LiveKit URL the *browser* dials for signaling+media. By default it's derived per-request
# from the host the browser used to reach this token server (see _livekit_url_for) — so a LAN
# client that loaded http://192.168.x.x:3000 gets ws://192.168.x.x:7880, and a tailnet client
# that loaded http://100.x.x.x:3000 gets ws://100.x.x.x:7880, each pointing at a host it can
# actually route to (LiveKit binds 0.0.0.0:7880, reachable at both). Set LIVEKIT_PUBLIC_URL only
# to FORCE one fixed URL for every client (e.g. an HTTPS reverse-proxy setup) — a fixed value is
# exactly what stops a non-tailnet LAN colleague from connecting when it's pinned to a tailnet IP.
LIVEKIT_PUBLIC_URL = os.environ.get("LIVEKIT_PUBLIC_URL", "")
LIVEKIT_RTC_PORT = int(os.environ.get("LIVEKIT_RTC_PORT", "7880"))
TOKEN_SERVER_PORT = int(os.environ.get("TOKEN_SERVER_PORT", "3000"))
WEB_DIR = os.path.join(PROJECT_ROOT, "web")


def _livekit_url_for(request: Request) -> str:
    """The LiveKit signaling URL this particular client should dial. Explicit override wins;
    otherwise derive it from the request host so each client reaches LiveKit at the same address
    it already reached this server on (no fixed tailnet/LAN assumption baked in)."""
    if LIVEKIT_PUBLIC_URL:
        return LIVEKIT_PUBLIC_URL
    host = (request.headers.get("host") or request.url.hostname or "").split(":")[0]
    scheme = "wss" if request.url.scheme == "https" else "ws"
    return f"{scheme}://{host}:{LIVEKIT_RTC_PORT}"

TOKEN_TTL_SECONDS = 6 * 60 * 60  # long enough for one call session; not a durable credential


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # vLLM/Chatterbox/the agent worker (if a selection was ever confirmed) are children of this
    # process, launched by orchestrator.py — tear them down when this process does, so Ctrl+C on
    # run.sh (which now just runs this server in the foreground) cleans up everything.
    await orchestrator.shutdown_all()


app = FastAPI(title="Bank Voice Agent — Token Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class TokenRequest(BaseModel):
    identity: str | None = None
    room: str | None = None


class TokenResponse(BaseModel):
    token: str
    url: str
    room: str
    identity: str


class SelectionRequest(BaseModel):
    llm: str
    stt: str
    tts: str


@app.post("/api/token", response_model=TokenResponse)
def issue_token(req: TokenRequest, request: Request) -> TokenResponse:
    livekit_url = _livekit_url_for(request)

    # A fresh room per call by default — one caller per room, exactly like today's one-WS-
    # connection-per-caller model; a caller-supplied room name (not used by the current
    # frontend) is honored for callers who want to rejoin a specific room.
    room = req.room or f"call-{secrets.token_hex(4)}"
    identity = req.identity or f"caller-{secrets.token_hex(4)}"

    grants = api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_ttl(timedelta(seconds=TOKEN_TTL_SECONDS))
        .with_grants(grants)
        .to_jwt()
    )
    return TokenResponse(token=token, url=livekit_url, room=room, identity=identity)


@app.get("/api/models")
def list_models() -> dict:
    """Id/label only — launch details (paths, ports, CUDA env) stay server-side."""
    cfg = orchestrator.load_models_config()
    return {
        category: [{"id": e["id"], "label": e["label"]} for e in entries]
        for category, entries in cfg.items()
    }


@app.post("/api/selection", status_code=202)
async def submit_selection(req: SelectionRequest) -> dict:
    try:
        await orchestrator.start_selection(req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"status": "starting"}


@app.get("/api/selection/status")
def selection_status() -> dict:
    return orchestrator.get_status()


# Serve the frontend last so it doesn't shadow /api/*.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=TOKEN_SERVER_PORT)
