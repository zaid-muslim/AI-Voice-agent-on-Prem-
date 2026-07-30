"""
dev_server.py - the developer-facing model/backend selector, SCHEMA V2 +
LATENCY COMPARISON.

WHAT CHANGED FROM V1:
  - /api/stt and /api/tts now take {"engine":..., "model":...} instead of
    a flat backend string - matches system_config.py's v2 schema (see
    that file's docstring for why).
  - NEW: GET /api/latency returns aggregated per-combination latency
    stats from latency_log.py, so you can actually SEE which
    LLM/STT/TTS combination performs best instead of guessing - this is
    what makes "try it and see" a real workflow instead of manual log
    reading.
  - NEW: POST /api/latency/clear resets the log (useful before a clean
    comparison run).

Everything else (auth pattern, LLM switching via vllm_manager with real
progress polling, the honest "LLM switch is not instant" design) is
unchanged from v1 - see vllm_manager.py's docstring for the full
reasoning.

RUN:
    DEV_PASSWORD=your-password VLLM_PYTHON=/path/to/vllm_venv/bin/python \\
        uvicorn dev_server:app --host 0.0.0.0 --port 7871
"""

from __future__ import annotations

import asyncio
import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

import latency_log
import system_config
import vllm_manager

app = FastAPI(title="Hospital Agent - Dev Console")
security = HTTPBasic()

DEV_PASSWORD = os.environ.get("DEV_PASSWORD")
if not DEV_PASSWORD:
    DEV_PASSWORD = secrets.token_urlsafe(12)
    print("=" * 68)
    print("  DEV_PASSWORD was not set. Generated a random one for this run:")
    print(f"      {DEV_PASSWORD}")
    print("  Set DEV_PASSWORD in the environment to choose your own.")
    print("=" * 68)

DEV_USERNAME = os.environ.get("DEV_USERNAME", "dev")

_manager = vllm_manager.VLLMManager(
    vllm_python=os.environ.get("VLLM_PYTHON", "vllm"),
    host=os.environ.get("VLLM_HOST", "0.0.0.0"),
    port=int(os.environ.get("VLLM_PORT", "8000")),
)


def _check_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    user_ok = secrets.compare_digest(credentials.username, DEV_USERNAME)
    pass_ok = secrets.compare_digest(credentials.password, DEV_PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class EngineModelChoice(BaseModel):
    engine: str
    model: str
    display_name: str | None = None


class LLMSwitchRequest(BaseModel):
    source: str
    served_model_name: str
    display_name: str | None = None


@app.get("/api/registry")
def get_registry(_: str = Depends(_check_auth)) -> dict:
    return system_config.load_registry()


@app.get("/api/config")
def get_config(_: str = Depends(_check_auth)) -> dict:
    cfg = system_config.get_config()
    cfg["_vllm_status"] = _manager.status()
    return cfg


@app.post("/api/tts")
def set_tts(choice: EngineModelChoice, _: str = Depends(_check_auth)) -> dict:
    cfg = system_config.get_config()
    cfg["tts"] = {
        "engine": choice.engine,
        "model": choice.model,
        "display_name": choice.display_name or f"{choice.engine}:{choice.model}",
    }
    try:
        system_config.save_config(cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "saved": True,
        "tts": cfg["tts"],
        "note": "Takes effect on the next call (see agent.py's entrypoint()).",
    }


@app.post("/api/stt")
def set_stt(choice: EngineModelChoice, _: str = Depends(_check_auth)) -> dict:
    cfg = system_config.get_config()
    cfg["stt"] = {
        "engine": choice.engine,
        "model": choice.model,
        "display_name": choice.display_name or f"{choice.engine}:{choice.model}",
    }
    try:
        system_config.save_config(cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "saved": True,
        "stt": cfg["stt"],
        "note": (
            "Takes effect for the NEXT newly-spawned worker process. An "
            "already-warmed/pooled worker keeps its current STT until "
            "recycled - see agent.py prewarm()'s docstring."
        ),
    }


@app.post("/api/llm/switch")
async def switch_llm(req: LLMSwitchRequest, _: str = Depends(_check_auth)) -> dict:
    if _manager.status()["switching"]:
        raise HTTPException(status_code=409, detail="A switch is already in progress.")

    async def _run():
        display = req.display_name or req.served_model_name
        result = await _manager.switch_model(
            source=req.source, served_model_name=req.served_model_name
        )
        if result["ok"]:
            cfg2 = system_config.get_config()
            cfg2["llm"]["display_name"] = display
            system_config.save_config(cfg2)

    asyncio.create_task(_run())
    return {"switching": True, "note": "Poll /api/llm/status for progress."}


@app.get("/api/llm/status")
def llm_status(_: str = Depends(_check_auth)) -> dict:
    return _manager.status()


@app.get("/api/latency")
def get_latency(_: str = Depends(_check_auth)) -> dict:
    """Aggregated latency stats per LLM/STT/TTS combination actually used
    on real calls - see latency_log.py for exactly what's tracked and why
    per-event (not per-turn-combined) records were the honest choice."""
    return {"combinations": latency_log.get_summary()}


@app.post("/api/latency/clear")
def clear_latency(_: str = Depends(_check_auth)) -> dict:
    latency_log.clear()
    return {"cleared": True}


@app.get("/", response_class=HTMLResponse)
def dev_ui() -> str:
    from pathlib import Path

    return (Path(__file__).parent / "dev_ui.html").read_text(encoding="utf-8")
