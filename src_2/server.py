"""
server.py — Serves the voice pipeline over WebRTC so it can be reached from a
browser on another machine (e.g. your laptop) while STT/LLM/TTS run on this PC.

Install:
    pip install "pipecat-ai[webrtc]" pipecat-ai-small-webrtc-prebuilt

Run (from ~/voice-agent-pipeline):
    python -m src_2.server --host 0.0.0.0 --port 7860

Then open http://<PC-LAN-IP>:7860 in a browser on your laptop and click Connect.
"""

import argparse
from contextlib import asynccontextmanager
import os
from typing import Optional
import uuid

from fastapi import BackgroundTasks, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from loguru import logger
import uvicorn

# Pipecat & WebRTC imports
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from .main3 import run_bot

# Initialize NVML for local GPU monitoring
try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception as e:
    logger.warning(
        f"NVML could not be initialized. GPU metrics will be mocked. Error: {e}"
    )
    NVML_AVAILABLE = False

# Initialize the WebRTC handler
small_webrtc_handler = SmallWebRTCRequestHandler()


# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await small_webrtc_handler.close()
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


# Initialize FastAPI app instance
app = FastAPI(lifespan=lifespan)

# Mount static UI elements
app.mount("/client", SmallWebRTCPrebuiltUI)


# --- REST & WebSocket UI Endpoints ---


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass


@app.get("/api/status")
async def get_status():
    return {"status": "healthy", "agent": "idle"}


@app.get("/api/omni/status")
async def get_omni_status():
    return {"status": "online", "model": "vllm-local"}


@app.get("/api/vllm/metrics/all")
async def get_vllm_metrics():
    return {"requests_per_second": 0.0, "tokens_per_second": 0.0, "queue_size": 0}


@app.get("/api/gpu-status")
async def get_gpu_status():
    if not NVML_AVAILABLE:
        return {"gpu_utilization": "N/A", "vram_used": "0 GB", "vram_total": "0 GB"}

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        gpu_util = f"{util.gpu}%"
        vram_used = f"{round(mem.used / (1024**3), 2)} GB"
        vram_total = f"{round(mem.total / (1024**3), 2)} GB"

        return {
            "gpu_utilization": gpu_util,
            "vram_used": vram_used,
            "vram_total": vram_total,
        }
    except Exception as e:
        logger.error(f"Failed to fetch NVML metrics: {e}")
        return {"gpu_utilization": "Error", "vram_used": "0 GB", "vram_total": "0 GB"}


# --- Pipecat WebRTC Core Endpoints ---


@app.post("/start")
async def start_agent(request: Request):
    """Mimics Pipecat Cloud's /start endpoint that the Playground UI expects
    before it opens the WebRTC connection."""
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    session_id = str(uuid.uuid4())
    result = {"sessionId": session_id}

    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }

    return result


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def webrtc_connection_callback(connection: SmallWebRTCConnection):
        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )
        background_tasks.add_task(run_bot, transport)

    try:
        answer = await small_webrtc_handler.handle_web_request(
            request=request,
            webrtc_connection_callback=webrtc_connection_callback,
        )
    except Exception:
        logger.exception("Failed to handle WebRTC offer")
        raise
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


# --- Session-scoped aliases ---
# The Playground client namespaces its calls under /sessions/{session_id}/...
# once it has a sessionId from /start. We don't track per-session state
# server-side (SmallWebRTCRequestHandler already keys connections by their
# own pc_id), so these just forward straight through to the handlers above.


@app.post("/sessions/{session_id}/api/offer")
async def offer_session(
    session_id: str, request: SmallWebRTCRequest, background_tasks: BackgroundTasks
):
    return await offer(request, background_tasks)


@app.patch("/sessions/{session_id}/api/offer")
async def ice_candidate_session(session_id: str, request: SmallWebRTCPatchRequest):
    return await ice_candidate(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    logger.info(f"Starting WebRTC server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
