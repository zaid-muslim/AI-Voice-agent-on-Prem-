"""
LAN/browser testing entry point - SmallWebRTCTransport instead of
LocalAudioTransport, so you can open a page in your phone's browser (same
WiFi network) and talk to the SAME hospital agent.

FIXED IN THIS REVISION - matches a PROVEN WORKING reference file from a
sibling project on the same machine (src_2/server.py), rather than guessed
API shapes. Two real bugs this fixes:

1. /api/offer was type-hinted with plain `Request` and manually parsed as a
   dict, but handle_web_request() actually requires the real Pydantic
   model - SmallWebRTCRequest (it has a REQUIRED .pc_id attribute a plain
   dict doesn't have, confirmed by the AttributeError in the traceback this
   was debugged from). FastAPI auto-validates the JSON body into this model
   when you type-hint the route parameter with it directly - no manual
   .json() parsing needed or wanted.

2. /start's response shape now mirrors the proven reference exactly
   ({"sessionId": ...}, optionally with "iceConfig") rather than the
   {"webrtcUrl": ...} shape from Pipecat's generic client-js docs - the
   installed prebuilt UI bundle apparently expects the former. Also added
   the /sessions/{session_id}/api/offer aliases the reference includes,
   as a defensive match in case this bundle version calls those instead of
   plain /api/offer once it has a sessionId.

WHY THIS FILE LOOKS DIFFERENT FROM THE ORIGINAL ATTEMPT:
The very first version used Pipecat's convenience CLI (`pipecat.runner.run`),
which requires Python 3.11+ (imports `http.HTTPMethod`, added in 3.11) -
your venv is 3.10.12, confirmed in an earlier traceback. This version's
manual FastAPI server has no such requirement.

SETUP (once):
    pip install "pipecat-ai[webrtc]" fastapi uvicorn
    pip install pipecat-ai-small-webrtc-prebuilt

RUN:
    python main_hospital_lan.py

ON YOUR PHONE (same WiFi as this machine):
    http://<this-machine's-LAN-IP>:7860/prebuilt/
    (Confirmed reachable already - your server log showed 200 OK on
    /prebuilt/ and its assets before this fix.)

CRITICAL BROWSER CAVEAT, unchanged from before:
Browsers only allow microphone access over a "secure context" - https://,
or http://localhost/127.0.0.1. A plain http://192.168.x.x URL will
SILENTLY block mic permission on most phone browsers.
  - Android Chrome: chrome://flags/#unsafely-treat-insecure-origin-as-secure
    -> add your http://<ip>:7860 to the allowlist.
  - iPhone Safari has no such flag - needs a local HTTPS cert (mkcert)
    instead.
"""

import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import RedirectResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_context import LLMContext, ToolsSchema
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

# Reuse EVERYTHING from main_hospital.py - same tools, same prompt, same
# model config. This file only ever changes the transport.
try:
    from .safety_gate_processor import SafetyGateProcessor
    from .booking import (
        check_availability,
        book_appointment,
        cancel_appointment,
        update_appointment,
    )
    from .rag import search_hospital_info
    from .qwen_bridge import QwenTTSService
    from .main_hospital import (
        _build_system_prompt,
        VLLM_BASE_URL,
        GEMMA_MODEL_NAME,
        QWEN_TTS_MODEL_ID,
        QWEN_TTS_SPEAKER,
        QWEN_TTS_LANGUAGE,
        MAX_TOKENS,
        VAD_STOP_SECS,
    )
except ImportError:
    from safety_gate_processor import SafetyGateProcessor
    from booking import (
        check_availability,
        book_appointment,
        cancel_appointment,
        update_appointment,
    )
    from rag import search_hospital_info
    from qwen_bridge import QwenTTSService
    from main_hospital import (
        _build_system_prompt,
        VLLM_BASE_URL,
        GEMMA_MODEL_NAME,
        QWEN_TTS_MODEL_ID,
        QWEN_TTS_SPEAKER,
        QWEN_TTS_LANGUAGE,
        MAX_TOKENS,
        VAD_STOP_SECS,
    )


async def run_bot(transport):
    """Identical pipeline to main_hospital.py's main() - only the transport
    object passed in differs."""
    stt = WhisperSTTService(
        model="distil-medium.en",
        device="cuda",
        compute_type="int8_float16",
        ttfs_p99_latency=0.2,
    )

    safety_gate = SafetyGateProcessor()

    llm_text = OpenAILLMService(
        api_key="not-needed",
        base_url=VLLM_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=GEMMA_MODEL_NAME,
            max_tokens=MAX_TOKENS,
        ),
    )
    llm_text.register_direct_function(check_availability)
    llm_text.register_direct_function(book_appointment)
    llm_text.register_direct_function(cancel_appointment)
    llm_text.register_direct_function(update_appointment)
    llm_text.register_direct_function(search_hospital_info)

    tts = QwenTTSService(
        model_id=QWEN_TTS_MODEL_ID,
        speaker=QWEN_TTS_SPEAKER,
        language=QWEN_TTS_LANGUAGE,
        sample_rate=24000,
        chunk_size=8,
    )

    context = LLMContext(
        messages=[{"role": "system", "content": _build_system_prompt()}],
        tools=ToolsSchema(
            standard_tools=[
                check_availability,
                book_appointment,
                cancel_appointment,
                update_appointment,
                search_hospital_info,
            ]
        ),
    )
    aggregators = LLMContextAggregatorPair(context)
    user_aggregator = aggregators.user()
    assistant_aggregator = aggregators.assistant()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            safety_gate,
            user_aggregator,
            llm_text,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
        ),
        idle_timeout_secs=1200,
    )

    runner = PipelineRunner()
    await runner.run(task)


# --- WebRTC signaling server, mirroring the proven-working reference -----

webrtc_handler = SmallWebRTCRequestHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await webrtc_handler.close()


app = FastAPI(lifespan=lifespan)
app.mount("/prebuilt", SmallWebRTCPrebuiltUI)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/prebuilt/")


@app.post("/start")
async def start_agent(request: Request):
    """Mirrors the proven reference's /start response shape exactly -
    {"sessionId": ...}, optionally with iceConfig. This is what the
    installed prebuilt UI bundle expects, confirmed working elsewhere on
    this same machine."""
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
    """FIXED: type-hinted with the real Pydantic model (SmallWebRTCRequest)
    instead of plain Request + manual dict parsing - FastAPI validates the
    JSON body into this automatically, which is what handle_web_request()
    actually requires (it reads request.pc_id internally)."""

    async def webrtc_connection_callback(connection: SmallWebRTCConnection):
        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(stop_secs=VAD_STOP_SECS)
                ),
            ),
        )
        background_tasks.add_task(run_bot, transport)

    try:
        answer = await webrtc_handler.handle_web_request(
            request=request,
            webrtc_connection_callback=webrtc_connection_callback,
        )
    except Exception:
        logger.exception("Failed to handle WebRTC offer")
        raise
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


# Session-scoped aliases - defensive match to the proven reference, in case
# this prebuilt bundle version calls these (namespaced under the sessionId
# from /start) instead of plain /api/offer once it has one.
@app.post("/sessions/{session_id}/api/offer")
async def offer_session(
    session_id: str, request: SmallWebRTCRequest, background_tasks: BackgroundTasks
):
    return await offer(request, background_tasks)


@app.patch("/sessions/{session_id}/api/offer")
async def ice_candidate_session(session_id: str, request: SmallWebRTCPatchRequest):
    return await ice_candidate(request)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
