"""
Single-path voice pipeline: mic -> STT -> LLM (vLLM/Gemma) -> TTS -> speaker.

    transport.input -> Whisper STT -> user_agg -> Gemma (vLLM) -> Qwen TTS
        -> transport.output -> assistant_agg

This is the linear design confirmed working end-to-end (~0.45-1.25s from
user-stop to first audio in real runs). The dual-path native-audio router
was removed on purpose: every hard bug came from routing frames between two
branches, never from STT/LLM/TTS themselves. router.py and
services/audio_native_llm.py remain in the repo, unused, if Branch A is ever
worth revisiting in isolation.

Run (from ~/voice-agent-pipeline, one level above src_2/):
    python -m src_2.main

vLLM launch:  --gpu-memory-utilization 0.70   (leave room for Whisper + TTS)
"""

import asyncio

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_context import LLMContext, ToolsSchema
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

from .tools.websearch import web_search
from .services.qwen_bridge import QwenTTSService

VLLM_BASE_URL = "http://localhost:8000/v1"
GEMMA_MODEL_NAME = "gemma-4-12b"  # must match --served-model-name in vllm serve

QWEN_TTS_MODEL_ID = "/home/nauyan/voice-agent-pipeline/models/Qwen3-TTS-0.6B-custom"
QWEN_TTS_SPEAKER = "aiden"
QWEN_TTS_LANGUAGE = "English"

# Short, voice-appropriate replies. Without this the model can ramble for
# 400+ tokens / ~9s (seen in logs) - fine for chat, terrible for voice.
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Answer in one or two short sentences "
    "unless asked for more detail. Keep responses conversational and brief."
)

# Cap generation so a single answer can't monopolise the turn. ~150 tokens is
# a few spoken sentences.
MAX_TOKENS = 150

# End-of-turn silence. 0.5s is a good conversational default; lower to ~0.4
# for snappier turnaround, raise to ~0.7 if it cuts you off mid-thought.
VAD_STOP_SECS = 0.5


async def main():
    # 16kHz in for Silero VAD + Whisper, 24kHz out for Qwen TTS.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        )
    )

    stt = WhisperSTTService(
        model="distil-medium.en",
        device="cuda",
        compute_type="int8_float16",
        ttfs_p99_latency=0.2,
    )

    llm_text = OpenAILLMService(
        api_key="not-needed",
        base_url=VLLM_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=GEMMA_MODEL_NAME,
            max_tokens=MAX_TOKENS,
        ),
    )
    llm_text.register_direct_function(web_search)

    tts = QwenTTSService(
        model_id=QWEN_TTS_MODEL_ID,
        speaker=QWEN_TTS_SPEAKER,
        language=QWEN_TTS_LANGUAGE,
        sample_rate=24000,
        chunk_size=8,  # smaller first chunk = faster time-to-first-audio
    )

    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=ToolsSchema(standard_tools=[web_search]),
    )
    # NOT tuple-unpackable in this pipecat version - use .user()/.assistant().
    aggregators = LLMContextAggregatorPair(context)
    user_aggregator = aggregators.user()
    assistant_aggregator = aggregators.assistant()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
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


if __name__ == "__main__":
    asyncio.run(main())
