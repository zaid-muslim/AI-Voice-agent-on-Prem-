"""
Single-path voice pipeline (SOTA-latency build):

    mic -> TurnAudioRouter -> AudioNativeGemmaService (Gemma 4 12B, audio-in)
        -> QwenTTSService -> speaker

Whisper/Branch B REMOVED on purpose. Sending audio straight into Gemma is
ONE model before the first token; Whisper->Gemma would be TWO stacked, plus
a second GPU tenant on the 3090. The native path is the low-latency choice.
(If you ever need >30s single-turn handling, add Whisper back as a rare
fallback - it is not a latency win for normal turns.)

THE LATENCY FIX THAT MATTERS (read this):
Your logs showed the turn detector firing `strategy: None` at EXACTLY 5.000s
after you stopped talking, every turn - that flat 5s was ~75% of your total
latency. Cause: a smart-turn analyzer that returned INCOMPLETE 100% of the
time, so every turn fell through to its 5s max-wait. Gemma (0.3s) and TTS
(0.25s warm) were never the problem.
Fix here: end turns with plain Silero VAD at stop_secs=0.6, and DISABLE the
smart-turn analyzer (turn_analyzer=None). Deterministic ~0.6s turn-end.

    >>> Also launch vLLM with:  --gpu-memory-utilization 0.75  <<<
    (Whisper is gone, so vLLM can have more VRAM. TTS worker needs ~2-3GB.)
"""

import asyncio
import time

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.processors.aggregators.llm_context import LLMContext, ToolsSchema

from .router import TurnAudioRouter
from .services.audio_native_llm import AudioNativeGemmaService
from .tools.websearch import web_search
from .services.qwen_bridge import QwenTTSService

# --- config ---------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000/v1"
GEMMA_MODEL_NAME = "gemma-4-12b"

QWEN_TTS_MODEL_ID = "/home/nauyan/voice-agent-pipeline/models/Qwen3-TTS-0.6B-custom"
QWEN_TTS_SPEAKER = "aiden"
QWEN_TTS_LANGUAGE = "English"

SYSTEM_PROMPT = "You are a helpful, concise voice assistant."

# Set True when you're wearing HEADPHONES. Then real barge-in works and the
# echo mic-gate is turned off. With open SPEAKERS keep this False: the gate
# suppresses the bot's own voice looping back into the mic (the "results not
# good" phantom-turn behaviour you saw). Speakers without echo cancellation
# = strict walkie-talkie turn taking; that's physics, not a bug.
USE_HEADPHONES = False

# End-of-turn silence. 0.6s is a good conversational default: long enough not
# to cut you off mid-sentence, short enough to feel snappy. Lower toward 0.4
# for faster turnaround if you don't get clipped; raise toward 0.8 if it cuts
# you off when you pause to think.
VAD_STOP_SECS = 0.6


# --- end-to-end latency probe --------------------------------------------
# The ONLY latency number that reflects what the user feels: wall-clock from
# "user stopped speaking" to the first audio sample leaving the pipeline.
# The per-service TTFB metrics each measure their own slice and some measure
# nothing at all (Whisper's fake 15s TTFB was that). Trust THIS line.
class LatencyProbe(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._t_user_stopped = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._t_user_stopped = time.perf_counter()
        elif isinstance(frame, TTSAudioRawFrame) and self._t_user_stopped is not None:
            dt_ms = (time.perf_counter() - self._t_user_stopped) * 1000
            logger.info(
                f"END-TO-END latency: user-stopped -> first-audio = {dt_ms:.0f}ms"
            )
            self._t_user_stopped = None  # only the first chunk of each turn
        await self.push_frame(frame, direction)


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            # Plain VAD ends turns at VAD_STOP_SECS. This is the fast path.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
            # CRITICAL: no smart-turn analyzer. This is what removes the flat
            # 5-second fallback. If your pipecat build rejects this kwarg,
            # delete this one line and instead find where the smart-turn
            # analyzer is constructed and lower its stop_secs to ~1.0 - but
            # turn_analyzer=None is the clean off-switch and the reason turns
            # now end in ~0.6s instead of 5s.
            turn_analyzer=None,
        )
    )

    # Shared context. AudioNativeGemmaService manages history itself (it
    # appends the user placeholder + assistant text directly), so there is
    # NO user/assistant aggregator in this single path - that also avoids
    # double-logging each turn.
    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=ToolsSchema(standard_tools=[web_search]),
    )

    # Echo gate: on with speakers, off with headphones (barge-in enabled).
    router = TurnAudioRouter(gate_mic_while_bot_speaks=not USE_HEADPHONES)

    gemma = AudioNativeGemmaService(
        context=context,
        system_prompt=SYSTEM_PROMPT,
        model=GEMMA_MODEL_NAME,
        base_url=VLLM_BASE_URL,
        api_key="not-needed",
    )

    tts = QwenTTSService(
        model_id=QWEN_TTS_MODEL_ID,
        speaker=QWEN_TTS_SPEAKER,
        language=QWEN_TTS_LANGUAGE,
        sample_rate=24000,
        chunk_size=8,  # smaller first chunk = faster time-to-first-audio
    )

    pipeline = Pipeline(
        [
            transport.input(),
            router,
            gemma,
            tts,
            LatencyProbe(),
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=USE_HEADPHONES,  # only meaningful w/ headphones
        ),
        idle_timeout_secs=1200,
    )

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
