"""
Pipecat glue for safety_gate.py's pure-Python emergency check.

Sits between STT and the user aggregator/LLM. On an ordinary transcript it's
invisible - the frame passes through untouched. On a matched emergency, it
SWALLOWS the TranscriptionFrame (the real LLM never sees it, so it never
becomes an ordinary turn in conversation history) and instead pushes the
escalation message straight downstream as a TTSSpeakFrame - the same
"speak this directly, skip generation" mechanism tool_filler.py already uses
and that we verified works for the filler phrase.

Kept as a separate file from safety_gate.py on purpose: that file has zero
framework dependencies and its self-test runs anywhere (already verified,
15/15). This file is the thin part that actually needs pipecat.

HONEST LIMITATION: unlike safety_gate.py and tool_filler.py, this file could
NOT be executed end-to-end in the environment used to write it (no pipecat
install available there). The logic is a direct, minimal wrapping of
run_safety_gate() - already proven correct - around the same TTSSpeakFrame
injection pattern already proven to work in tool_filler.py. Both halves are
tested; this specific wiring of the two is not, yet. Before trusting it:
speak a known emergency phrase (e.g. "I'm having chest pain") into the real
pipeline and confirm (a) the escalation message is spoken and (b) the LLM
never generates a response for that turn - check the logs for the absence
of an OpenAILLMService TTFB line on that turn.
"""

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

try:
    from .safety import run_safety_gate
except ImportError:
    from safety import run_safety_gate


class SafetyGateProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            result = run_safety_gate(frame.text)
            if result is not None:
                logger.warning(
                    f"SafetyGateProcessor: emergency matched "
                    f"(category={result['category']}, kind={result['kind']}) "
                    f"- bypassing LLM, speaking escalation message directly"
                )
                # Same flagged API surface as elsewhere in this project:
                # verify `append_to_context` exists on your installed
                # pipecat's TTSSpeakFrame before relying on this.
                await self.push_frame(
                    TTSSpeakFrame(result["message"], append_to_context=False),
                    FrameDirection.DOWNSTREAM,
                )
                return  # swallow the transcript - LLM/context never see it

        await self.push_frame(frame, direction)
