"""
Audio-native Gemma LLM (Branch A).

Sends whole-turn PCM audio straight to vLLM's OpenAI-compatible endpoint
as a multimodal `input_audio` block, using Gemma 4 12B's native
(encoder-free) audio input. Bypasses pipecat's LLMContext/run_inference
machinery for this branch and calls the OpenAI async client directly,
then emits standard LLMFullResponseStartFrame / LLMTextFrame /
LLMFullResponseEndFrame so `assistant_aggregator` downstream in main.py
works unmodified.

(Earlier revisions' verified notes retained: `self._client` is the right
attribute on BaseOpenAILLMService; `settings=OpenAILLMService.Settings(
model=...)` avoids the DeprecationWarning that passing `model=` to the
concrete OpenAILLMService emits; deltas must go through the inherited
`self._push_llm_text()` so they're wrapped in LLMTextFrame, not plain
TextFrame, for correct inter-chunk spacing.)

FIXED IN THIS REVISION:

#1 - CONTEXT NO LONGER ACCUMULATES RAW AUDIO. Previously each native
turn appended its base64 WAV (~1.25MB for a 30s turn) into the shared
context.messages, which was then RE-SENT and RE-ENCODED by vLLM on every
subsequent request from either branch - request bodies and prefill cost
grew linearly with conversation length, i.e. the agent got measurably
slower every turn. Now: a text placeholder is appended immediately, and
a background (off-critical-path) transcription request replaces it in
place once the response is done. If transcription fails, the placeholder
stays - history remains valid text either way, which also resolves the
old mixed-modality hazard for Branch B.

#2 - THE TURN IS NOW INTERRUPTIBLE. The stream was previously awaited
inside process_frame with no cancellation point, so a user barge-in
(StartInterruptionFrame) could not stop generation: Gemma kept streaming
and kept feeding the TTS. The turn now runs as a managed task
(self.create_task) that _handle_interruption cancels - matching how the
base OpenAILLMService runs its own inference. Partial text generated
before the interruption is still recorded in history, since that's what
the user actually heard.

#3 - super().process_frame() is now called for EVERY frame, including
NativeAudioTurnFrame (FrameProcessor does required bookkeeping there;
it was being skipped for exactly the frame type we handle).

#4 - Errors go UPSTREAM via push_error() (so the PipelineTask sees
them), not downstream into the TTS via push_frame(ErrorFrame).

STILL GENUINELY UNVERIFIED (environment-specific):
1. Whether your installed vLLM build routes audio for THIS checkpoint -
   though your curl tests already returned real transcriptions, which is
   good evidence it does.
2. Whether the W4A16 quant preserved the audio projection layers.

KNOWN GAP (not a regression - was never wired): Branch A does not pass
`tools=` to the completion call, so web_search is unavailable on the
native-audio path. Wiring tool-call deltas through this custom stream is
a separate piece of work; flagging so short spoken questions that need
search don't silently get hallucinated answers.
"""

import base64
import io
import wave

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService

from ..router import NativeAudioTurnFrame

# What goes into shared history for the user's spoken turn until (unless)
# the background transcription replaces it.
AUDIO_TURN_PLACEHOLDER = "[spoken audio turn - transcript pending]"

TRANSCRIBE_PROMPT = (
    "Transcribe the audio verbatim. Output only the transcript text, "
    "with no preamble, labels, or quotation marks."
)


def _pcm_to_wav_b64(pcm_bytes: bytes, sample_rate: int, num_channels: int) -> str:
    """Wrap headerless PCM16 into a real WAV container, then base64 it.
    vLLM's audio decode path (soundfile/PyAV) needs an actual container -
    raw PCM under format="wav" will not decode as audio."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit, matches router.py's SAMPLE_WIDTH_BYTES
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class AudioNativeGemmaService(OpenAILLMService):
    """Branch A: whole-turn audio -> Gemma 4 12B's native audio input via vLLM."""

    def __init__(self, *, context, system_prompt: str, model: str, **kwargs):
        super().__init__(settings=OpenAILLMService.Settings(model=model), **kwargs)
        self._model = model
        self._context = context
        self._system_prompt = system_prompt
        self._turn_task = None  # in-flight native-audio turn, if any

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # FIX #3: base bookkeeping runs for every frame, ours included.
        await super().process_frame(frame, direction)
        if isinstance(frame, NativeAudioTurnFrame):
            # A new turn while one is somehow still running supersedes it.
            await self._cancel_turn_task()
            # FIX #2: run as a managed, cancellable task instead of
            # blocking process_frame for the whole stream.
            self._turn_task = self.create_task(self._run_native_audio_turn(frame))

    async def _cancel_turn_task(self):
        if self._turn_task is not None and not self._turn_task.done():
            await self.cancel_task(self._turn_task)
        self._turn_task = None

    async def _handle_interruption(self, frame, direction):
        # FIX #2: user barge-in must actually stop generation. The task's
        # finally-block records whatever partial text was already spoken.
        await self._cancel_turn_task()
        await super()._handle_interruption(frame, direction)

    async def _run_native_audio_turn(self, frame: NativeAudioTurnFrame):
        audio_b64 = _pcm_to_wav_b64(frame.audio, frame.sample_rate, frame.num_channels)

        user_content = [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
        ]

        # context.messages[0] already holds the system prompt from main.py.
        # Note: history is all-text (placeholders/transcripts, see FIX #1),
        # so this is the ONLY audio block in the request - request size no
        # longer grows with conversation length.
        messages = [
            *self._context.messages,
            {"role": "user", "content": user_content},
        ]

        # FIX #1: record the user turn as text immediately. Keep a direct
        # reference to the dict so the background transcription can replace
        # its content in place later, regardless of what else gets appended
        # to the list in the meantime.
        user_msg = {"role": "user", "content": AUDIO_TURN_PLACEHOLDER}
        self._context.messages.append(user_msg)

        await self.start_ttfb_metrics()
        await self.push_frame(LLMFullResponseStartFrame())

        full_text = ""
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
            )
            first_chunk = True
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if not delta:
                    continue
                if first_chunk:
                    await self.stop_ttfb_metrics()
                    first_chunk = False
                full_text += delta
                await self._push_llm_text(delta)

        except Exception as exc:
            logger.error(f"AudioNativeGemmaService: vLLM audio-in call failed: {exc}")
            # FIX #4: errors flow upstream to the PipelineTask, not into TTS.
            await self.push_error(ErrorFrame(f"vLLM audio-in call failed: {exc}"))
        finally:
            # Runs on success, error, AND cancellation (barge-in): close the
            # response frame pair and record whatever the user actually
            # heard, even if partial.
            await self.push_frame(LLMFullResponseEndFrame())
            if full_text:
                self._context.messages.append(
                    {"role": "assistant", "content": full_text}
                )
            # FIX #1 (part 2): transcribe off the critical path - this runs
            # between turns and mutates user_msg in place when done. Fire
            # and forget; if it fails, the placeholder simply remains.
            self.create_task(self._background_transcribe(audio_b64, user_msg))

    async def _background_transcribe(self, audio_b64: str, user_msg: dict):
        """Replace the placeholder user message with a real transcript.

        Off the critical path by design: this races nothing, blocks nothing,
        and its only side effect is mutating user_msg['content'] in place.
        Costs one short GPU inference between turns; if the next turn's
        request lands first, vLLM just queues it - acceptable trade for
        keeping history textual and request sizes flat.
        """
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": "wav"},
                            },
                            {"type": "text", "text": TRANSCRIBE_PROMPT},
                        ],
                    }
                ],
                stream=False,
                max_tokens=512,
                temperature=0.0,
            )
            transcript = (resp.choices[0].message.content or "").strip()
            if transcript:
                user_msg["content"] = transcript
        except Exception as exc:
            logger.warning(
                f"AudioNativeGemmaService: background transcription failed "
                f"(placeholder kept in history): {exc}"
            )
