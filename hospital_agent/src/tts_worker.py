"""
Dual-path audio routing: commits each turn to either the audio-native
Gemma branch or the STT-fallback branch, the instant the 30s cap is
crossed (not after the turn ends).
"""

from dataclasses import dataclass

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# --- tuning ---------------------------------------------------------------

AUDIO_NATIVE_CAP_SECONDS = 30.0
SAFETY_MARGIN_SECONDS = 1.0
COMMIT_THRESHOLD = AUDIO_NATIVE_CAP_SECONDS - SAFETY_MARGIN_SECONDS
SAMPLE_WIDTH_BYTES = 2
SAMPLE_RATE = 16000
NUM_CHANNELS = 1


# --- tagged frames ----------------------------------------------------------


@dataclass
class NativeAudioTurnFrame(AudioRawFrame):
    """Whole-turn audio, emitted only when the turn stayed under the cap."""


@dataclass
class STTTriggerFrame(AudioRawFrame):
    """Audio chunk routed to STT because this turn committed to that path."""


class TurnAudioRouter(FrameProcessor):
    """Buffers audio for the current turn and decides which branch gets fed."""

    def __init__(
        self, sample_rate: int = SAMPLE_RATE, num_channels: int = NUM_CHANNELS, **kwargs
    ):
        super().__init__(**kwargs)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._bytes_per_second = sample_rate * num_channels * SAMPLE_WIDTH_BYTES
        self._buffer = bytearray()
        self._committed_to_stt = False

    def _reset_turn(self):
        self._buffer = bytearray()
        self._committed_to_stt = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):

        # 1. INTERCEPT AND BUFFER RAW AUDIO
        # We do NOT call super() here, which safely swallows the frame so it doesn't leak.
        if isinstance(frame, AudioRawFrame) and not isinstance(
            frame, (NativeAudioTurnFrame, STTTriggerFrame)
        ):
            if self._committed_to_stt:
                await self.push_frame(
                    STTTriggerFrame(
                        audio=frame.audio,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                    ),
                    direction,
                )
                return

            self._buffer.extend(frame.audio)
            buffered_seconds = len(self._buffer) / self._bytes_per_second

            if buffered_seconds >= COMMIT_THRESHOLD:
                self._committed_to_stt = True
                flushed, self._buffer = bytes(self._buffer), bytearray()
                await self.push_frame(
                    STTTriggerFrame(
                        audio=flushed,
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                    ),
                    direction,
                )
            return

        # 2. HANDLE USER SPEECH START
        if isinstance(frame, UserStartedSpeakingFrame):
            self._reset_turn()
            await super().process_frame(frame, direction)
            return

        # 3. HANDLE USER SPEECH END
        if isinstance(frame, UserStoppedSpeakingFrame):
            if not self._committed_to_stt and self._buffer:
                await self.push_frame(
                    NativeAudioTurnFrame(
                        audio=bytes(self._buffer),
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                    ),
                    direction,
                )
            self._reset_turn()
            await super().process_frame(frame, direction)
            return

        # 4. CATCH-ALL FOR START, CANCEL, AND CONTROL FRAMES
        # Passing these to super() guarantees Pipecat's internal _started flags switch correctly.
        await super().process_frame(frame, direction)


class OnlyPass(FrameProcessor):
    """Branch gate: swallows audio meant for the other branch, lets everything else through."""

    def __init__(self, allowed_types, **kwargs):
        super().__init__(**kwargs)
        self._allowed = tuple(allowed_types)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # If it is an Audio frame, ONLY let it pass if it matches the allowed branch type.
        if isinstance(frame, AudioRawFrame):
            if isinstance(frame, self._allowed):
                await super().process_frame(frame, direction)
            return  # Swallow blocked audio frames

        # Pass all other system, text, and control frames safely down the pipeline.
        await super().process_frame(frame, direction)
