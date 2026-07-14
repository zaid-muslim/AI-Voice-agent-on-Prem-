"""
Dual-path audio routing: commits each turn to either the audio-native
Gemma branch or the STT-fallback branch, the instant the 30s cap is
crossed (not after the turn ends).

FIXED IN THIS REVISION (from live-run log analysis):

#1 - TURN END = SMART-TURN-CONFIRMED ONLY. The transport runs a smart
turn analyzer, and the previous code ALSO treated every
VADUserStoppedSpeakingFrame (fires on any ~200ms pause) as end-of-turn.
Result seen in the logs: a 0.28s "turn" emitted 300ms into the user's
sentence WHILE smart turn was reporting INCOMPLETE -> Gemma answered a
fragment, the bot started talking over the user, and the real turn got
answered a second time (the duplicated text_len=62/25 responses). Now
the router emits/resets ONLY on UserStartedSpeakingFrame /
UserStoppedSpeakingFrame - the same smart-turn-confirmed signals
LLMUserAggregator keys on - and passes raw VAD frames through untouched.

#2 - OPTIONAL MIC GATE WHILE THE BOT SPEAKS (echo-loop breaker).
LocalAudioTransport has no echo cancellation; the logs show the classic
feedback loop (metronomic text_len=20/25 responses, "user turns"
captured while only the bot was talking): speaker -> mic -> VAD -> new
turn -> answer -> speaker... With gate_mic_while_bot_speaks=True the
router drops input audio between BotStartedSpeakingFrame and
BotStoppedSpeakingFrame. Trade-off: barge-in is disabled while gated -
but with open speakers and no AEC, "barge-in" was already just echo.
Use headphones + gate=False when you want real interruptions.

(Prior fix retained: AudioRawFrame is evaluated BEFORE the generic
SystemFrame fast-track, because InputAudioRawFrame IS a SystemFrame in
current Pipecat - the original silent-buffer bug.)
"""

import time
from dataclasses import dataclass

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    SystemFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    StartFrame,
    CancelFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger

# --- tuning ---------------------------------------------------------------

AUDIO_NATIVE_CAP_SECONDS = 30.0
SAFETY_MARGIN_SECONDS = 1.0
COMMIT_THRESHOLD = AUDIO_NATIVE_CAP_SECONDS - SAFETY_MARGIN_SECONDS
# Keep the mic gate closed briefly after the bot stops: the echo tail of
# its last words is still in flight through speakers/room/ALSA buffers.
POST_SPEECH_GRACE_SECS = 0.4
SAMPLE_WIDTH_BYTES = 2
SAMPLE_RATE = 16000
NUM_CHANNELS = 1


# --- tagged frames ----------------------------------------------------------


@dataclass
class NativeAudioTurnFrame(DataFrame, AudioRawFrame):
    """Whole-turn audio, emitted only when the turn stayed under the cap."""

    def __post_init__(self):
        super().__post_init__()
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))


@dataclass
class STTTriggerFrame(DataFrame, AudioRawFrame):
    """Audio chunk routed to STT because this turn committed to that path."""

    def __post_init__(self):
        super().__post_init__()
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))


class TurnAudioRouter(FrameProcessor):
    """Buffers audio for the current turn and decides which branch gets fed."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        num_channels: int = NUM_CHANNELS,
        gate_mic_while_bot_speaks: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._bytes_per_second = sample_rate * num_channels * SAMPLE_WIDTH_BYTES
        self._buffer = bytearray()
        self._committed_to_stt = False
        self._is_started = False
        self._gate_mic_while_bot_speaks = gate_mic_while_bot_speaks
        self._bot_speaking = False
        self._gate_open_after = 0.0  # monotonic time; grace after bot stops
        self._suppress_turn = False  # bot spoke over an in-progress turn

    def _reset_turn(self):
        self._buffer = bytearray()
        self._committed_to_stt = False

    def _mic_gated(self) -> bool:
        if not self._gate_mic_while_bot_speaks:
            return False
        return self._bot_speaking or time.monotonic() < self._gate_open_after

    async def process_frame(self, frame: Frame, direction: FrameDirection):

        # 0. BOT SPEAKING STATE (frames flow upstream from output transport;
        #    track them in whatever direction they arrive).
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            # If a user turn was mid-capture when the bot started talking,
            # the rest of that turn is about to be gated away - emitting the
            # leading fragment later would make Gemma answer half a
            # sentence (observed: 1.44s/1.32s decapitated turns). Discard
            # it and suppress the eventual stop-frame emission instead.
            if len(self._buffer) > 0:
                logger.debug(
                    f"TurnAudioRouter: bot started speaking mid-turn; "
                    f"discarding {len(self._buffer)} buffered bytes and "
                    f"suppressing this turn's emission"
                )
                self._suppress_turn = True
                self._reset_turn()
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._gate_open_after = time.monotonic() + POST_SPEECH_GRACE_SECS
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # 0.5. ECHO SHIELD FOR THE TURN MACHINERY. VAD runs in the input
        #    transport, UPSTREAM of this router - so the buffer gate alone
        #    couldn't stop echo from triggering user-start events,
        #    interruption broadcasts, and phantom turns (observed: bot cut
        #    off mid-sentence by its own voice). While gated, swallow the
        #    raw VAD frames so the turn analyzer/aggregator never hear the
        #    echo at all. Trade-off: NO barge-in while gated - with open
        #    speakers and no AEC, barge-in was echo anyway. For real
        #    barge-in: headphones + gate_mic_while_bot_speaks=False.
        if isinstance(
            frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
        ):
            if self._mic_gated():
                return
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # 1. TURN BOUNDARIES - react ONLY to smart-turn-confirmed frames.
        if isinstance(frame, UserStartedSpeakingFrame):
            self._suppress_turn = False
            self._reset_turn()
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            if self._suppress_turn:
                logger.debug(
                    "TurnAudioRouter: turn stop for a suppressed turn "
                    "(bot spoke over it); not emitting"
                )
                self._suppress_turn = False
                self._reset_turn()
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
                return
            if not self._committed_to_stt and len(self._buffer) > 0:
                buf_len_sec = len(self._buffer) / self._bytes_per_second
                logger.debug(
                    f"TurnAudioRouter: emitting NativeAudioTurnFrame "
                    f"({len(self._buffer)} bytes, {buf_len_sec:.3f}s)"
                )
                # ALWAYS downstream. UserStoppedSpeakingFrame originates at
                # LLMUserAggregator (which is DOWNSTREAM of us, in Branch B)
                # and reaches this router travelling UPSTREAM - so pushing
                # with `direction` sent the entire turn's audio backwards
                # into the input transport, where it silently vanished.
                # (This is why full turns were never answered while
                # VAD-fragment turns - triggered by frames travelling
                # downstream from the transport - were.)
                await self.push_frame(
                    NativeAudioTurnFrame(
                        audio=bytes(self._buffer),
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                    ),
                    FrameDirection.DOWNSTREAM,
                )
            self._reset_turn()
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # 2. AUDIO INTERCEPTION - must come before the SystemFrame
        #    fast-track (InputAudioRawFrame IS a SystemFrame).
        if isinstance(frame, AudioRawFrame) and not isinstance(
            frame, (NativeAudioTurnFrame, STTTriggerFrame)
        ):
            if not self._is_started:
                return  # shield against pre-StartFrame audio

            # Echo-loop breaker (buffer side): while the bot is speaking
            # or within the post-speech grace window, input "speech" is
            # overwhelmingly our own output re-entering the mic. Drop it.
            if self._mic_gated():
                return

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

            if frame.audio:
                self._buffer.extend(frame.audio)
            buffered_seconds = len(self._buffer) / self._bytes_per_second

            if buffered_seconds >= COMMIT_THRESHOLD:
                self._committed_to_stt = True
                flushed, self._buffer = bytes(self._buffer), bytearray()
                logger.debug(
                    f"TurnAudioRouter: committed to STT at {buffered_seconds:.3f}s, "
                    f"flushing {len(flushed)} bytes"
                )
                await self.push_frame(
                    STTTriggerFrame(
                        audio=flushed,
                        sample_rate=self._sample_rate,
                        num_channels=self._num_channels,
                    ),
                    direction,
                )
            return

        # 3. SYSTEM FRAME FAST-TRACK (includes raw VAD start/stop frames,
        #    which downstream consumers may still want - just not us).
        if isinstance(frame, SystemFrame):
            if isinstance(frame, StartFrame):
                self._is_started = True
            elif isinstance(frame, CancelFrame):
                self._is_started = False
                self._reset_turn()
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # 4. SHIELD for any remaining data frames before start
        if not self._is_started:
            return

        # 5. CATCH-ALL: pass control/context/text frames through
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class OnlyPass(FrameProcessor):
    """Branch gate: lets specific audio frames (and, if explicitly allowed,
    the turn-completion signal) pass, swallows the rest."""

    def __init__(self, allowed_types, **kwargs):
        super().__init__(**kwargs)
        self._allowed = tuple(allowed_types)
        self._is_started = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):

        # Turn-stop frames are SystemFrame subclasses: gate them explicitly
        # so they only reach the branch that asked for them.
        if isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            if type(frame) in self._allowed:
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
            return

        # AUDIO GATE - before the SystemFrame fast-track, because
        # InputAudioRawFrame is a SystemFrame.
        if isinstance(frame, AudioRawFrame):
            if not self._is_started:
                return
            if isinstance(frame, self._allowed):
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
            return

        # Fast-track remaining system frames (Start/Cancel/interruptions...)
        if isinstance(frame, SystemFrame):
            if isinstance(frame, StartFrame):
                self._is_started = True
            elif isinstance(frame, CancelFrame):
                self._is_started = False
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # Shield against early data frames
        if not self._is_started:
            return

        # Pass all text, context, and control frames natively
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
