"""Turn-filler mechanism, ported from ``app/main.py:401-476,703-748`` and
``app/helpers.py:95-132``. Generic race/pre-render/backstop logic - the
only thing generalized is that phrase content is now a constructor
argument (``DomainPack.filler``), not a module constant.

Two independent fillers exist here, same as the original:

1. ``TurnFiller`` - fires ``delay_secs`` after a user turn is confirmed if
   the agent hasn't started real reply audio yet (covers the ~2-3.5s
   tool-invoking-turn gap: two LLM round trips plus TTS).
2. ``run_with_filler()`` - races an individual TOOL call against a much
   shorter threshold, for the rare tool that's independently slow (a local
   SQLite/RAG lookup, normally a few ms).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from livekit import rtc
from loguru import logger

FILLER_THRESHOLD_SECS = 0.7


async def run_with_filler(
    session: Any,
    awaitable: Awaitable[Any],
    *,
    filler: str,
    threshold: float = FILLER_THRESHOLD_SECS,
) -> Any:
    """Await a tool call, speaking a filler phrase only if it runs slow.

    Args:
        session: The active ``AgentSession`` to speak the filler through.
        awaitable: The tool call to run.
        filler: The phrase to speak if `awaitable` is still running after
            `threshold` seconds.
        threshold: Seconds to wait before speaking `filler`.

    Returns:
        Whatever `awaitable` itself returns, once it completes.
    """
    task = asyncio.ensure_future(awaitable)
    done, _pending = await asyncio.wait({task}, timeout=threshold)
    if not done:
        logger.info(f"run_with_filler: tool slow (> {threshold}s) - speaking filler")
        try:
            session.say(filler, add_to_chat_ctx=False)
        except TypeError:
            session.say(filler)
    return await task


async def _single_frame_aiter(frame: rtc.AudioFrame):
    """Wraps one pre-rendered ``rtc.AudioFrame`` as the async iterable
    ``session.say(audio=...)`` expects."""
    yield frame


async def prerender_filler_audio(
    tts_service: Any, phrases: tuple[str, ...]
) -> dict[str, rtc.AudioFrame]:
    """Synthesize each turn-filler phrase once per call, with the actual
    TTS engine/voice this call is using, so a fired filler can play a
    real, already-in-memory frame instead of live-synthesizing (which
    would compete for the shared TTS engine's single-lane lock with the
    real reply's own synthesis).

    Args:
        tts_service: The active TTS plugin instance for this call.
        phrases: The domain's filler phrase set.

    Returns:
        ``{phrase: AudioFrame}`` for every phrase that synthesized
        successfully. Failure per-phrase is non-fatal - a missing entry
        falls back to live synthesis if that phrase fires.
    """
    cache: dict[str, rtc.AudioFrame] = {}
    for phrase in phrases:
        try:
            cache[phrase] = await tts_service.synthesize(phrase).collect()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"turn filler: pre-render failed for {phrase!r} ({exc}) - "
                "falling back to live synthesis if it fires."
            )
    return cache


class TurnFiller:
    """Per-call turn-level filler state - one instance per active call.

    Attributes:
        delay_secs: Seconds to wait after turn confirmation before
            firing, if no real reply audio has started yet.
        phrases: Round-robin phrase set.
        filler_audio: Pre-rendered ``{phrase: AudioFrame}`` cache, set via
            ``prerender_filler_audio()`` once the TTS engine is known.
    """

    def __init__(self, delay_secs: float, phrases: tuple[str, ...]) -> None:
        self.delay_secs = delay_secs
        self.phrases = phrases
        self.filler_audio: dict[str, rtc.AudioFrame] = {}
        self._task: asyncio.Task | None = None
        self._index = 0
        # Explicit backstop, independent of asyncio's cancellation-
        # delivery timing - see cancel()'s docstring for the real race
        # this closes.
        self._real_reply_started = False

    def cancel(self) -> None:
        """Call the moment real audio actually starts (whether that's
        this filler's own playback or the real reply beating it).
        Cancelling an already-fired/completed task is a harmless no-op,
        so this is safe to call unconditionally on every "speaking"
        transition. Also flips the explicit backstop flag - a filler
        task's only await point is its delay sleep; once that's
        genuinely elapsed, ``Task.cancel()`` can lose the race against
        real speech starting at nearly the same instant, so this flag is
        checked explicitly after the sleep too."""
        self._real_reply_started = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    def arm(self, session: Any) -> None:
        """Start a new filler timer for the current turn (replacing any
        stale prior-turn timer).

        Args:
            session: The active ``AgentSession`` to speak the filler
                through, if it fires.
        """
        self.cancel()
        self._real_reply_started = False
        self._task = asyncio.create_task(self._speak_after_delay(session))

    async def _speak_after_delay(self, session: Any) -> None:
        logger.debug(f"turn filler: armed, sleeping {self.delay_secs}s")
        try:
            await asyncio.sleep(self.delay_secs)
        except asyncio.CancelledError:
            logger.debug("turn filler: cancelled before firing (real reply beat it)")
            return
        if self._real_reply_started:
            logger.debug(
                "turn filler: sleep completed but real reply already started "
                "(cancel() lost the race) - not firing"
            )
            return
        phrase = self.phrases[self._index % len(self.phrases)]
        self._index += 1
        frame = self.filler_audio.get(phrase)
        logger.debug(
            f"turn filler: firing now - {phrase!r} "
            f"({'pre-rendered' if frame is not None else 'LIVE SYNTHESIS'})"
        )
        try:
            if frame is not None:
                session.say(phrase, audio=_single_frame_aiter(frame), add_to_chat_ctx=False)
            else:
                session.say(phrase, add_to_chat_ctx=False)
        except TypeError:
            session.say(phrase)
