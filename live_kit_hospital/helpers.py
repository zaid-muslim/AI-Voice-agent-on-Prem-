"""
Two cross-cutting helpers:

1. run_with_filler() - the LiveKit-native port of tool_filler.py's
   @with_adaptive_filler. Same semantics: race the tool against a timeout,
   speak a filler ONLY if the call is genuinely slow, never on fast calls,
   and always wait for the real completion.

2. push_ui() - sends structured JSON to the browser frontend over LiveKit's
   data channel (topic "hospital.ui"). This is what makes the doctor
   directory and availability chips in frontend/index.html live: when the
   agent checks availability, the caller SEES the slots appear as the agent
   says them.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable

from loguru import logger

from livekit.agents import get_job_context

UI_TOPIC = "hospital.ui"

# Speak a filler only if the tool hasn't returned within this window.
# Your Pipecat timing self-test proved this pattern: fast tool -> no filler;
# slow tool -> filler + wait for the real result.
FILLER_THRESHOLD_SECS = 1.2

DEFAULT_FILLERS = {
    "check_availability": "One moment, let me check the schedule for you.",
    "book_appointment": "Just a second while I book that for you.",
    "cancel_appointment": "One moment while I pull up that appointment.",
    "update_appointment": "Let me update that for you, one moment.",
    "search_hospital_info": "Let me look that up for you.",
}


async def run_with_filler(
    session,
    awaitable: Awaitable[Any],
    *,
    filler: str,
    threshold: float = FILLER_THRESHOLD_SECS,
) -> Any:
    """Await `awaitable`; if it takes longer than `threshold`, speak `filler`
    (not added to chat context - it's presentation, not conversation) and
    keep waiting for the real result."""
    task = asyncio.ensure_future(awaitable)
    done, _pending = await asyncio.wait({task}, timeout=threshold)
    if not done:
        logger.info(f"run_with_filler: tool slow (> {threshold}s) - speaking filler")
        try:
            session.say(filler, add_to_chat_ctx=False)
        except TypeError:
            # Older livekit-agents: kwarg was named differently / absent.
            session.say(filler)
    return await task


async def push_ui(payload: dict) -> None:
    """Publish a UI event to every participant in the room. Non-fatal by
    design: a missing/closed room must never break the voice conversation."""
    try:
        room = get_job_context().room
        await room.local_participant.publish_data(
            json.dumps(payload).encode("utf-8"),
            reliable=True,
            topic=UI_TOPIC,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see above
        logger.debug(f"push_ui: skipped ({exc})")
