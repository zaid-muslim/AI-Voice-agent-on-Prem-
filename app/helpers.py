"""
Three cross-cutting helpers:

1. run_with_filler() - the LiveKit-native port of tool_filler.py's
   @with_adaptive_filler. Same semantics: race the tool against a timeout,
   speak a filler ONLY if the call is genuinely slow, never on fast calls,
   and always wait for the real completion.

2. push_ui() - sends structured JSON to the browser frontend over LiveKit's
   data channel (topic "hospital.ui"). This is what makes the doctor
   directory and availability chips in frontend/index.html live: when the
   agent checks availability, the caller SEES the slots appear as the agent
   says them.

3. require_real_livekit_credentials() - fail-closed startup guard shared by
   main.py (the agent worker) and token_server.py (the join-token minter).
   LiveKit's own `--dev` mode auto-provisions the well-known "devkey"/
   "secret" pair - convenient for a laptop demo, but if a real deployment
   forgets to override them, the token server keeps minting (and
   livekit-server keeps accepting) tokens signed with publicly-documented
   credentials. Call this once at import time in anything that mints or
   relies on a token so a misconfigured deployment fails at startup, not
   silently in production.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Awaitable

from loguru import logger

from livekit.agents import get_job_context

UI_TOPIC = "hospital.ui"

_INSECURE_DEFAULTS = {"devkey": "secret"}


def require_real_livekit_credentials() -> tuple[str, str]:
    """Read LIVEKIT_API_KEY/LIVEKIT_API_SECRET from the environment and
    raise RuntimeError if either is unset or still matches LiveKit's
    well-known --dev default pair. Returns (key, secret) so callers don't
    need a second os.environ.get() round-trip."""
    key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set. Generate a "
            "real pair (`docker run --rm livekit/livekit-server "
            "generate-keys`, or `livekit-server generate-keys` bare-metal) "
            "and set both in app/.env - see the README's Deployment section."
        )
    if _INSECURE_DEFAULTS.get(key) == secret:
        raise RuntimeError(
            "LIVEKIT_API_KEY/LIVEKIT_API_SECRET are still LiveKit's "
            "well-known --dev default ('devkey'/'secret') - anyone who "
            "knows this public pair can mint their own room-join tokens "
            "against this deployment. Generate a real pair (`livekit-server "
            "generate-keys`) and set it in app/.env before running for "
            "real. (Fine to keep for a throwaway local demo ONLY.)"
        )
    return key, secret

# Speak a filler only if the tool hasn't returned within this window.
# Your Pipecat timing self-test proved this pattern: fast tool -> no filler;
# slow tool -> filler + wait for the real result.
FILLER_THRESHOLD_SECS = 0.7

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
