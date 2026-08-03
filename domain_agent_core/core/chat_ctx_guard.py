"""Chat-context truncation guard, ported from ``app/main.py:484-507,812``.

REAL BUG this defends against (found live in the original hospital
pipeline, not hypothetical): nothing bounded chat-history growth, so a
long-running call eventually crossed vLLM's ``--max-model-len`` ceiling and
every subsequent turn failed identically, forever. ``ChatContext.
truncate(max_items=N)`` is the framework's own sanctioned tool for this -
keeps the last N items, preserves the system/instructions message, never
leaves a dangling function_call without its output.

Made manifest-tunable (default unchanged from the original 16) since a
domain with much larger tool schemas shouldn't be forced to share one
hardcoded, platform-wide budget.
"""

from __future__ import annotations

DEFAULT_CHAT_CTX_MAX_ITEMS = 16


def truncate_chat_ctx(turn_ctx, max_items: int = DEFAULT_CHAT_CTX_MAX_ITEMS) -> None:
    """Truncate a turn's chat context to bound prompt-size growth.

    Args:
        turn_ctx: The ``llm.ChatContext`` passed into
            ``on_user_turn_completed``.
        max_items: How many recent items to keep (default 16, roughly
            5-6 exchanges - conservative headroom for a typical
            system-prompt + tool-schema token budget).
    """
    turn_ctx.truncate(max_items=max_items)
