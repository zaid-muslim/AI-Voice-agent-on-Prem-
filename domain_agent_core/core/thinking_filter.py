"""Thinking-channel leak filter, ported from
``app/main.py:510-575`` (``_ThinkingChannelStreamFilter``).

Tied to a specific LLM model's quirk (gemma-4-12b sometimes opens a
``<|channel>thought...`` reasoning block on its own initiative, even with
thinking disabled in the chat template), not to any domain - gated by
``DomainPack.strip_thinking_channel``, which a pack sets based on which
model its ``engines.llm`` selects, not based on domain identity.

Streaming-safe: content arrives one token/fragment at a time, not as one
final string.
"""

from __future__ import annotations


class ThinkingChannelStreamFilter:
    """Strips a leaked ``<|channel>thought...<channel|>`` block from a
    streamed completion, one fragment at a time.

    Attributes:
        state: One of "checking", "stripping", "clean".
    """

    _OPEN = "<|channel>"
    _CLOSE = "<channel|>"

    def __init__(self) -> None:
        self.state = "checking"
        self._buffer = ""

    def push(self, content: str) -> str:
        """Feed the next content fragment through the filter.

        Args:
            content: The next raw text fragment from the LLM stream.

        Returns:
            The visible portion of ``content`` (may be empty while still
            buffering inside a leaked thought block).
        """
        if self.state == "clean":
            return content

        self._buffer += content

        if self.state == "stripping":
            idx = self._buffer.find(self._CLOSE)
            if idx == -1:
                return ""
            self.state = "clean"
            rest = self._buffer[idx + len(self._CLOSE) :]
            self._buffer = ""
            return rest

        if self._OPEN in self._buffer:
            self.state = "stripping"
            self._buffer = self._buffer.split(self._OPEN, 1)[1]
            idx = self._buffer.find(self._CLOSE)
            if idx == -1:
                return ""
            self.state = "clean"
            rest = self._buffer[idx + len(self._CLOSE) :]
            self._buffer = ""
            return rest

        if len(self._buffer) < len(self._OPEN) and self._OPEN.startswith(self._buffer):
            return ""

        self.state = "clean"
        out = self._buffer
        self._buffer = ""
        return out
