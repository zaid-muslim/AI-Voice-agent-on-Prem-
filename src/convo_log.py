#!/usr/bin/env python3
"""Human-readable conversation log. Per turn it records: end-to-end latency, the caller's input,
the always-on retrieval (query + the exact chunks pulled, with similarity scores), any tools the
model invoked, and the agent's spoken reply.

Diagnostic intent: retrieval runs on every turn, so the RETRIEVAL block shows exactly what context
the model was grounded on. If the reply states a fact that isn't in the retrieved chunks, that's a
fabrication — visible at a glance by reading the chunks next to the reply. The TIMING line captures
how long the caller waited from their query until the agent's first spoken word.

Best-effort and side-effect-only: logging must never break a call, so the server wraps calls to
log_turn() in a try/except. Appends to logs/conversation.log (gitignored).
"""
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "conversation.log")

_CHUNK_PREVIEW_CHARS = 220


def _timing_str(timings: dict | None) -> str:
    if not timings:
        return ""
    parts = []
    ttft = timings.get("ttft_ms")
    total = timings.get("total_ms")
    retr = timings.get("retrieval_ms")
    parts.append(f"ttft {ttft:.0f}ms" if isinstance(ttft, (int, float)) else "ttft n/a")
    if isinstance(retr, (int, float)):
        parts.append(f"retrieval {retr:.1f}ms")
    if isinstance(total, (int, float)):
        parts.append(f"total {total/1000:.2f}s")
    return " | " + " | ".join(parts)


def _retrieval_lines(retrieval: dict | None) -> list[str]:
    if retrieval is None:
        return ["RETRIEVAL: (skipped)"]
    q = retrieval.get("query", "")
    status = retrieval.get("status")
    lines = [f"RETRIEVAL (query={q!r}) -> {status}"]
    for r in retrieval.get("results", []):
        score = r.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
        preview = " ".join((r.get("text") or "").split())[:_CHUNK_PREVIEW_CHARS]
        lines.append(f"    [{score_str}] {r.get('source')} :: {r.get('section')}")
        lines.append(f"           {preview}")
    return lines


def _tool_lines(tool_events: list[dict]) -> list[str]:
    if not tool_events:
        return []
    lines = ["TOOLS:"]
    for e in tool_events:
        tool = e.get("tool", "?")
        if tool == "web_search":
            lines.append(f"  web_search(query={e.get('query', '')!r})")
        elif tool == "remember":
            lines.append(f"  remember(fact={e.get('fact', '')!r})")
        elif tool == "block_card":
            lines.append(f"  block_card -> {e.get('status')}")   # never log the verification inputs
        elif tool == "request_human_handoff":
            lines.append(f"  request_human_handoff -> ticket {e.get('ticket_id')}")
        else:
            lines.append(f"  {tool}")
    return lines


def log_turn(session_id: str, user_text: str, retrieval: dict | None,
             tool_events: list[dict], reply: str, timings: dict | None = None) -> None:
    """Append one formatted turn block to the conversation log."""
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 88,
        f"{ts} | session {session_id}{_timing_str(timings)}",
        f"USER : {user_text}",
    ]
    lines.extend(_retrieval_lines(retrieval))
    lines.extend(_tool_lines(tool_events))
    lines.append(f"AGENT: {reply}" if reply else "AGENT: (no spoken reply)")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")
