"""
latency_log.py - records per-turn latency, tagged by which LLM/STT/TTS
combination was active, so different combinations can be compared side by
side in the dev console instead of guessed at.

WHAT GETS LOGGED (one line per LLM turn, appended to latency_log.jsonl):
    {
        "timestamp": "2026-07-19T16:30:00.000Z",
        "combo": "llm=gemma-4-12b|stt=whisper:distil-large-v3|tts=qwen:aiden",
        "llm_served_model_name": "gemma-4-12b",
        "stt_engine": "whisper", "stt_model": "distil-large-v3",
        "tts_engine": "qwen", "tts_model": "aiden",
        "end_of_utterance_delay": 0.53,
        "transcription_delay": 0.52,
        "llm_ttft": 0.33,
        "tts_ttfb": 0.33
    }

WHY A FLAT .jsonl FILE, NOT A DATABASE: this is genuinely simple,
append-only, write-heavy, read-rarely data - exactly what a database adds
the least value for. Appending a line is a single atomic write syscall (no
lock contention risk like a shared SQLite writer under concurrent calls -
recall booking.py's whole reason for using SQLite was the OPPOSITE need,
real transactional integrity for double-booking prevention; this data has
no such requirement, it's just a log). Reading/aggregating happens
on-demand in the dev UI, not on any latency-critical path.

HOW THIS CONNECTS TO agent.py: entrypoint() already listens for
"metrics_collected" events (see agent.py's _on_metrics handler) for
console logging. This module adds a SECOND listener that extracts the
same fields and appends a tagged record - see agent.py's entrypoint() for
the actual wiring.

WHY EOU/TTFT/TTFB SPECIFICALLY: these are exactly the fields this whole
project's manual latency investigation already relied on (the
whisper-vs-Parakeet comparison earlier this week compared
transcription_delay by hand, reading it out of raw logs one call at a
time). This module makes that comparison automatic and cumulative across
many calls instead of a one-off manual read.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_FILE = Path(__file__).parent / "latency_log.jsonl"

# Simple size-based rotation: once the log exceeds this, the current file
# becomes the single ".1" backup before the new line is appended. Latency
# records are a few hundred bytes each, so 20MB is tens of thousands of
# turns - plenty for the dev console's comparisons while capping how much
# disk a long-running production deployment loses to a log nobody rotates
# otherwise (see the module docstring's original honest note on this).
_MAX_LOG_BYTES = 20 * 1024 * 1024

# Only these metric TYPES are recorded - matches exactly what agent.py's
# metrics_collected handler already sees from LiveKit's own EOU/LLM/TTS
# metrics events (see agent.py entrypoint()'s _on_metrics for the mapping).
_TRACKED_FIELDS = (
    "end_of_utterance_delay",
    "transcription_delay",
    "llm_ttft",
    "tts_ttfb",
)


def record(combo: dict, metrics_fields: dict[str, float]) -> None:
    """Append one latency record. `combo` is the config dict shape from
    system_config.get_config() (llm/stt/tts sections); `metrics_fields` is
    whichever of _TRACKED_FIELDS this particular event carried (LiveKit
    emits EOU, LLM, and TTS metrics as SEPARATE events per turn, not one
    combined event - callers accumulate fields across a turn and call this
    once per completed turn; see agent.py for exactly how).

    Never raises - a logging failure must not break a real call. Any
    error is silently swallowed; this is diagnostic data, not
    load-bearing."""
    try:
        import system_config  # local import: avoids a hard dependency for
        # any caller that only wants the read-side (get_summary) below

        cfg = combo
        record_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "combo": system_config.combo_key(cfg),
            "llm_served_model_name": cfg["llm"]["served_model_name"],
            "stt_engine": cfg["stt"]["engine"],
            "stt_model": cfg["stt"]["model"],
            "tts_engine": cfg["tts"]["engine"],
            "tts_model": cfg["tts"]["model"],
        }
        for field in _TRACKED_FIELDS:
            if field in metrics_fields:
                record_obj[field] = metrics_fields[field]

        _rotate_if_oversized()
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_obj) + "\n")
    except Exception:  # noqa: BLE001
        pass  # diagnostic logging must never break a real call


def _rotate_if_oversized() -> None:
    """Renames latency_log.jsonl -> latency_log.jsonl.1 (overwriting any
    previous backup) once the live file crosses _MAX_LOG_BYTES. Checked
    right before each append rather than on a timer - this file is only
    ever written from here, so a size check at write time is sufficient
    and avoids a background task for something this low-stakes."""
    try:
        if _LOG_FILE.exists() and _LOG_FILE.stat().st_size >= _MAX_LOG_BYTES:
            backup = _LOG_FILE.with_suffix(_LOG_FILE.suffix + ".1")
            _LOG_FILE.replace(backup)
    except OSError:
        pass  # rotation failing must not block logging or break a real call


def _read_all() -> list[dict]:
    if not _LOG_FILE.exists():
        return []
    records = []
    with open(_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupt line rather than fail the whole read
    return records


def get_summary(limit_per_combo: int = 500) -> list[dict]:
    """Aggregate the log into one row per COMBO, with mean/median/count for
    each tracked field, sorted by combo name for stable display. Each
    combo's own most-recent `limit_per_combo` records are used, so one
    combination tested heavily doesn't drown out a lightly-tested one in
    memory usage (this file only grows, nothing rotates it automatically -
    see the module docstring's honest note on that below)."""
    records = _read_all()
    by_combo: dict[str, list[dict]] = {}
    for r in records:
        by_combo.setdefault(r.get("combo", "unknown"), []).append(r)

    summary = []
    for combo, recs in by_combo.items():
        recs = recs[-limit_per_combo:]
        row: dict[str, Any] = {
            "combo": combo,
            "sample_count": len(recs),
            "llm_served_model_name": recs[-1].get("llm_served_model_name"),
            "stt_engine": recs[-1].get("stt_engine"),
            "stt_model": recs[-1].get("stt_model"),
            "tts_engine": recs[-1].get("tts_engine"),
            "tts_model": recs[-1].get("tts_model"),
        }
        for field in _TRACKED_FIELDS:
            values = [r[field] for r in recs if field in r]
            if values:
                row[f"{field}_mean"] = round(statistics.mean(values), 3)
                row[f"{field}_median"] = round(statistics.median(values), 3)
                row[f"{field}_n"] = len(values)
        summary.append(row)

    summary.sort(key=lambda r: r["combo"])
    return summary


def clear() -> None:
    """Wipe the log (and any rotated backup). Exposed for the dev UI's
    'reset latency data' action - useful when you've been testing/tuning
    and want a clean slate before a real comparison run."""
    if _LOG_FILE.exists():
        _LOG_FILE.unlink()
    backup = _LOG_FILE.with_suffix(_LOG_FILE.suffix + ".1")
    if backup.exists():
        backup.unlink()


# Rotation: see _rotate_if_oversized() above - the live file is capped at
# _MAX_LOG_BYTES with a single ".1" backup, checked on every append.
# get_summary()'s _read_all() only reads the live file, not the rotated
# backup - a rotation drops old records from the dev console's aggregates,
# which is the intended tradeoff (bounded disk over unbounded history).


if __name__ == "__main__":
    # Quick manual inspection: python latency_log.py
    summary = get_summary()
    if not summary:
        print("No latency data recorded yet.")
    else:
        for row in summary:
            print(f"\n{row['combo']}  (n={row['sample_count']})")
            for field in _TRACKED_FIELDS:
                if f"{field}_mean" in row:
                    print(
                        f"  {field}: mean={row[f'{field}_mean']}s "
                        f"median={row[f'{field}_median']}s n={row[f'{field}_n']}"
                    )
