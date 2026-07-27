"""
system_config.py - the shared settings store for "which model is active
right now" (LLM, TTS, STT) - SCHEMA V2.

WHY THIS CHANGED FROM A FLAT STRING: the original version stored
"tts_backend": "qwen" - one string, one choice per engine. That stopped
being enough the moment the requirement became "4-5 variants EACH of
LLM/STT/TTS" - e.g. two different Whisper sizes both use the SAME plugin
code (plugins/whisper_stt.py) but need a DIFFERENT model name passed to
it. A flat string can't express "which engine" separately from "which
specific model/checkpoint for that engine." So each of llm/stt/tts is now
a small object: {"engine": ..., "model": ..., "display_name": ...}.
  - "engine" picks WHICH PLUGIN CLASS agent.py uses (e.g. "whisper",
    "parakeet", "canary" for STT) - this maps directly to a branch in
    agent.py's _make_stt()/_make_tts().
  - "model" is whatever that engine's plugin needs to load the SPECIFIC
    variant (a model name, a repo id, a local path, a voice id - the
    exact meaning is engine-specific, same as it always was).

THE "TAKES EFFECT ON NEXT CALL" CONTRACT is unchanged from v1 - see the
original design note preserved below.

FILE FORMAT (system_config.json):
{
  "llm": {"served_model_name": "gemma-4-12b",
          "source": "/path/to/model", "display_name": "Gemma 4 12B (w4a16)"},
  "stt": {"engine": "whisper", "model": "distil-large-v3",
          "display_name": "Whisper distil-large-v3"},
  "tts": {"engine": "qwen", "model": "aiden",
          "display_name": "Qwen TTS (aiden voice)"}
}

ORIGINAL DESIGN NOTE (still true): get_config() ALWAYS reads fresh from
disk. No caching. Cheap: it's one small JSON file. agent.py calls this
once per call, at the start of entrypoint(), so a call already in
progress is never affected mid-call, but the NEXT call always sees the
latest choice. LLM switching has an extra real-world step beyond just
"read the new value": vLLM must actually be RUNNING that model before
agent.py can use it - see vllm_manager.py. TTS/STT switches are cheaper
(no shared server to restart) and take effect purely by this file
changing, modulo the cross-process worker-pool caveat documented in
agent.py's prewarm().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONFIG_FILE = Path(__file__).parent / "system_config.json"

_DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "served_model_name": "gemma-4-12b",
        "source": "/home/nauyan/voice-agent-pipeline/models/gemma-4-12b-w4a16",
        "display_name": "Gemma 4 12B (w4a16)",
    },
    "stt": {
        "engine": "whisper_shared",
        "model": "distil-large-v3",
        "display_name": "Whisper distil-large-v3 (shared service, default)",
    },
    "tts": {
        "engine": "qwen",
        "model": "aiden",
        "display_name": "Qwen TTS (aiden voice, proven default)",
    },
}

_VALID_TTS_ENGINES = {"qwen", "chatterbox", "kokoro", "piper"}
_VALID_STT_ENGINES = {"whisper_shared", "whisper", "parakeet", "canary"}


def get_config() -> dict:
    """Read fresh from disk every call - deliberately no caching, see
    module docstring. Falls back to defaults if the file is missing or
    corrupt, so the system never fails to start because of a bad config
    file - it just falls back to a known-good default instead."""
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # shallow-merge each section over defaults so a partially-
            # written/older file (e.g. missing a newly-added key) doesn't
            # crash callers
            merged = {}
            for key in ("llm", "stt", "tts"):
                merged[key] = {**_DEFAULT_CONFIG[key], **cfg.get(key, {})}
            return merged
        except (OSError, ValueError):
            pass
    return {k: dict(v) for k, v in _DEFAULT_CONFIG.items()}


def save_config(new_cfg: dict) -> None:
    """Validate and atomically write a new config. Raises ValueError with a
    human-readable message on anything invalid, so the dev UI can show the
    developer exactly what to fix."""
    problems = []

    llm = new_cfg.get("llm", {})
    if not llm.get("served_model_name"):
        problems.append("llm.served_model_name is required")
    if not llm.get("source"):
        problems.append("llm.source is required (a local path or HF repo id)")

    stt = new_cfg.get("stt", {})
    if stt.get("engine") not in _VALID_STT_ENGINES:
        problems.append(
            f"stt.engine must be one of {sorted(_VALID_STT_ENGINES)}, got {stt.get('engine')!r}"
        )
    if not stt.get("model"):
        problems.append(
            "stt.model is required (a specific model name/path for the chosen engine)"
        )

    tts = new_cfg.get("tts", {})
    if tts.get("engine") not in _VALID_TTS_ENGINES:
        problems.append(
            f"tts.engine must be one of {sorted(_VALID_TTS_ENGINES)}, got {tts.get('engine')!r}"
        )
    if not tts.get("model"):
        problems.append(
            "tts.model is required (a specific voice/model name for the chosen engine)"
        )

    if problems:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(problems))

    tmp = _CONFIG_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2)
    tmp.replace(_CONFIG_FILE)  # atomic swap, same pattern as hospital_kb.save_kb


def combo_key(cfg: dict | None = None) -> str:
    """A short, stable string identifying the CURRENT (or given) llm+stt+tts
    combination - used to tag latency log entries so different
    combinations can be compared later. Deliberately just the engine/model
    identifiers, not the display names, so it stays stable even if display
    text is edited."""
    cfg = cfg or get_config()
    return (
        f"llm={cfg['llm']['served_model_name']}"
        f"|stt={cfg['stt']['engine']}:{cfg['stt']['model']}"
        f"|tts={cfg['tts']['engine']}:{cfg['tts']['model']}"
    )


def load_registry() -> dict:
    """Load the curated list of known-good/candidate models and STT/TTS
    engine+model combinations the dev UI offers as choices. See
    models_registry.json's own comments for what "verified" vs
    "candidate" means, and the real license findings noted per entry."""
    registry_file = Path(__file__).parent / "models_registry.json"
    with open(registry_file, "r", encoding="utf-8") as f:
        return json.load(f)
