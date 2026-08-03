"""Per-pack engine selection store, ported from ``app/system_config.py``'s
schema-v2 pattern (validated JSON, shallow-merged over defaults, atomic
write, "takes effect on next call" contract - see that module's docstring
for the full original rationale).

GENERALIZED FROM THE ORIGINAL: one file per domain pack
(``<pack_dir>/engine_config.json``), not one shared file - two domains
running concurrently must be able to pick different active LLM/STT/TTS
combos independently. The default config is seeded from the pack's
manifest (``DomainPack.engines``) rather than a hardcoded module constant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_VALID_TTS_ENGINES = {"qwen", "qwen_shared", "chatterbox", "kokoro", "piper"}
_VALID_STT_ENGINES = {"whisper_shared", "whisper", "parakeet", "canary"}


def config_file_path(pack_dir: Path) -> Path:
    """Build the on-disk path for a pack's engine-config file.

    Args:
        pack_dir: The domain pack's own directory.

    Returns:
        Path to ``<pack_dir>/engine_config.json``.
    """
    return pack_dir / "engine_config.json"


def get_engine_config(
    config_file: Path, defaults: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Read the current llm/stt/tts selection for one domain pack.

    Reads fresh from disk every call - deliberately no caching, same
    contract as ``system_config.get_config()``. Falls back to ``defaults``
    (seeded from the pack's manifest) if the file is missing or corrupt.

    Args:
        config_file: Result of ``config_file_path()``.
        defaults: The pack's manifest-declared engine selections, shaped
            like ``{"llm": {...}, "stt": {...}, "tts": {...}}``.

    Returns:
        A dict with ``llm``, ``stt``, ``tts`` keys, each shallow-merged
        over ``defaults`` so a partially-written file never crashes a
        caller.
    """
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            merged = {}
            for key in ("llm", "stt", "tts"):
                merged[key] = {**defaults.get(key, {}), **cfg.get(key, {})}
            return merged
        except (OSError, ValueError):
            pass
    return {k: dict(v) for k, v in defaults.items()}


def save_engine_config(config_file: Path, new_cfg: dict[str, Any]) -> None:
    """Validate and atomically write a new llm/stt/tts config.

    Args:
        config_file: Result of ``config_file_path()``.
        new_cfg: A dict shaped like ``get_engine_config()``'s return
            value.

    Raises:
        ValueError: If any required field is missing or an engine isn't
            in the known-valid set - lists every problem at once, same
            convention as ``system_config.save_config()``.
    """
    problems: list[str] = []

    llm = new_cfg.get("llm", {})
    if not llm.get("served_model_name") and not llm.get("engine"):
        problems.append("llm.served_model_name (or llm.engine) is required")
    if not llm.get("source") and not llm.get("model"):
        problems.append("llm.source (or llm.model) is required")

    stt = new_cfg.get("stt", {})
    if stt.get("engine") not in _VALID_STT_ENGINES:
        problems.append(
            f"stt.engine must be one of {sorted(_VALID_STT_ENGINES)}, got "
            f"{stt.get('engine')!r}"
        )
    if not stt.get("model"):
        problems.append("stt.model is required")

    tts = new_cfg.get("tts", {})
    if tts.get("engine") not in _VALID_TTS_ENGINES:
        problems.append(
            f"tts.engine must be one of {sorted(_VALID_TTS_ENGINES)}, got "
            f"{tts.get('engine')!r}"
        )
    if not tts.get("model"):
        problems.append("tts.model is required")

    if problems:
        raise ValueError("Invalid engine config:\n  - " + "\n  - ".join(problems))

    config_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_file.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2)
    tmp.replace(config_file)
