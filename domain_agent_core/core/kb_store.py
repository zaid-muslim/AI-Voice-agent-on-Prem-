"""Generic knowledge-base storage, generalized from
``app/hospital_core/hospital_kb.py``'s atomic-write pattern.

Each domain's KB is a JSON file of entries (``{"id", "category", "title",
"text"}``) at ``<pack_dir>/kb/<domain_id>_kb.json``. Loading falls back to
a pack-supplied default list if the file is missing, so a domain never
starts with no data. Saving does an atomic tmp-file-replace write, same
pattern as ``hospital_kb.save_kb()``/``system_config.save_config()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MIN_TEXT_LEN = 20
_MAX_TEXT_LEN = 400
_REQUIRED_FIELDS = ("id", "category", "title", "text")


def kb_file_path(kb_directory: Path, domain_id: str) -> Path:
    """Build the on-disk path for a domain's KB file.

    Args:
        kb_directory: The pack's ``kb/`` directory.
        domain_id: The domain identifier (e.g. "banking").

    Returns:
        Path to ``<kb_directory>/<domain_id>_kb.json``.
    """
    return kb_directory / f"{domain_id}_kb.json"


def load_kb(path: Path, default_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load a domain's KB entries from disk, falling back to defaults.

    Args:
        path: Result of ``kb_file_path()``.
        default_entries: Used verbatim if ``path`` doesn't exist or is
            corrupt - so the system never starts with no data.

    Returns:
        The loaded (or default) list of KB entries.
    """
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return list(default_entries)


def validate_entries(entries: list[dict[str, Any]]) -> list[str]:
    """Structural validation, mirroring ``hospital_kb._validate_entries``.

    Args:
        entries: KB entries to check.

    Returns:
        A list of problems - missing fields, duplicate ids, or
        out-of-range text length. Empty if the entries are valid.
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        missing = [field for field in _REQUIRED_FIELDS if not entry.get(field)]
        if missing:
            problems.append(f"entry[{i}] missing required field(s): {missing}")
            continue
        if entry["id"] in seen_ids:
            problems.append(f"entry[{i}] has duplicate id {entry['id']!r}")
        seen_ids.add(entry["id"])
        text_len = len(entry["text"])
        if not (_MIN_TEXT_LEN <= text_len <= _MAX_TEXT_LEN):
            problems.append(
                f"entry[{i}] ({entry['id']!r}) text length {text_len} outside "
                f"[{_MIN_TEXT_LEN}, {_MAX_TEXT_LEN}]"
            )
    return problems


def save_kb(path: Path, entries: list[dict[str, Any]]) -> None:
    """Atomically write a domain's KB entries to disk.

    Args:
        path: Result of ``kb_file_path()``.
        entries: The full new entry list to persist.

    Raises:
        ValueError: If ``validate_entries(entries)`` finds any problem.
    """
    problems = validate_entries(entries)
    if problems:
        raise ValueError("Invalid KB entries:\n  - " + "\n  - ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    tmp.replace(path)  # atomic swap, same pattern as system_config.save_config()
