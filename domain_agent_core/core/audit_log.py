"""Compliance-aware structured audit logging.

Deliberately diverges from ``app/latency_log.py``'s size-triggered
single-backup rotation (``_MAX_LOG_BYTES``-gated, keeps one ``.1`` backup
then discards anything older) - that rotation model silently discards old
data, which is exactly wrong for a multi-year regulatory retention
requirement (HIPAA: 6 years: ``CompliancePolicy.retention_days=2190``;
PCI-DSS/GLBA: 7 years: ``retention_days=2555``). Instead, this rolls to a
new dated file per month and never deletes any of them from this codebase
- ``retention_days`` is used only to flag files eligible for EXTERNAL
archival, never to delete them here.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from domain_agent_core.core.compliance_profiles import CompliancePolicy


def _redact_value(value: Any, policy: CompliancePolicy) -> Any:
    """Hash (not mask) a value if it's free text matching a redact
    pattern; leave non-string values untouched.

    Args:
        value: The raw value to check.
        policy: The active ``CompliancePolicy``.

    Returns:
        The value unchanged, or ``"<redacted:sha256:...>"`` if it matched
        a redact pattern.
    """
    if not isinstance(value, str):
        return value
    for pattern in policy.redact_patterns:
        if pattern.search(value):
            import hashlib

            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
            return f"<redacted:sha256:{digest}>"
    return value


def _redact_payload(payload: dict[str, Any], policy: CompliancePolicy) -> dict[str, Any]:
    """Redact a tool-call payload per the active compliance policy.

    Args:
        payload: Raw event payload (tool args/result).
        policy: The active ``CompliancePolicy``.

    Returns:
        A new dict - fields in ``policy.redact_fields`` are hashed
        outright; every remaining string field is scanned against
        ``policy.redact_patterns``.
    """
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if key in policy.redact_fields and isinstance(value, str):
            import hashlib

            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
            redacted[key] = f"<redacted:sha256:{digest}>"
        else:
            redacted[key] = _redact_value(value, policy)
    return redacted


def _log_file_path(audit_dir: Path, domain_id: str, when: datetime) -> Path:
    """Build the dated, append-only log file path for one domain/month.

    Args:
        audit_dir: The pack's audit-log directory.
        domain_id: The domain identifier.
        when: Timestamp used to pick the month partition.

    Returns:
        ``<audit_dir>/<domain_id>_audit_<YYYY-MM>.jsonl``.
    """
    return audit_dir / f"{domain_id}_audit_{when.strftime('%Y-%m')}.jsonl"


def record(
    audit_dir: Path,
    domain_id: str,
    policy: CompliancePolicy,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Append one redacted, timestamped audit event.

    Args:
        audit_dir: The pack's audit-log directory (created if missing).
        domain_id: The domain identifier.
        policy: The active ``CompliancePolicy`` (governs redaction; a
            "none"-level policy still writes standard log lines, just
            without strict redaction).
        event_type: Short label, e.g. "tool_call" or "safety_escalation".
        payload: Event-specific fields (tool name, args, result, etc.).

    Note:
        This function NEVER deletes or rotates away an existing file -
        it only ever appends to (or creates) the current month's file.
        ``policy.retention_days`` is metadata for an external archival
        process, not something this function acts on.
    """
    if policy.audit_log_level == "none":
        return
    audit_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    entry = {
        "timestamp": now.isoformat(),
        "domain_id": domain_id,
        "event_type": event_type,
        "compliance_profile": policy.name,
        "payload": (
            _redact_payload(payload, policy)
            if policy.audit_log_level == "strict"
            else payload
        ),
    }
    path = _log_file_path(audit_dir, domain_id, now)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


_MONTH_FILE_RE = re.compile(r"_audit_(\d{4}-\d{2})\.jsonl$")


def list_audit_files(audit_dir: Path, domain_id: str) -> list[Path]:
    """List every audit file ever written for a domain, oldest first.

    Args:
        audit_dir: The pack's audit-log directory.
        domain_id: The domain identifier.

    Returns:
        Every ``<domain_id>_audit_*.jsonl`` file found, sorted by the
        month encoded in its filename.
    """
    if not audit_dir.is_dir():
        return []
    files = [
        p
        for p in audit_dir.glob(f"{domain_id}_audit_*.jsonl")
        if _MONTH_FILE_RE.search(p.name)
    ]
    return sorted(files, key=lambda p: _MONTH_FILE_RE.search(p.name).group(1))
