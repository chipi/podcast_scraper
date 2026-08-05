"""Per-user comms / delivery consent (#1414, PRD-046 FR1, RFC-110 §3.1).

The consent + cadence a user has set for outbound delivery (the "Your Week" digest email +
Web-Push resurfacing nudges). Same file-based per-user overlay as ``app_user_preferences``
(RFC-098 §3): one ``comms.json`` per user, FileLock-serialised read-modify-writes.

This store is the gate for delivery: the digest assembler (#1415) only enqueues a
``DeliveryEnvelope`` for a user whose ``digest.enabled`` (or ``push.enabled``) is set. The
``unsubscribe_ref`` is an opaque, rotatable handle the delivery service embeds in the
one-click unsubscribe link; :func:`unsubscribe` resolves it back to the user and disables the
digest. ``email_verified`` is NOT stored here — it is identity-derived (from the OAuth
provider) at the route layer.
"""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server.app_user_store import _is_safe_user_id
from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_FILE_NAME = "comms.json"

#: Off by default (opt-in). Sunday (Python weekday 6) 13:00 in the user's cadence window.
DEFAULTS: dict[str, Any] = {
    "digest": {
        "enabled": False,
        "cadence": "weekly",
        "day_of_week": 6,
        "hour": 13,
        "paused": False,
    },
    "push": {"enabled": False},
}


def _comms_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / _FILE_NAME


def _comms_lock(data_dir: Path, user_id: str) -> FileLock:
    path = _comms_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{_FILE_NAME}.lock")), timeout=_LOCK_TIMEOUT_S)


def _read_raw(data_dir: Path, user_id: str) -> dict[str, Any]:
    path = _comms_path(data_dir, user_id)
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _merged(stored: dict[str, Any]) -> dict[str, Any]:
    """Overlay a stored payload onto a deep copy of DEFAULTS (missing nested keys default)."""
    out = copy.deepcopy(DEFAULTS)
    for section in ("digest", "push"):
        val = stored.get(section)
        if isinstance(val, dict):
            out[section].update({k: v for k, v in val.items() if k in out[section]})
    if isinstance(stored.get("unsubscribe_ref"), str):
        out["unsubscribe_ref"] = stored["unsubscribe_ref"]
    return out


def get_comms(data_dir: Path, user_id: str) -> dict[str, Any]:
    """Return the user's comms settings merged onto defaults (read-only; no ref minted)."""
    if not _is_safe_user_id(user_id):
        return _merged({})
    return _merged(_read_raw(data_dir, user_id))


def set_comms(
    data_dir: Path,
    user_id: str,
    *,
    digest: dict[str, Any] | None = None,
    push: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Partial-update the user's comms settings; mint an ``unsubscribe_ref`` on first write.

    Only the known keys of each section are written (unknown keys ignored). Returns the merged
    settings (including the ref). Raises ValueError for an unsafe user id.
    """
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    with _comms_lock(data_dir, user_id):
        current = _merged(_read_raw(data_dir, user_id))
        if digest:
            current["digest"].update({k: v for k, v in digest.items() if k in current["digest"]})
        if push:
            current["push"].update({k: v for k, v in push.items() if k in current["push"]})
        if not current.get("unsubscribe_ref"):
            current["unsubscribe_ref"] = uuid.uuid4().hex
        path = _comms_path(data_dir, user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(current, ensure_ascii=False, indent=2))
    return current


def unsubscribe(data_dir: Path, ref: str) -> bool:
    """Resolve an ``unsubscribe_ref`` to its user and disable the digest. One-click, no auth.

    O(users) scan (acceptable at current scale; RFC-101 OQ-1). Returns True when a matching
    user was found and updated, False otherwise. Idempotent — re-hitting a used link is a no-op
    that still returns True.
    """
    if not ref:
        return False
    users_dir = data_dir / "users"
    if not users_dir.is_dir():
        return False
    for child in sorted(users_dir.iterdir()):
        if not child.is_dir():
            continue
        raw = _read_raw(data_dir, child.name)
        if raw.get("unsubscribe_ref") == ref:
            with _comms_lock(data_dir, child.name):
                current = _merged(_read_raw(data_dir, child.name))
                # Re-verify under the lock — the ref could have been rotated out between the
                # unlocked scan and here; don't disable the digest for a stale/rotated ref.
                if current.get("unsubscribe_ref") != ref:
                    return False
                current["digest"]["enabled"] = False
                atomic_write_text(
                    _comms_path(data_dir, child.name),
                    json.dumps(current, ensure_ascii=False, indent=2),
                )
            return True
    return False
