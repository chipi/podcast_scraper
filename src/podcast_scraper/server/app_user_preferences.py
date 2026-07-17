"""USERPREFS-1 — per-user preferences persistence (JSONB-shaped, file-backed).

Cross-device sync for the UI's opinion-state that has lived in ``localStorage``
until now (graph lens flags, theme choice, panel collapse state, corpus path,
etc.). Same file-based per-user overlay as ``app_user_state`` (RFC-098 §3):

- One file per user under ``<data_dir>/users/<id>/preferences.json``.
- FileLock-serialised read-modify-writes to keep concurrent tab writes safe.
- Payload is a single free-form JSON object — the client owns the shape, so
  new preference keys don't require a server release. The server never
  interprets the payload; it just round-trips it.

Kept intentionally naive: no schema validation, no migrations, no versioning.
The client-side store (``useUserPreferencesStore``) treats absent keys as
"use the local default" so payload shape drift is a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_FILE_NAME = "preferences.json"


def _prefs_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / _FILE_NAME


def _prefs_lock(data_dir: Path, user_id: str) -> FileLock:
    path = _prefs_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{_FILE_NAME}.lock")), timeout=_LOCK_TIMEOUT_S)


def get_preferences(data_dir: Path, user_id: str) -> dict[str, Any]:
    """Return the user's preferences payload, or ``{}`` when unset / unreadable."""
    path = _prefs_path(data_dir, user_id)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def replace_preferences(data_dir: Path, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically replace the user's preferences with ``payload``. Returns the stored value."""
    if not isinstance(payload, dict):
        raise ValueError("preferences payload must be an object")
    with _prefs_lock(data_dir, user_id):
        path = _prefs_path(data_dir, user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def patch_preferences(data_dir: Path, user_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge ``updates`` into the user's preferences and return the new state.

    Keys whose value is ``None`` are DELETED from the stored payload — the client uses
    that to reset a specific preference to its local default without touching others.
    Any other value type (bool, string, number, nested object, array) is stored as-is.
    """
    if not isinstance(updates, dict):
        raise ValueError("preferences patch must be an object")
    with _prefs_lock(data_dir, user_id):
        current = get_preferences(data_dir, user_id)
        for key, value in updates.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        path = _prefs_path(data_dir, user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(current, ensure_ascii=False, indent=2))
    return current
