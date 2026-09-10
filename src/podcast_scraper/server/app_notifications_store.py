"""Per-user in-app notification inbox (wave-I, the ``in_app`` channel).

The inbox is what's *waiting when you open the app* — distinct from OS push (which reaches you
while the app is closed). One ``notifications.json`` per user: a bounded, newest-first list of
records, FileLock-serialised read-modify-writes (same overlay pattern as ``app_comms_store``).

Emitters (new-episode alerts J, product updates I.6) call :func:`emit`, which writes only when the
user has the ``in_app`` channel enabled for that TYPE — the consent gate lives here so every
emitter honours it the same way. The store itself is otherwise pure add/list/mark.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server import app_comms_store
from podcast_scraper.server.app_user_store import _is_safe_user_id
from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_FILE_NAME = "notifications.json"
#: Keep the inbox bounded — the newest this many records survive a write (older ones drop).
_MAX_RECORDS = 200


def _path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / _FILE_NAME


def _lock(data_dir: Path, user_id: str) -> FileLock:
    path = _path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{_FILE_NAME}.lock")), timeout=_LOCK_TIMEOUT_S)


def _read_raw(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    path = _path(data_dir, user_id)
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(doc, list):
        return []
    return [r for r in doc if isinstance(r, dict)]


def list_notifications(data_dir: Path, user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """The user's inbox, newest-first, capped at ``limit`` (read-only)."""
    if not _is_safe_user_id(user_id):
        return []
    records = _read_raw(data_dir, user_id)
    records.sort(key=lambda r: int(r.get("created_at", 0)), reverse=True)
    return records[: max(0, limit)]


def unread_count(data_dir: Path, user_id: str) -> int:
    """How many unread notifications the user has (drives the bell badge)."""
    if not _is_safe_user_id(user_id):
        return 0
    return sum(1 for r in _read_raw(data_dir, user_id) if not r.get("read"))


def add_notification(
    data_dir: Path,
    user_id: str,
    *,
    ntype: str,
    title: str,
    body: str | None = None,
    deep_link: str | None = None,
    meta: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
    now: int | None = None,
) -> dict[str, Any] | None:
    """Append a notification to the user's inbox (newest-first, bounded). Returns the record.

    ``dedupe_key`` — when given, an existing unread record with the same key is left in place and
    None is returned (so a re-run doesn't double-post the same alert). Raises ValueError for an
    unsafe user id.
    """
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    now = int(time.time()) if now is None else now
    with _lock(data_dir, user_id):
        records = _read_raw(data_dir, user_id)
        if dedupe_key is not None and any(r.get("dedupe_key") == dedupe_key for r in records):
            return None
        record: dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "type": ntype,
            "title": title,
            "read": False,
            "created_at": now,
        }
        if body is not None:
            record["body"] = body
        if deep_link is not None:
            record["deep_link"] = deep_link
        if meta:
            record["meta"] = meta
        if dedupe_key is not None:
            record["dedupe_key"] = dedupe_key
        records.append(record)
        records.sort(key=lambda r: int(r.get("created_at", 0)), reverse=True)
        _write(data_dir, user_id, records[:_MAX_RECORDS])
    return record


def emit(
    data_dir: Path,
    user_id: str,
    *,
    ntype: str,
    title: str,
    body: str | None = None,
    deep_link: str | None = None,
    meta: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
    now: int | None = None,
) -> dict[str, Any] | None:
    """Add a notification IFF the user has the in-app channel on for ``ntype`` (the consent gate).

    Returns the record when written, None when the in-app channel is off (or a dedupe hit). This
    is the entrypoint emitters use — the ``in_app`` matrix cell is checked here so no emitter can
    forget it."""
    comms = app_comms_store.get_comms(data_dir, user_id)
    if not app_comms_store.channel_enabled(comms, ntype, "in_app"):
        return None
    return add_notification(
        data_dir,
        user_id,
        ntype=ntype,
        title=title,
        body=body,
        deep_link=deep_link,
        meta=meta,
        dedupe_key=dedupe_key,
        now=now,
    )


def mark_read(data_dir: Path, user_id: str, notif_id: str) -> bool:
    """Mark one notification read. Returns True if it existed (idempotent when already read)."""
    if not _is_safe_user_id(user_id):
        return False
    with _lock(data_dir, user_id):
        records = _read_raw(data_dir, user_id)
        found = False
        for r in records:
            if r.get("id") == notif_id:
                found = True
                r["read"] = True
        if found:
            _write(data_dir, user_id, records)
    return found


def mark_all_read(data_dir: Path, user_id: str) -> int:
    """Mark every notification read. Returns how many were newly marked."""
    if not _is_safe_user_id(user_id):
        return 0
    with _lock(data_dir, user_id):
        records = _read_raw(data_dir, user_id)
        changed = 0
        for r in records:
            if not r.get("read"):
                r["read"] = True
                changed += 1
        if changed:
            _write(data_dir, user_id, records)
    return changed


def _write(data_dir: Path, user_id: str, records: list[dict[str, Any]]) -> None:
    path = _path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(records, ensure_ascii=False, indent=2))
