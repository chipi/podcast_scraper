"""Per-user Web-Push subscriptions (#1415, PRD-046 FR1 / RFC-110 §6).

A user may have several push subscriptions (one per browser/device). Each is a W3C
``PushSubscription`` JSON (``{endpoint, expirationTime, keys:{p256dh,auth}}``), stored opaquely —
the app never inspects the keys; the infra worker signs + delivers (VAPID). Same file-based per-user
overlay as ``app_user_preferences``: ``<data_dir>/users/<id>/push_subscriptions.json``.

Dedupe is on ``endpoint`` (the stable identity of a subscription). A dead subscription is removed
either by the client (unsubscribe) or by the outbox suppression path on a push ``410/404`` bounce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server.app_user_store import _is_safe_user_id
from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_FILE_NAME = "push_subscriptions.json"


def _path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / _FILE_NAME


def _lock(data_dir: Path, user_id: str) -> FileLock:
    path = _path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{_FILE_NAME}.lock")), timeout=_LOCK_TIMEOUT_S)


def list_subscriptions(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Return the user's stored push subscriptions ([] when unset / unreadable / unsafe id)."""
    if not _is_safe_user_id(user_id):
        return []
    path = _path(data_dir, user_id)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return (
        [s for s in data if isinstance(s, dict) and s.get("endpoint")]
        if isinstance(data, list)
        else []
    )


def add_subscription(
    data_dir: Path, user_id: str, subscription: dict[str, Any]
) -> list[dict[str, Any]]:
    """Add (or replace, keyed on ``endpoint``) a subscription. Returns the full list."""
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    endpoint = str(subscription.get("endpoint") or "")
    if not endpoint:
        raise ValueError("subscription must carry an endpoint")
    with _lock(data_dir, user_id):
        subs = [s for s in list_subscriptions(data_dir, user_id) if s.get("endpoint") != endpoint]
        subs.append(subscription)
        path = _path(data_dir, user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(subs, ensure_ascii=False, indent=2))
    return subs


def remove_subscription(data_dir: Path, user_id: str, endpoint: str) -> list[dict[str, Any]]:
    """Remove the subscription with ``endpoint``. Returns the remaining list."""
    if not _is_safe_user_id(user_id):
        return []
    with _lock(data_dir, user_id):
        subs = [s for s in list_subscriptions(data_dir, user_id) if s.get("endpoint") != endpoint]
        path = _path(data_dir, user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(subs, ensure_ascii=False, indent=2))
    return subs
