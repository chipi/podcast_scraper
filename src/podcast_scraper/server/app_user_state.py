"""Per-user mutable state as plain files (#1065, RFC-098 §3): playback, queue, library.

Builds on the per-user directory from #1063 (``<data_dir>/users/<id>/``). Each kind is one
JSON file; reads return a default when absent, writes are atomic. No DB — the personal
overlay only; shared corpus artifacts are never touched here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from podcast_scraper.server.atomic_write import atomic_write_text


def _state_path(data_dir: Path, user_id: str, name: str) -> Path:
    return data_dir / "users" / user_id / f"{name}.json"


def _read(data_dir: Path, user_id: str, name: str, default: Any) -> Any:
    path = _state_path(data_dir, user_id, name)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(data_dir: Path, user_id: str, name: str, obj: Any) -> None:
    path = _state_path(data_dir, user_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


# --- playback positions (slug -> {position_seconds, updated_at}) ---


def get_playback(data_dir: Path, user_id: str, slug: str) -> dict[str, Any] | None:
    data = _read(data_dir, user_id, "playback", {})
    rec = data.get(slug) if isinstance(data, dict) else None
    return rec if isinstance(rec, dict) else None


def set_playback(
    data_dir: Path, user_id: str, slug: str, position_seconds: float, updated_at: int
) -> dict[str, Any]:
    data = _read(data_dir, user_id, "playback", {})
    if not isinstance(data, dict):
        data = {}
    rec = {"position_seconds": position_seconds, "updated_at": updated_at}
    data[slug] = rec
    _write(data_dir, user_id, "playback", data)
    return rec


# --- queue (ordered list of slugs) ---


def get_queue(data_dir: Path, user_id: str) -> list[str]:
    data = _read(data_dir, user_id, "queue", [])
    return [str(x) for x in data] if isinstance(data, list) else []


def set_queue(data_dir: Path, user_id: str, items: list[str]) -> list[str]:
    clean = [str(x) for x in items]
    _write(data_dir, user_id, "queue", clean)
    return clean


# --- library (subscriptions; list of {feed_id, feed_url?, title?, added_at?}) ---


def get_library(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    data = _read(data_dir, user_id, "library", [])
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def add_subscription(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a subscription by ``feed_id`` (idempotent on feed_id)."""
    feed_id = item.get("feed_id")
    library = [x for x in get_library(data_dir, user_id) if x.get("feed_id") != feed_id]
    library.append(item)
    _write(data_dir, user_id, "library", library)
    return library


def remove_subscription(data_dir: Path, user_id: str, feed_id: str) -> list[dict[str, Any]]:
    library = [x for x in get_library(data_dir, user_id) if x.get("feed_id") != feed_id]
    _write(data_dir, user_id, "library", library)
    return library
