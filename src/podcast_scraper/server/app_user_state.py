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
    """Return the saved playback record for an episode, or ``None`` when unset."""
    data = _read(data_dir, user_id, "playback", {})
    rec = data.get(slug) if isinstance(data, dict) else None
    return rec if isinstance(rec, dict) else None


def set_playback(
    data_dir: Path, user_id: str, slug: str, position_seconds: float, updated_at: int
) -> dict[str, Any]:
    """Save the playback position for an episode; return the stored record."""
    data = _read(data_dir, user_id, "playback", {})
    if not isinstance(data, dict):
        data = {}
    rec = {"position_seconds": position_seconds, "updated_at": updated_at}
    data[slug] = rec
    _write(data_dir, user_id, "playback", data)
    return rec


def list_playback(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """All saved playback positions, newest-updated first (for the Home 'Continue' rail)."""
    data = _read(data_dir, user_id, "playback", {})
    if not isinstance(data, dict):
        return []
    out: list[dict[str, Any]] = []
    for slug, rec in data.items():
        if isinstance(rec, dict):
            out.append(
                {
                    "slug": str(slug),
                    "position_seconds": float(rec.get("position_seconds", 0.0)),
                    "updated_at": rec.get("updated_at"),
                }
            )
    out.sort(key=lambda r: (r.get("updated_at") or 0), reverse=True)
    return out


# --- queue (ordered list of slugs) ---


def get_queue(data_dir: Path, user_id: str) -> list[str]:
    """Return the user's play queue (ordered slugs); empty when unset."""
    data = _read(data_dir, user_id, "queue", [])
    return [str(x) for x in data] if isinstance(data, list) else []


def set_queue(data_dir: Path, user_id: str, items: list[str]) -> list[str]:
    """Replace the user's play queue; return the stored list."""
    clean = [str(x) for x in items]
    _write(data_dir, user_id, "queue", clean)
    return clean


# --- interests (personalized discovery; ordered list of cluster ids) ---


def get_interests(data_dir: Path, user_id: str) -> list[str]:
    """Return the user's interest cluster ids (graph_compound_parent_id); empty when unset."""
    data = _read(data_dir, user_id, "interests", [])
    return [str(x) for x in data] if isinstance(data, list) else []


def set_interests(data_dir: Path, user_id: str, cluster_ids: list[str]) -> list[str]:
    """Replace the user's interests; return the stored list (de-duplicated, order preserved)."""
    seen: set[str] = set()
    clean: list[str] = []
    for x in cluster_ids:
        s = str(x)
        if s and s not in seen:
            seen.add(s)
            clean.append(s)
    _write(data_dir, user_id, "interests", clean)
    return clean


# --- library (subscriptions; list of {feed_id, feed_url?, title?, added_at?}) ---


def get_library(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Return the user's subscriptions; empty when unset."""
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
    """Remove a subscription by ``feed_id`` (no-op if absent); return the remaining list."""
    library = [x for x in get_library(data_dir, user_id) if x.get("feed_id") != feed_id]
    _write(data_dir, user_id, "library", library)
    return library
