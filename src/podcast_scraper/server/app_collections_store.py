"""Per-user collections / boards — the curation layer (#1417, PRD-046 FR4 / RFC-111 §1).

Named sets of highlights that span episodes: the active "organize the interesting bits" surface
above the flat highlight list. A per-user overlay (PRD-035 Principle 3 — no forking of shared
artifacts), same file-based store as the rest: ``<data_dir>/users/<id>/collections.json`` =
``{collections: [{id, name, created_at}], items: {collection_id: [highlight_id, ...]}}``.

A highlight may belong to N collections (membership is a plain id list). Deleting a collection drops
its membership; the highlights themselves are untouched (they live in the capture store).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server.app_user_store import _is_safe_user_id
from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_FILE_NAME = "collections.json"
_MAX_NAME_LEN = 120


def _path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / _FILE_NAME


def _lock(data_dir: Path, user_id: str) -> FileLock:
    path = _path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{_FILE_NAME}.lock")), timeout=_LOCK_TIMEOUT_S)


def _read(data_dir: Path, user_id: str) -> dict[str, Any]:
    path = _path(data_dir, user_id)
    if not path.is_file():
        return {"collections": [], "items": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"collections": [], "items": {}}
    if not isinstance(doc, dict):
        return {"collections": [], "items": {}}
    doc.setdefault("collections", [])
    doc.setdefault("items", {})
    return doc


def _write(data_dir: Path, user_id: str, doc: dict[str, Any]) -> None:
    path = _path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(doc, ensure_ascii=False, indent=2))


def list_collections(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Collections with their current item counts (newest-first)."""
    if not _is_safe_user_id(user_id):
        return []
    doc = _read(data_dir, user_id)
    items = doc["items"]
    out = [
        {**c, "count": len(items.get(c["id"], []))}
        for c in doc["collections"]
        if isinstance(c, dict) and c.get("id")
    ]
    return sorted(out, key=lambda c: int(c.get("created_at", 0)), reverse=True)


def create_collection(data_dir: Path, user_id: str, name: str) -> dict[str, Any]:
    """Create a named collection. Raises ValueError on an unsafe id or empty/too-long name."""
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    clean = (name or "").strip()
    if not clean or len(clean) > _MAX_NAME_LEN:
        raise ValueError("collection name must be 1..%d chars" % _MAX_NAME_LEN)
    with _lock(data_dir, user_id):
        doc = _read(data_dir, user_id)
        collection = {
            "id": f"col_{uuid.uuid4().hex[:12]}",
            "name": clean,
            "created_at": int(time.time()),
        }
        doc["collections"].append(collection)
        _write(data_dir, user_id, doc)
    return {**collection, "count": 0}


def delete_collection(data_dir: Path, user_id: str, collection_id: str) -> bool:
    """Remove a collection + its membership. Returns True when something was removed."""
    if not _is_safe_user_id(user_id):
        return False
    with _lock(data_dir, user_id):
        doc = _read(data_dir, user_id)
        before = len(doc["collections"])
        doc["collections"] = [c for c in doc["collections"] if c.get("id") != collection_id]
        doc["items"].pop(collection_id, None)
        removed = len(doc["collections"]) != before
        if removed:
            _write(data_dir, user_id, doc)
    return removed


def _collection_exists(doc: dict[str, Any], collection_id: str) -> bool:
    return any(c.get("id") == collection_id for c in doc["collections"])


def add_item(data_dir: Path, user_id: str, collection_id: str, highlight_id: str) -> list[str]:
    """Add a highlight to a collection (idempotent). Returns the collection's item ids.

    Raises KeyError when the collection doesn't exist.
    """
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    with _lock(data_dir, user_id):
        doc = _read(data_dir, user_id)
        if not _collection_exists(doc, collection_id):
            raise KeyError(collection_id)
        members = doc["items"].setdefault(collection_id, [])
        if highlight_id not in members:
            members.append(highlight_id)
        _write(data_dir, user_id, doc)
        return list(members)


def remove_item(data_dir: Path, user_id: str, collection_id: str, highlight_id: str) -> list[str]:
    """Remove a highlight from a collection. Returns the remaining item ids."""
    if not _is_safe_user_id(user_id):
        return []
    with _lock(data_dir, user_id):
        doc = _read(data_dir, user_id)
        if not _collection_exists(doc, collection_id):
            return []  # don't persist a ghost membership entry for an unknown collection
        members = [h for h in doc["items"].get(collection_id, []) if h != highlight_id]
        doc["items"][collection_id] = members
        _write(data_dir, user_id, doc)
        return list(members)


def get_items(data_dir: Path, user_id: str, collection_id: str) -> list[str]:
    """The highlight ids in a collection ([] when unknown)."""
    if not _is_safe_user_id(user_id):
        return []
    return list(_read(data_dir, user_id)["items"].get(collection_id, []))
