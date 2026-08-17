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

from podcast_scraper.server.app_user_state import UserStateUnreadable
from podcast_scraper.server.app_user_store import _is_safe_user_id
from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_FILE_NAME = "collections.json"
_MAX_NAME_LEN = 120
#: Generous count caps (#51). High enough that no real user reaches one, low enough that a
#: runaway client loop cannot grow this account's own files until its own reads degrade.
#: Enforced on the WRITE, so a violation is an error rather than a silent no-op, and applied
#: to new writes only — nothing already stored is trimmed.
_MAX_COLLECTIONS = 200
_MAX_ITEMS_PER_COLLECTION = 1_000


def _path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / _FILE_NAME


def _lock(data_dir: Path, user_id: str) -> FileLock:
    path = _path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{_FILE_NAME}.lock")), timeout=_LOCK_TIMEOUT_S)


def _read(data_dir: Path, user_id: str, *, strict: bool = False) -> dict[str, Any]:
    """The user's collections doc, or the empty doc.

    With ``strict``, an unusable EXISTING file raises.

    Readers stay lenient — a browsable UI over a temporarily bad file beats a 500. Writers must not
    be: every mutator here persists what it just read, so answering a bad read with the empty doc
    meant creating one collection replaced every collection AND every membership the user had. See
    :class:`podcast_scraper.server.app_user_state.UserStateUnreadable` for the same rule stated for
    the per-user state files.
    """
    path = _path(data_dir, user_id)
    if not path.is_file():
        return {"collections": [], "items": {}}  # genuinely absent — the empty doc is the truth
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if strict:
            raise UserStateUnreadable(f"collections.json is unreadable for user {user_id}") from exc
        return {"collections": [], "items": {}}
    if not isinstance(doc, dict):
        if strict:
            raise UserStateUnreadable(f"collections.json is not a mapping for user {user_id}")
        return {"collections": [], "items": {}}
    doc.setdefault("collections", [])
    doc.setdefault("items", {})
    return doc


def _write(data_dir: Path, user_id: str, doc: dict[str, Any]) -> None:
    path = _path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(doc, ensure_ascii=False, indent=2))


def list_collections(
    data_dir: Path, user_id: str, *, live_item_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    """Collections with their current item counts (newest-first).

    ``live_item_ids`` is the set of highlight ids that still exist. Pass it and ``count`` counts
    only members that can actually render; omit it and ``count`` is the raw membership length.

    It has to be passed in rather than read here: membership is a plain id list and this store
    deliberately knows nothing about the capture store. But the raw length is a LIE the moment a
    highlight is deleted — deleting a highlight touches only ``highlights.json``, so the id stays
    in every collection that held it. The detail view already drops ids it cannot hydrate, and
    ``CollectionDetail`` carries both numbers in one response, so the badge said 5 while 3 cards
    rendered. Counting at read time (rather than cascading a delete across three stores) matches
    how the rest of this codebase handles derived truth.
    """
    if not _is_safe_user_id(user_id):
        return []
    doc = _read(data_dir, user_id)
    items = doc["items"]

    def _count(cid: str) -> int:
        members = items.get(cid, [])
        if live_item_ids is None:
            return len(members)
        return sum(1 for h in members if h in live_item_ids)

    out = [
        {**c, "count": _count(c["id"])}
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
        doc = _read(data_dir, user_id, strict=True)
        if len(doc["collections"]) >= _MAX_COLLECTIONS:
            raise ValueError(f"at most {_MAX_COLLECTIONS} collections per user")
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
        doc = _read(data_dir, user_id, strict=True)
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
        doc = _read(data_dir, user_id, strict=True)
        if not _collection_exists(doc, collection_id):
            raise KeyError(collection_id)
        members = doc["items"].setdefault(collection_id, [])
        if highlight_id not in members:
            # Checked only when actually appending: re-adding an existing member is idempotent and
            # must keep working at the cap, or a full collection could never be tidied.
            if len(members) >= _MAX_ITEMS_PER_COLLECTION:
                raise ValueError(f"at most {_MAX_ITEMS_PER_COLLECTION} items per collection")
            members.append(highlight_id)
        _write(data_dir, user_id, doc)
        return list(members)


def remove_item(data_dir: Path, user_id: str, collection_id: str, highlight_id: str) -> list[str]:
    """Remove a highlight from a collection. Returns the remaining item ids."""
    if not _is_safe_user_id(user_id):
        return []
    with _lock(data_dir, user_id):
        doc = _read(data_dir, user_id, strict=True)
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
