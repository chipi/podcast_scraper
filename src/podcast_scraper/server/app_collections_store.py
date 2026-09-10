"""Per-user collections / boards — the curation layer (#1417 / RFC-111 §1; RFC-119 typed items).

Named **mixed** buckets spanning the corpus — "prepare what to listen to next" (Pinterest-style).
A per-user overlay (PRD-035 Principle 3 — no forking of shared artifacts), same file-based store as
the rest: ``<data_dir>/users/<id>/collections.json`` =
``{collections: [{id, name, created_at}], items: {collection_id: [{kind, ref, ...}, ...]}}``.

An item is a typed reference ``{kind, ref}`` where ``kind`` ∈ highlight | episode | show | search |
topic | person | link (search carries ``scope``; link carries an optional ``title``). Membership is
de-duped by ``(kind, ref)`` and an item may belong to N collections. Deleting a collection drops its
membership only; the referenced things (highlights, episodes, follows) are untouched.

**Back-compat (RFC-119 migration):** the legacy shape stored bare highlight-id strings; a bare
string is read as ``{kind: highlight, ref}`` and rewritten in the new shape on the next mutation.
"""

from __future__ import annotations

import json
import re
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
#: A client may only supply an id in the exact minted shape (``col_`` + hex/base36); anything else
#: is ignored and a fresh id is minted, so a client-supplied id can never be an arbitrary string.
_COLLECTION_ID_RE = re.compile(r"^col_[a-z0-9]{1,32}$")

#: Item kinds a collection can hold (RFC-119). ``search`` carries a ``scope``; ``link`` an optional
#: ``title``; the rest are a bare ``ref`` (highlight id / episode slug / feed_id / topic|person id).
VALID_ITEM_KINDS = frozenset({"highlight", "episode", "show", "search", "topic", "person", "link"})
#: Optional per-kind extras carried through untouched.
_ITEM_EXTRAS = ("scope", "title")


def _normalize_item(raw: Any) -> dict[str, Any] | None:
    """A stored membership entry → a typed ``{kind, ref, ...}`` item, or ``None`` if unusable.

    Back-compat (RFC-119): a bare string is a legacy highlight id → ``{kind: highlight, ref}``.
    """
    if isinstance(raw, str):
        return {"kind": "highlight", "ref": raw} if raw else None
    if isinstance(raw, dict):
        kind = str(raw.get("kind") or "")
        ref = str(raw.get("ref") or "")
        if kind not in VALID_ITEM_KINDS or not ref:
            return None
        item: dict[str, Any] = {"kind": kind, "ref": ref}
        for extra in _ITEM_EXTRAS:
            if raw.get(extra) is not None:
                item[extra] = raw[extra]
        return item
    return None


def _item_key(item: dict[str, Any]) -> tuple[str, str]:
    """Stable identity for de-dup + removal — ``(kind, ref)``."""
    return (str(item.get("kind")), str(item.get("ref")))


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
    # Normalize membership to typed items on read (lazy RFC-119 migration): bare legacy highlight-id
    # strings become {kind: highlight, ref}; unusable entries are dropped. Writers persist the
    # normalized shape, so the file migrates on the next mutation.
    raw = doc.get("items")
    raw_items = raw if isinstance(raw, dict) else {}
    items: dict[str, list[dict[str, Any]]] = {}
    for cid, members in raw_items.items():
        if isinstance(members, list):
            items[str(cid)] = [it for it in (_normalize_item(m) for m in members) if it is not None]
    doc["items"] = items
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
        # Highlights can be deleted from under a collection (membership isn't cascaded), so a
        # highlight item counts only if it still exists; every other kind resolves at render time
        # from a shared store and is counted as present.
        return sum(
            1 for m in members if m.get("kind") != "highlight" or m.get("ref") in live_item_ids
        )

    out = [
        {
            **c,
            "count": _count(c["id"]),
            "updated_at": int(c.get("updated_at") or c.get("created_at") or 0),
        }
        for c in doc["collections"]
        if isinstance(c, dict) and c.get("id")
    ]
    return sorted(out, key=lambda c: int(c.get("created_at", 0)), reverse=True)


def create_collection(
    data_dir: Path, user_id: str, name: str, client_id: str | None = None
) -> tuple[dict[str, Any], bool]:
    """Create a named collection; returns ``(row, created)``.

    Idempotent under a client-minted ``client_id`` (#2004): a first write wins and a replay returns
    the existing row with ``created=False`` — the mechanism that lets an offline create-then-pin
    keep working, since the queued pin targets this same id. A ``client_id`` that is not the minted
    shape is ignored (a fresh id is minted) so a client cannot smuggle an arbitrary id into storage.

    Raises ValueError on an unsafe user id or empty/too-long name.
    """
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    clean = (name or "").strip()
    if not clean or len(clean) > _MAX_NAME_LEN:
        raise ValueError("collection name must be 1..%d chars" % _MAX_NAME_LEN)
    safe_client_id = client_id if client_id and _COLLECTION_ID_RE.match(client_id) else None
    with _lock(data_dir, user_id):
        doc = _read(data_dir, user_id, strict=True)
        if safe_client_id is not None:
            existing = next(
                (
                    c
                    for c in doc["collections"]
                    if isinstance(c, dict) and c.get("id") == safe_client_id
                ),
                None,
            )
            if existing is not None:
                count = len(doc["items"].get(safe_client_id, []))
                return {**existing, "count": count}, False
        if len(doc["collections"]) >= _MAX_COLLECTIONS:
            raise ValueError(f"at most {_MAX_COLLECTIONS} collections per user")
        now = int(time.time())
        collection = {
            "id": safe_client_id or f"col_{uuid.uuid4().hex[:12]}",
            "name": clean,
            "created_at": now,
            "updated_at": now,
            # derived + cached by the route on the first membership change (CO.6)
            "cover_url": None,
        }
        doc["collections"].append(collection)
        _write(data_dir, user_id, doc)
    return {**collection, "count": 0}, True


def set_cover(data_dir: Path, user_id: str, collection_id: str, cover_url: str | None) -> None:
    """Persist a collection's derived cover thumbnail (CO.6). No-op if the collection is gone or the
    value is unchanged. The route recomputes this on each membership change, so it stays a single
    cheap field on the list read rather than a per-render fan-out over members' artwork."""
    if not _is_safe_user_id(user_id):
        return
    with _lock(data_dir, user_id):
        doc = _read(data_dir, user_id, strict=True)
        row = next(
            (c for c in doc["collections"] if isinstance(c, dict) and c.get("id") == collection_id),
            None,
        )
        if row is None or row.get("cover_url") == cover_url:
            return
        row["cover_url"] = cover_url
        _write(data_dir, user_id, doc)


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


def _touch(doc: dict[str, Any], collection_id: str) -> None:
    """Bump a collection's ``updated_at`` (last-modified) — call on any membership change."""
    now = int(time.time())
    for c in doc["collections"]:
        if c.get("id") == collection_id:
            c["updated_at"] = now
            return


def add_item(
    data_dir: Path, user_id: str, collection_id: str, item: dict[str, Any]
) -> list[dict[str, Any]]:
    """Add a typed item ``{kind, ref, ...}`` to a collection (idempotent by ``(kind, ref)``).

    Returns the collection's items. Raises KeyError when the collection doesn't exist, ValueError on
    an unsafe id / invalid item / cap.
    """
    if not _is_safe_user_id(user_id):
        raise ValueError("unsafe user id")
    norm = _normalize_item(item)
    if norm is None:
        raise ValueError("invalid collection item")
    with _lock(data_dir, user_id):
        doc = _read(data_dir, user_id, strict=True)
        if not _collection_exists(doc, collection_id):
            raise KeyError(collection_id)
        members = doc["items"].setdefault(collection_id, [])
        key = _item_key(norm)
        if not any(_item_key(m) == key for m in members):
            # Checked only when actually appending: re-adding an existing member is idempotent and
            # must keep working at the cap, or a full collection could never be tidied.
            if len(members) >= _MAX_ITEMS_PER_COLLECTION:
                raise ValueError(f"at most {_MAX_ITEMS_PER_COLLECTION} items per collection")
            members.append(norm)
            _touch(doc, collection_id)
        _write(data_dir, user_id, doc)
        return list(members)


def remove_item(
    data_dir: Path, user_id: str, collection_id: str, kind: str, ref: str
) -> list[dict[str, Any]]:
    """Remove the item ``(kind, ref)`` from a collection. Returns the remaining items."""
    if not _is_safe_user_id(user_id):
        return []
    key = (str(kind), str(ref))
    with _lock(data_dir, user_id):
        doc = _read(data_dir, user_id, strict=True)
        if not _collection_exists(doc, collection_id):
            return []  # don't persist a ghost membership entry for an unknown collection
        before = doc["items"].get(collection_id, [])
        members = [m for m in before if _item_key(m) != key]
        if len(members) != len(before):
            _touch(doc, collection_id)
        doc["items"][collection_id] = members
        _write(data_dir, user_id, doc)
        return list(members)


def get_items(data_dir: Path, user_id: str, collection_id: str) -> list[dict[str, Any]]:
    """The typed items ``[{kind, ref, ...}]`` in a collection ([] when unknown)."""
    if not _is_safe_user_id(user_id):
        return []
    return list(_read(data_dir, user_id)["items"].get(collection_id, []))
