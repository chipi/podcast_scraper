"""Per-user corpus revision counter + change log (RFC-114 Phase 1, #1470).

The keystone primitive of the personal-corpus definition: a monotonic per-user ``revision`` + a
bounded append-only **change log** whose entries carry ``added`` / ``removed`` — so a consumer
polling ``changes_since(rev)`` gets **both** additions and **tombstones** (removals). A flat
"changed-after-timestamp" cannot express deletions or retroactive membership; this can. It is the
primitive for episode-granular incremental consumers; RFC-113's Obsidian export ships its own
finer-grained content-hash vault snapshot instead (it must catch highlight-text/label edits, not
just episode membership), so it does not consume this log today — any episode consumer can.

**Reconcile-on-read (not writer-instrumented).** Consistent with the read-time-projection design
(PRD-041): rather than bumping a counter from every signal-writing route (fragile — one missed call
site silently corrupts the log), the revision advances when the corpus is **read**. ``reconcile``
recomputes the current ``{experienced, saved}`` membership, diffs it against the last persisted
snapshot, appends one event per add/remove, bumps the revision, and stores the new snapshot. This
captures playback-heard-crossings for free (a full recompute) and cannot drift from missed writers.

Layout: ``<data_dir>/users/<id>/corpus_log.json`` =
``{revision, snapshot: ["experienced:slug", …], events: [{seq, kind, facet, ref}]}``. The log is
bounded (``_MAX_EVENTS``); a consumer behind that window gets ``truncated: true`` and does a full
re-export (RFC-113 §2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server import app_user_corpus
from podcast_scraper.server.app_user_store import _is_safe_user_id
from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_FILE_NAME = "corpus_log.json"
_MAX_EVENTS = 1000  # bounded log; a consumer behind this window does a full re-export


def _path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / _FILE_NAME


def _lock(data_dir: Path, user_id: str) -> FileLock:
    path = _path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{_FILE_NAME}.lock")), timeout=_LOCK_TIMEOUT_S)


def _read(data_dir: Path, user_id: str) -> dict[str, Any]:
    path = _path(data_dir, user_id)
    if not path.is_file():
        return {"revision": 0, "snapshot": [], "events": []}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"revision": 0, "snapshot": [], "events": []}
    if not isinstance(doc, dict):
        return {"revision": 0, "snapshot": [], "events": []}
    doc.setdefault("revision", 0)
    doc.setdefault("snapshot", [])
    doc.setdefault("events", [])
    return doc


def _membership_keys(root: Path, data_dir: Path, user_id: str) -> set[str]:
    """Current membership as ``"facet:slug"`` keys. `saved` = episode-favorites − experienced."""
    experienced = app_user_corpus.experienced_episode_set(root, data_dir, user_id)
    saved = app_user_corpus.saved_episode_set(data_dir, user_id) - experienced
    keys = {f"experienced:{s}" for s in experienced}
    keys |= {f"saved:{s}" for s in saved}
    return keys


def reconcile(root: Path, data_dir: Path, user_id: str) -> int:
    """Recompute membership, append add/remove events for the delta, bump + return the revision."""
    if not _is_safe_user_id(user_id):
        return 0
    with _lock(data_dir, user_id):
        # Compute membership INSIDE the lock: two concurrent reconciles (web + native shell) that
        # each computed outside could persist stale-then-fresh out of order, emitting phantom
        # tombstone+re-add events into the very change stream that is supposed to be truthful (M5).
        current_keys = _membership_keys(root, data_dir, user_id)
        doc = _read(data_dir, user_id)
        prev_keys = set(doc["snapshot"])
        added = sorted(current_keys - prev_keys)
        removed = sorted(prev_keys - current_keys)
        if not added and not removed:
            return int(doc["revision"])  # nothing changed → no bump
        rev = int(doc["revision"])
        events = doc["events"]
        for key in removed:  # tombstones first, then adds
            rev += 1
            facet, _, ref = key.partition(":")
            events.append({"seq": rev, "kind": "removed", "facet": facet, "ref": ref})
        for key in added:
            rev += 1
            facet, _, ref = key.partition(":")
            events.append({"seq": rev, "kind": "added", "facet": facet, "ref": ref})
        if len(events) > _MAX_EVENTS:
            del events[: len(events) - _MAX_EVENTS]
        doc.update(revision=rev, snapshot=sorted(current_keys), events=events)
        atomic_write_text(_path(data_dir, user_id), json.dumps(doc, ensure_ascii=False, indent=2))
    return rev


def current(root: Path, data_dir: Path, user_id: str) -> int:
    """The user's current corpus revision, after reconciling against live membership."""
    return reconcile(root, data_dir, user_id)


def changes_since(root: Path, data_dir: Path, user_id: str, since: int) -> dict[str, Any]:
    """Events after ``since`` + the current revision + a ``truncated`` full-re-export flag.

    Reconciles first so the returned delta reflects live membership (incl. playback crossings).
    ``truncated`` is True when ``since`` predates the retained log window.
    """
    reconcile(root, data_dir, user_id)
    doc = _read(data_dir, user_id)
    events = [e for e in doc["events"] if isinstance(e, dict)]
    oldest = int(events[0]["seq"]) if events else 0
    # Truncated when the retained window starts AFTER the event right past `since` — i.e. the
    # consumer's next-needed seq (`since + 1`) was already trimmed. Covers `since=0` (a fresh
    # consumer) against an already-trimmed log, which `since > 0` previously missed (M6).
    truncated = bool(events) and oldest > since + 1
    delta = [e for e in events if int(e.get("seq", 0)) > since]
    return {
        "revision": int(doc["revision"]),
        "since": since,
        "truncated": truncated,
        "events": delta,
    }
