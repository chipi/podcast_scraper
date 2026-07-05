"""Per-user mutable state as plain files (#1065, RFC-098 §3): playback, queue, library.

Builds on the per-user directory from #1063 (``<data_dir>/users/<id>/``). Each kind is one
JSON file; reads return a default when absent, writes are atomic. No DB — the personal
overlay only; shared corpus artifacts are never touched here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server.atomic_write import atomic_write_text

# Read-modify-write mutations on one user's file must not interleave (a second writer reading the
# pre-write state would lose the first's append). Each mutator holds a per-(user, file) lock over
# its read+write; the timeout makes a stuck lock fail loudly rather than deadlock.
_LOCK_TIMEOUT_S = 15.0


def _state_path(data_dir: Path, user_id: str, name: str) -> Path:
    return data_dir / "users" / user_id / f"{name}.json"


def _user_lock(data_dir: Path, user_id: str, name: str) -> FileLock:
    """A per-(user, file) write lock; serialises concurrent read-modify-write on that file."""
    path = _state_path(data_dir, user_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{name}.lock")), timeout=_LOCK_TIMEOUT_S)


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
    with _user_lock(data_dir, user_id, "playback"):
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


# --- listen events (append-only log of episode opens, for analytics ) ---
#
# One line of JSON per "open", in <data_dir>/users/<id>/listen_events.jsonl. Append-only so the
# series is cheap to write and never rewrites history; aggregation (streaks, sparklines, cross-user
# listener counts) reads the whole small log. This is the ONLY per-listen history we keep — playback
# stays last-position-only.


def _events_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / "listen_events.jsonl"


def append_listen_event(
    data_dir: Path, user_id: str, slug: str, feed_id: str | None, ts: int
) -> None:
    """Append one 'opened this episode' event to the user's listen log."""
    path = _events_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"slug": str(slug), "feed_id": feed_id, "ts": int(ts)}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def list_listen_events(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """All of one user's listen events (chronological as written); skips corrupt lines."""
    path = _events_path(data_dir, user_id)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("slug") and rec.get("ts") is not None:
            out.append(rec)
    return out


def iter_user_ids(data_dir: Path) -> list[str]:
    """Every user id with a per-user directory (for cross-user aggregation)."""
    users_dir = data_dir / "users"
    if not users_dir.is_dir():
        return []
    return [p.name for p in users_dir.iterdir() if p.is_dir()]


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


# --- favorites (polymorphic "saved things": episodes, insights, … keyed by kind+ref) ---


def get_favorites(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Return the user's saved favorites (newest-last as stored); empty when unset."""
    data = _read(data_dir, user_id, "favorites", [])
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict) and x.get("kind") and x.get("ref")]


def add_favorite(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a favorite (idempotent on ``kind``+``ref``); appended newest-last."""
    kind, ref = item.get("kind"), item.get("ref")
    with _user_lock(data_dir, user_id, "favorites"):
        favorites = [
            x
            for x in get_favorites(data_dir, user_id)
            if (x.get("kind"), x.get("ref")) != (kind, ref)
        ]
        favorites.append(item)
        _write(data_dir, user_id, "favorites", favorites)
        return favorites


def remove_favorite(data_dir: Path, user_id: str, kind: str, ref: str) -> list[dict[str, Any]]:
    """Remove a favorite by ``kind``+``ref`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "favorites"):
        favorites = [
            x
            for x in get_favorites(data_dir, user_id)
            if (x.get("kind"), x.get("ref")) != (kind, ref)
        ]
        _write(data_dir, user_id, "favorites", favorites)
        return favorites


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


def add_interest(data_dir: Path, user_id: str, token: str) -> list[str]:
    """Follow one interest token (cluster ``tc:``, topic ``topic:`` or person ``person:``)."""
    with _user_lock(data_dir, user_id, "interests"):
        return set_interests(data_dir, user_id, [*get_interests(data_dir, user_id), token])


def remove_interest(data_dir: Path, user_id: str, token: str) -> list[str]:
    """Unfollow one interest token (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "interests"):
        return set_interests(
            data_dir, user_id, [x for x in get_interests(data_dir, user_id) if x != token]
        )


# --- library (subscriptions; list of {feed_id, feed_url?, title?, added_at?}) ---


def get_library(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Return the user's subscriptions; empty when unset."""
    data = _read(data_dir, user_id, "library", [])
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def add_subscription(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a subscription by ``feed_id`` (idempotent on feed_id)."""
    feed_id = item.get("feed_id")
    with _user_lock(data_dir, user_id, "library"):
        library = [x for x in get_library(data_dir, user_id) if x.get("feed_id") != feed_id]
        library.append(item)
        _write(data_dir, user_id, "library", library)
        return library


def remove_subscription(data_dir: Path, user_id: str, feed_id: str) -> list[dict[str, Any]]:
    """Remove a subscription by ``feed_id`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "library"):
        library = [x for x in get_library(data_dir, user_id) if x.get("feed_id") != feed_id]
        _write(data_dir, user_id, "library", library)
        return library


# --- highlights (P2 Capture, PRD-040 / RFC-098 §7: "mark this moment" + transcript spans) ---
#
# A highlight is a captured moment in an episode the user wants to keep: a transcript ``span``
# selection, a one-tap ``moment`` (a single timestamp), or a saved ``insight`` (grounded GIL claim).
# Stored as one ``highlights.json`` list (newest-last), keyed by an opaque ``id`` the route mints.
#
# **The timestamp is the stable anchor.** Char offsets and segment ids are positional and drift when
# an episode is re-scraped (transcript text shifts); ``start_ms``/``end_ms`` survive.
# ``reanchor_highlight`` recomputes the positional fields against a fresh transcript and NEVER
# drops a highlight — a span that no longer resolves is marked ``anchor_status="drifted"`` (§7).

# ``kind`` (span|moment|insight) + ``target`` (highlight|insight|episode) are validated at the API
# boundary by the route's Pydantic ``Literal`` fields; the store stays permissive and only protects
# the immutable identity fields below from being overwritten on update.
_IMMUTABLE_HIGHLIGHT_FIELDS = frozenset({"id", "episode_slug", "created_at"})


def get_highlights(
    data_dir: Path, user_id: str, episode_slug: str | None = None
) -> list[dict[str, Any]]:
    """Return saved highlights (newest-last), optionally scoped to one episode."""
    data = _read(data_dir, user_id, "highlights", [])
    if not isinstance(data, list):
        return []
    out = [
        x
        for x in data
        if isinstance(x, dict)
        and x.get("id")
        and x.get("episode_slug")
        and x.get("kind")
        and x.get(
            "created_at"
        )  # required by the Highlight response model; drop hand-corrupted rows
    ]
    if episode_slug is not None:
        out = [x for x in out if x.get("episode_slug") == episode_slug]
    return out


def add_highlight(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a highlight (idempotent on ``id``); appended newest-last."""
    hid = item.get("id")
    with _user_lock(data_dir, user_id, "highlights"):
        highlights = [x for x in get_highlights(data_dir, user_id) if x.get("id") != hid]
        highlights.append(item)
        _write(data_dir, user_id, "highlights", highlights)
        return highlights


def update_highlight(
    data_dir: Path, user_id: str, highlight_id: str, fields: dict[str, Any]
) -> dict[str, Any] | None:
    """Merge ``fields`` into a highlight by ``id`` (no-op if absent); return the updated record.

    Used for in-place edits (``color``, ``quote_text``) and persisting a re-anchor. ``id``,
    ``episode_slug`` and ``created_at`` are immutable and cannot be overwritten via ``fields``.
    """
    with _user_lock(data_dir, user_id, "highlights"):
        highlights = get_highlights(data_dir, user_id)
        updated: dict[str, Any] | None = None
        for rec in highlights:
            if rec.get("id") == highlight_id:
                rec.update(
                    {k: v for k, v in fields.items() if k not in _IMMUTABLE_HIGHLIGHT_FIELDS}
                )
                updated = rec
                break
        if updated is not None:
            _write(data_dir, user_id, "highlights", highlights)
        return updated


def remove_highlight(data_dir: Path, user_id: str, highlight_id: str) -> list[dict[str, Any]]:
    """Remove a highlight by ``id`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "highlights"):
        highlights = [x for x in get_highlights(data_dir, user_id) if x.get("id") != highlight_id]
        _write(data_dir, user_id, "highlights", highlights)
        return highlights


def reanchor_highlight(highlight: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-resolve a highlight's positional fields against a fresh transcript by its time anchor.

    ``segments`` are the new transcript segments, each ``{segment_id, start_ms, end_ms, char_start,
    char_end}``. The highlight's ``start_ms``/``end_ms`` are the stable anchor; this recomputes
    ``segment_ids``/``char_start``/``char_end`` from the segments that overlap that time window and
    sets ``anchor_status`` to ``"anchored"``. If nothing overlaps (transcript shifted out from under
    it) the positional fields are left untouched and ``anchor_status`` becomes ``"drifted"`` — the
    highlight is never dropped. ``insight`` highlights (anchored by ``source_insight_id``, not time)
    pass through unchanged. Returns a NEW dict; the input is not mutated.
    """
    result = dict(highlight)
    if highlight.get("kind") == "insight":
        return result
    start_ms = highlight.get("start_ms")
    end_ms = highlight.get("end_ms")
    if start_ms is None:
        result["anchor_status"] = "drifted"
        return result
    # A moment is a point; a span is a window. Treat end as start for point overlap.
    lo = int(start_ms)
    hi = int(end_ms) if end_ms is not None else lo
    overlapping = [
        s
        for s in segments
        if isinstance(s, dict)
        and s.get("start_ms") is not None
        and s.get("end_ms") is not None
        and int(s["start_ms"]) <= hi
        and int(s["end_ms"]) >= lo
    ]
    if not overlapping:
        result["anchor_status"] = "drifted"
        return result
    result["segment_ids"] = [str(s["segment_id"]) for s in overlapping if s.get("segment_id")]
    char_starts = [int(s["char_start"]) for s in overlapping if s.get("char_start") is not None]
    char_ends = [int(s["char_end"]) for s in overlapping if s.get("char_end") is not None]
    if char_starts:
        result["char_start"] = min(char_starts)
    if char_ends:
        result["char_end"] = max(char_ends)
    result["anchor_status"] = "anchored"
    return result


# --- notes (P2 Capture: free-text notes attached to a highlight, insight or whole episode) ---
#
# A note is plain user text targeting one of three things (``target`` = highlight|insight|episode,
# ``target_id`` = its id/slug). Stored as one ``notes.json`` list, keyed by an opaque ``id``. A
# separate file from highlights so a note can attach independently (e.g. an episode-level note with
# no highlight). The route mints ``id``/``created_at``/``updated_at``.


def get_notes(
    data_dir: Path,
    user_id: str,
    target: str | None = None,
    target_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return saved notes (newest-last), optionally scoped to one ``target``/``target_id``."""
    data = _read(data_dir, user_id, "notes", [])
    if not isinstance(data, list):
        return []
    out = [
        x
        for x in data
        if isinstance(x, dict) and x.get("id") and x.get("target") and x.get("target_id")
    ]
    if target is not None:
        out = [x for x in out if x.get("target") == target]
    if target_id is not None:
        out = [x for x in out if x.get("target_id") == target_id]
    return out


def add_note(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a note (idempotent on ``id``); appended newest-last."""
    nid = item.get("id")
    with _user_lock(data_dir, user_id, "notes"):
        notes = [x for x in get_notes(data_dir, user_id) if x.get("id") != nid]
        notes.append(item)
        _write(data_dir, user_id, "notes", notes)
        return notes


def update_note(
    data_dir: Path, user_id: str, note_id: str, text: str, updated_at: int
) -> dict[str, Any] | None:
    """Edit a note's ``text`` by ``id`` (no-op if absent); return the updated record."""
    with _user_lock(data_dir, user_id, "notes"):
        notes = get_notes(data_dir, user_id)
        updated: dict[str, Any] | None = None
        for rec in notes:
            if rec.get("id") == note_id:
                rec["text"] = text
                rec["updated_at"] = updated_at
                updated = rec
                break
        if updated is not None:
            _write(data_dir, user_id, "notes", notes)
        return updated


def remove_note(data_dir: Path, user_id: str, note_id: str) -> list[dict[str, Any]]:
    """Remove a note by ``id`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "notes"):
        notes = [x for x in get_notes(data_dir, user_id) if x.get("id") != note_id]
        _write(data_dir, user_id, "notes", notes)
        return notes


# --- resurfacing state (P3 #1123): per-highlight {last_surfaced, count} + pacing settings ---
#
# Read-time spaced resurfacing (RFC-101 §5) needs only to remember, per highlight, when it was last
# shown and how many times — the due ladder is computed on read (``app_resurfacing.select_due``).
# ``resurfacing.json`` = {highlight_id: {last_surfaced, count}}; ``resurfacing_settings.json`` =
# {paused}. No scheduler.


def get_resurfacing_state(data_dir: Path, user_id: str) -> dict[str, Any]:
    """Per-highlight resurfacing bookkeeping ({highlight_id: {last_surfaced, count}})."""
    data = _read(data_dir, user_id, "resurfacing", {})
    return data if isinstance(data, dict) else {}


def mark_surfaced(data_dir: Path, user_id: str, highlight_id: str, ts: int) -> dict[str, Any]:
    """Record that a highlight was just surfaced (bumps ``count``, sets ``last_surfaced``)."""
    with _user_lock(data_dir, user_id, "resurfacing"):
        data = get_resurfacing_state(data_dir, user_id)
        prev = data.get(highlight_id)
        count = int(prev.get("count", 0)) + 1 if isinstance(prev, dict) else 1
        rec = {"last_surfaced": int(ts), "count": count}
        data[highlight_id] = rec
        _write(data_dir, user_id, "resurfacing", data)
        return rec


def get_resurfacing_settings(data_dir: Path, user_id: str) -> dict[str, Any]:
    """Pacing settings ({paused}); defaults to not-paused when unset."""
    data = _read(data_dir, user_id, "resurfacing_settings", {})
    paused = bool(data.get("paused")) if isinstance(data, dict) else False
    return {"paused": paused}


def set_resurfacing_settings(data_dir: Path, user_id: str, *, paused: bool) -> dict[str, Any]:
    """Replace the pacing settings; return the stored record."""
    settings = {"paused": bool(paused)}
    _write(data_dir, user_id, "resurfacing_settings", settings)
    return settings
