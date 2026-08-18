"""Per-user mutable state as plain files (#1065, RFC-098 §3): playback, queue, library.

Builds on the per-user directory from #1063 (``<data_dir>/users/<id>/``). Each kind is one
JSON file; reads return a default when absent, writes are atomic. No DB — the personal
overlay only; shared corpus artifacts are never touched here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
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


class UserStateUnreadable(RuntimeError):
    """A state file exists but could not be read or parsed.

    Raised only on the read-modify-WRITE path. Readers stay lenient: rendering an empty library
    because a file is briefly unreadable is recoverable and destroys nothing. Overwriting it is
    not — every mutator here persists what it just read, so answering a read error with the empty
    default meant one bad read plus one new capture replaced the user's entire history with a
    single row. "Absent" and "unreadable" must not be the same answer to a writer.
    """


def _read(data_dir: Path, user_id: str, name: str, default: Any, *, strict: bool = False) -> Any:
    """Parsed state, or ``default``. With ``strict``, an unreadable EXISTING file raises."""
    path = _state_path(data_dir, user_id, name)
    if not path.is_file():
        return default  # genuinely absent — the empty default is the truth
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if strict:
            raise UserStateUnreadable(f"{name}.json is unreadable for user {user_id}") from exc
        return default


def _rows_for_update(data_dir: Path, user_id: str, name: str) -> list[dict[str, Any]]:
    """The RAW rows of a list-shaped state file, for read-modify-write.

    Two differences from the public getters, both deliberate:

    * unreadable raises (see :class:`UserStateUnreadable`) instead of silently yielding ``[]``;
    * rows the getters filter out — missing a field a response model requires, or written by a
      different schema version — are KEPT. The getters drop them so the API can still render; a
      mutator that wrote that filtered list back would delete them permanently, as a side effect of
      changing an unrelated row. Dropping on read is a display decision; persisting the drop is
      data loss.

    The invariant every caller must preserve: mutating row X must not rewrite any other row.

    A payload that parses but is the WRONG SHAPE (a dict where the rows should be) raises too. It
    is either hand-corruption or a schema version this build does not understand, and in both cases
    "reset it to empty and write" destroys exactly as much as answering a parse error did. Absent,
    unreadable and unrecognised must not be the same answer to a writer — only the first is safe.
    """
    data = _read(data_dir, user_id, name, [], strict=True)
    if not isinstance(data, list):
        raise UserStateUnreadable(f"{name}.json is not a list for user {user_id}")
    return [x for x in data if isinstance(x, dict)]


def _strings_for_update(data_dir: Path, user_id: str, name: str) -> list[str]:
    """The RAW entries of a list-of-strings state file (``interests``), for read-modify-write.

    :func:`_rows_for_update`'s sibling for the one state file whose rows are bare strings rather
    than objects, with the same two guarantees: unreadable or wrong-shaped raises instead of
    yielding ``[]``, and nothing is dropped on the way to the writer.
    """
    data = _read(data_dir, user_id, name, [], strict=True)
    if not isinstance(data, list):
        raise UserStateUnreadable(f"{name}.json is not a list for user {user_id}")
    return [str(x) for x in data]


def _mapping_for_update(data_dir: Path, user_id: str, name: str) -> dict[str, Any]:
    """The RAW mapping of a dict-shaped state file, for read-modify-write.

    The dict-shaped sibling of :func:`_rows_for_update`, and it exists for the same reason: an
    unreadable ``resurfacing.json`` used to read as ``{}``, so marking ONE highlight surfaced
    persisted a single-key mapping and erased every other highlight's ``{count, last_surfaced}``.
    Losing that is not cosmetic — every count resets to 0 and ``last_seen`` falls back to
    ``created_at``, so the user's whole history floods back in as due at once and the schedule
    they built is gone.

    Wrong-shaped-but-parseable raises for the same reason it does in :func:`_rows_for_update`.
    """
    data = _read(data_dir, user_id, name, {}, strict=True)
    if not isinstance(data, dict):
        raise UserStateUnreadable(f"{name}.json is not a mapping for user {user_id}")
    return data


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
    data_dir: Path,
    user_id: str,
    slug: str,
    position_seconds: float,
    updated_at: int,
    finished: bool = False,
) -> dict[str, Any]:
    """Save the playback position for an episode; return the stored record.

    Reads strictly (:func:`_mapping_for_update`): this is the highest-frequency writer in the whole
    subsystem — clients save position continuously while audio plays — so it has the shortest window
    between "playback.json goes bad" and "playback.json gets overwritten". Resetting to ``{}`` here
    would drop every other episode's resume point on the next position save.
    """
    with _user_lock(data_dir, user_id, "playback"):
        data = _mapping_for_update(data_dir, user_id, "playback")
        rec = {
            "position_seconds": position_seconds,
            "updated_at": updated_at,
            "finished": bool(finished),
        }
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
                    # Absent on records written before the flag existed — an old record is
                    # unfinished, which is what it always behaved as.
                    "finished": bool(rec.get("finished", False)),
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
    """Append one 'opened this episode' event to the user's listen log.

    Canonical event (ADR-119): the shared ``{ts, schema, event_type}`` envelope with
    an ISO-8601 ``ts``. The epoch ``ts`` arg is converted to ISO; the consumers
    (``app_stats._ts_to_date`` / ``app_engagement_series._week_of_ts``) accept BOTH
    epoch and ISO, so pre-existing epoch-ts logs still bucket correctly.
    """
    from ..obs.events import emit_event

    emit_event(
        "listen",
        sink="file",
        path=_events_path(data_dir, user_id),
        ts=datetime.fromtimestamp(int(ts), timezone.utc).isoformat(),
        slug=str(slug),
        feed_id=feed_id,
    )


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


def _upsert_in_place(
    rows: list[dict[str, Any]], item: dict[str, Any], matches: "Callable[[dict[str, Any]], bool]"
) -> None:
    """Replace the first row ``matches`` selects, KEEPING its slot and its ``added_at``.

    "Idempotent on kind+ref" was only half-true: re-saving removed the row and appended the new one,
    so the item jumped to the end of the list and got a fresh ``added_at`` — the route always stamps
    ``time.time()``. Two consequences, both wrong: re-saving something reordered the user's list for
    no reason they took, and (RFC-103) each re-save read as a fresh save in the weekly momentum
    series, inflating engagement with an action that changed nothing.

    ``added_at`` is when the thing was FIRST saved. A re-save is the same save. Contrast
    :func:`add_interest`, which has always got this right — stable position, event only on a new
    follow.
    """
    for i, row in enumerate(rows):
        if matches(row):
            rows[i] = {**item, "added_at": row.get("added_at", item.get("added_at"))}
            return
    rows.append(item)


# --- favorites (polymorphic "saved things": episodes, insights, … keyed by kind+ref) ---


def get_favorites(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Return the user's saved favorites (newest-last as stored); empty when unset."""
    data = _read(data_dir, user_id, "favorites", [])
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict) and x.get("kind") and x.get("ref")]


def add_favorite(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a favorite (idempotent on ``kind``+``ref``); appended newest-last.

    Raw rows, not :func:`get_favorites`: that getter both swallows an unreadable file AND drops
    rows missing ``kind``/``ref``, so reading through it meant one bad read wiped the list and one
    ordinary save silently purged any row a different schema version wrote. Returns the getter
    view, so the API still sees only rows it can render.
    """
    kind, ref = item.get("kind"), item.get("ref")
    with _user_lock(data_dir, user_id, "favorites"):
        favorites = _rows_for_update(data_dir, user_id, "favorites")
        _upsert_in_place(favorites, item, lambda x: (x.get("kind"), x.get("ref")) == (kind, ref))
        _write(data_dir, user_id, "favorites", favorites)
        return get_favorites(data_dir, user_id)


def remove_favorite(data_dir: Path, user_id: str, kind: str, ref: str) -> list[dict[str, Any]]:
    """Remove a favorite by ``kind``+``ref`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "favorites"):
        favorites = [
            x
            for x in _rows_for_update(data_dir, user_id, "favorites")
            if (x.get("kind"), x.get("ref")) != (kind, ref)
        ]
        _write(data_dir, user_id, "favorites", favorites)
        return get_favorites(data_dir, user_id)


# --- interests (personalized discovery; ordered list of cluster ids) ---


def get_interests(data_dir: Path, user_id: str) -> list[str]:
    """Return the user's interest cluster ids (graph_compound_parent_id); empty when unset."""
    data = _read(data_dir, user_id, "interests", [])
    return [str(x) for x in data] if isinstance(data, list) else []


def _write_interests(data_dir: Path, user_id: str, cluster_ids: list[str]) -> list[str]:
    """De-duplicate (order preserved, blanks dropped) and write. CALLER MUST HOLD THE LOCK.

    Split out so the lock is taken in exactly one place per call path. ``_user_lock`` mints a fresh
    ``FileLock`` each call and is not a singleton, so a nested acquire from a mutator that already
    holds the lock would block until timeout rather than re-enter.
    """
    seen: set[str] = set()
    clean: list[str] = []
    for x in cluster_ids:
        s = str(x)
        if s and s not in seen:
            seen.add(s)
            clean.append(s)
    _write(data_dir, user_id, "interests", clean)
    return clean


def set_interests(data_dir: Path, user_id: str, cluster_ids: list[str]) -> list[str]:
    """Replace the user's interests; return the stored list (de-duplicated, order preserved).

    Locked, unlike the queue's equally-replacing ``set_queue``, and the difference matters: this
    file also has read-modify-write writers. ``add_interest`` reads under the lock, so an unlocked
    PUT /interests landing between that read and its write was not last-write-wins —
    ``add_interest`` then persisted a list derived from the PRE-PUT state and the replacement
    vanished silently.
    Symptom: saving the interest picker in one tab while following a person from an entity card in
    another, and watching the picker save revert.
    """
    with _user_lock(data_dir, user_id, "interests"):
        return _write_interests(data_dir, user_id, cluster_ids)


def add_interest(data_dir: Path, user_id: str, token: str) -> list[str]:
    """Follow one interest token (cluster ``tc:``, topic ``topic:`` or person ``person:``).

    Reads strictly: ``get_interests`` answers an unreadable file with ``[]``, so following one topic
    over a bad ``interests.json`` replaced the user's whole profile with that single token — and per
    RFC-103 the interest list is the source of truth the momentum history is derived against.
    """
    with _user_lock(data_dir, user_id, "interests"):
        return _write_interests(
            data_dir, user_id, [*_strings_for_update(data_dir, user_id, "interests"), token]
        )


def remove_interest(data_dir: Path, user_id: str, token: str) -> list[str]:
    """Unfollow one interest token (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "interests"):
        return _write_interests(
            data_dir,
            user_id,
            [x for x in _strings_for_update(data_dir, user_id, "interests") if x != token],
        )


# Follow events — an append-only, timestamped log of interest follows (momentum engagement source,
# RFC-103). The interest LIST above stays the source of truth for "what's followed"; this log is the
# timestamped history a weekly-momentum series needs (the list carries no per-token time). One line
# of JSON ({token, ts}) per newly-followed token, in <data_dir>/users/<id>/interest_events.jsonl.


def record_interest_follow(data_dir: Path, user_id: str, token: str, ts: int) -> None:
    """Append a timestamped follow event for *token* (call only when it was newly followed)."""
    path = data_dir / "users" / user_id / "interest_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"token": str(token), "ts": int(ts)}, ensure_ascii=False) + "\n")


# --- library (subscriptions; list of {feed_id, feed_url?, title?, added_at?}) ---


def get_library(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """Return the user's subscriptions; empty when unset."""
    data = _read(data_dir, user_id, "library", [])
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def add_subscription(data_dir: Path, user_id: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    """Add/replace a subscription by ``feed_id`` (idempotent on feed_id).

    Raw rows for the same reason as :func:`add_favorite` — an unreadable ``library.json`` read as
    ``[]``, so one follow replaced every subscription the user had with that single show.
    """
    feed_id = item.get("feed_id")
    with _user_lock(data_dir, user_id, "library"):
        library = _rows_for_update(data_dir, user_id, "library")
        _upsert_in_place(library, item, lambda x: x.get("feed_id") == feed_id)
        _write(data_dir, user_id, "library", library)
        return get_library(data_dir, user_id)


def remove_subscription(data_dir: Path, user_id: str, feed_id: str) -> list[dict[str, Any]]:
    """Remove a subscription by ``feed_id`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "library"):
        library = [
            x for x in _rows_for_update(data_dir, user_id, "library") if x.get("feed_id") != feed_id
        ]
        _write(data_dir, user_id, "library", library)
        return get_library(data_dir, user_id)


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
        # Raw rows, not get_highlights(): that filters for the response model, and writing the
        # filtered list back would delete a row of a different schema version as a side effect of
        # adding an unrelated one.
        rows = [x for x in _rows_for_update(data_dir, user_id, "highlights") if x.get("id") != hid]
        rows.append(item)
        _write(data_dir, user_id, "highlights", rows)
        return get_highlights(data_dir, user_id)


def update_highlight(
    data_dir: Path, user_id: str, highlight_id: str, fields: dict[str, Any]
) -> dict[str, Any] | None:
    """Merge ``fields`` into a highlight by ``id`` (no-op if absent); return the updated record.

    Used for in-place edits (``color``, ``quote_text``) and persisting a re-anchor. ``id``,
    ``episode_slug`` and ``created_at`` are immutable and cannot be overwritten via ``fields``.
    """
    with _user_lock(data_dir, user_id, "highlights"):
        rows = _rows_for_update(data_dir, user_id, "highlights")
        updated: dict[str, Any] | None = None
        for rec in rows:
            if rec.get("id") == highlight_id:
                rec.update(
                    {k: v for k, v in fields.items() if k not in _IMMUTABLE_HIGHLIGHT_FIELDS}
                )
                updated = rec
                break
        if updated is not None:
            _write(data_dir, user_id, "highlights", rows)
        return updated


def remove_highlight(data_dir: Path, user_id: str, highlight_id: str) -> list[dict[str, Any]]:
    """Remove a highlight by ``id`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "highlights"):
        rows = [
            x
            for x in _rows_for_update(data_dir, user_id, "highlights")
            if x.get("id") != highlight_id
        ]
        _write(data_dir, user_id, "highlights", rows)
        return get_highlights(data_dir, user_id)


def reanchor_highlight(highlight: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-resolve a highlight's positional fields against a fresh transcript.

    ``segments`` are the player's transcript contract (``segments_view.to_contract_segments``):
    ``{id, start, end, text}`` with **start/end in SECONDS**. That is the only segment shape this
    codebase produces, and naming the units here matters — an earlier signature asked for
    ``{segment_id, start_ms, end_ms, char_start, char_end}``, which nothing produced: the pipeline
    carries seconds and TRANSCRIPT-GLOBAL char offsets, while the client stores offsets relative to
    the first touched segment. Feeding one into the other silently rewrites offsets between two
    coordinate systems that share a field name.

    How an anchor is re-established, and why in this order:

    1. **Time selects the candidates.** ``start_ms``/``end_ms`` survive a re-scrape; segment ids do
       not (``to_contract_segments`` mints ``seg_{index}`` from list position, so inserting one
       segment renumbers every later id).
    2. **``quote_text`` decides whether they are RIGHT.** Time alone is not enough: if the audio
       timeline moved — an ad removed, an adfree segment file substituted whose own docstring warns
       it runs "minutes shorter" — the overlapping segment is simply the wrong passage, and
       stamping it ``anchored`` would be a confident lie. So the quote must actually be found in
       the candidates' text.
    3. **Finding the quote also recomputes the offsets**, in the client's coordinate system
       (relative to the first candidate segment), which is what makes them correct rather than
       merely present.

    A highlight is NEVER dropped. Unresolvable → ``anchor_status="drifted"`` with the stored
    positional fields left untouched, so a later re-anchor can recover it if the text returns.
    ``insight`` highlights are anchored by ``source_insight_id`` rather than time and pass through,
    so they receive no ``anchor_status`` at all — a GI re-run that retires an insight leaves the id
    dangling and nothing flags it.

    That is a deliberate gap, and VERIFIED as low-consequence rather than assumed (#34.7): the only
    consumers of ``source_insight_id`` are the client's save TOGGLE — ``savedInsightIds`` and the
    lookup that finds an existing highlight for an insight (``stores/capture.ts``). Nothing renders
    content through the id, because the quote is stored on the highlight at capture. So a dangling
    id degrades a toggle (the insight stops showing as already-saved); it never loses the capture.

    Revisit if any surface ever dereferences the id to READ the insight — at that point a stale id
    becomes a blank or wrong render, and these highlights need a status like the others.

    Returns a NEW dict; the input is not mutated.
    """
    result = dict(highlight)
    if highlight.get("kind") == "insight":
        return result
    start_ms = highlight.get("start_ms")
    if start_ms is None:
        result["anchor_status"] = "drifted"
        return result
    end_ms = highlight.get("end_ms")
    # A moment is a point; a span is a window. Treat end as start for point overlap.
    lo = int(start_ms)
    hi = int(end_ms) if end_ms is not None else lo
    if hi < lo:  # an inverted window would make the overlap test near-vacuous
        lo, hi = hi, lo

    # Overlap is STRICT for a span, matching the client's own selection maths
    # (transcriptCapture.ts: `r.start < sub.char_end && r.end > sub.char_start`). A `<=`/`>=` test
    # pulls in every segment the window merely TOUCHES: a 5s-9s span against 0-5s / 5-9s / 9-20s
    # matched all three, so a highlight of one sentence claimed the whole neighbourhood.
    #
    # A moment is a point, where strict comparison would match nothing at a boundary, so it uses
    # the half-open convention instead — the segment that CONTAINS the instant.
    is_point = lo == hi
    overlapping: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start")
        end = seg.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        seg_lo = int(start * 1000)
        seg_hi = int(end * 1000)
        hit = (seg_lo <= lo < seg_hi) if is_point else (seg_lo < hi and seg_hi > lo)
        if hit:
            overlapping.append(seg)
    if not overlapping:
        result["anchor_status"] = "drifted"
        return result

    quote = str(highlight.get("quote_text") or "").strip()
    if not quote:
        # A `moment` carries no quote — there is nothing to verify against, so time is all we have.
        # Say so honestly in the status rather than claiming a verified anchor.
        result["segment_ids"] = [str(s.get("id")) for s in overlapping if s.get("id")]
        result["anchor_status"] = "time_only"
        return result

    # Search the candidates' concatenated text, joined exactly as the client concatenates the
    # rendered transcript, so a recovered offset means the same thing on both sides.
    joined = "".join(str(s.get("text") or "") for s in overlapping)
    found = joined.find(quote)
    if found < 0:
        # The window exists but no longer contains the quote — the timeline moved under it.
        result["anchor_status"] = "drifted"
        return result

    result["segment_ids"] = [str(s.get("id")) for s in overlapping if s.get("id")]
    result["char_start"] = found
    result["char_end"] = found + len(quote)
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
        # Raw rows, not get_notes() — same reason as add_highlight: the getter filters for the
        # response model, and persisting that filtered list deletes rows it merely could not render.
        rows = [x for x in _rows_for_update(data_dir, user_id, "notes") if x.get("id") != nid]
        rows.append(item)
        _write(data_dir, user_id, "notes", rows)
        return get_notes(data_dir, user_id)


def update_note(
    data_dir: Path, user_id: str, note_id: str, text: str, updated_at: int
) -> dict[str, Any] | None:
    """Edit a note's ``text`` by ``id`` (no-op if absent); return the updated record."""
    with _user_lock(data_dir, user_id, "notes"):
        rows = _rows_for_update(data_dir, user_id, "notes")
        updated: dict[str, Any] | None = None
        for rec in rows:
            if rec.get("id") == note_id:
                rec["text"] = text
                rec["updated_at"] = updated_at
                updated = rec
                break
        if updated is not None:
            _write(data_dir, user_id, "notes", rows)
        return updated


def remove_note(data_dir: Path, user_id: str, note_id: str) -> list[dict[str, Any]]:
    """Remove a note by ``id`` (no-op if absent); return the remaining list."""
    with _user_lock(data_dir, user_id, "notes"):
        rows = [x for x in _rows_for_update(data_dir, user_id, "notes") if x.get("id") != note_id]
        _write(data_dir, user_id, "notes", rows)
        return get_notes(data_dir, user_id)


def remove_notes_for_target(data_dir: Path, user_id: str, target: str, target_id: str) -> int:
    """Drop every note attached to one target; return how many went.

    Notes on a deleted highlight used to survive it server-side. The client pruned them LOCALLY
    (``capture.ts``), so they looked deleted and then RESURRECTED on the next full load — the worst
    of both: the user is told the note is gone, and it is not. The client's own filter is the intent
    the server was failing to implement, so the server implements it.

    Separate from :func:`remove_note` so the sweep takes the notes lock once, not once per note.
    """
    with _user_lock(data_dir, user_id, "notes"):
        rows = _rows_for_update(data_dir, user_id, "notes")
        keep = [
            x
            for x in rows
            if not (x.get("target") == target and str(x.get("target_id")) == str(target_id))
        ]
        if len(keep) == len(rows):
            return 0
        _write(data_dir, user_id, "notes", keep)
        return len(rows) - len(keep)


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
        # Strict read: an unreadable file must not be answered with {} and then persisted as a
        # single-key mapping — that erases every other highlight's schedule.
        data = _mapping_for_update(data_dir, user_id, "resurfacing")
        prev = data.get(highlight_id)
        # A hand-edited or corrupt count must not crash the route or, worse, index the ladder
        # negatively and silently pick the wrong rung.
        try:
            previous_count = int(prev.get("count", 0)) if isinstance(prev, dict) else 0
        except (TypeError, ValueError):
            previous_count = 0
        rec = {"last_surfaced": int(ts), "count": max(previous_count, 0) + 1}
        data[highlight_id] = rec
        _write(data_dir, user_id, "resurfacing", data)
        return rec


def remove_resurfacing_state(data_dir: Path, user_id: str, highlight_id: str) -> None:
    """Drop a highlight's schedule entry (no-op if absent) — the delete cascade for #39.

    ``select_due`` iterates HIGHLIGHTS and looks their state up, so an orphaned entry is never
    read and nothing misbehaves; the file simply grows for ever, one dead key per deleted capture.
    It is also the only per-user file that had no cascade at all, so it was the one place a
    "deleted" thing left a trace — the same shape of promise-vs-reality gap the note cascade fixed.
    """
    with _user_lock(data_dir, user_id, "resurfacing"):
        data = _mapping_for_update(data_dir, user_id, "resurfacing")
        if highlight_id not in data:
            return  # nothing to write — do not rewrite the file for a no-op delete
        data.pop(highlight_id, None)
        _write(data_dir, user_id, "resurfacing", data)


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
