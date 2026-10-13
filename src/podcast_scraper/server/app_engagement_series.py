"""Engagement event series for the momentum layer (RFC-103 Phase 2; ADR-163 §5).

The *engagement* event source (complements the enricher's ``content_series``): rolls the per-user
**timestamped** logs into per-entity weekly scores, keyed by a namespaced ``(kind, entity_id)``.

**Distinct listeners, not events** (ADR-163 §5, advisor review 2026-10-11). Each listener counts
ONCE per signal per entity per week, weighted by what the signal means — so one listener's three
highlights are one "captured", not three, and a small, noisy user base cannot inflate an item by
repeating an action. The signals (``DEFAULT_WEIGHTS``):

* ``finished`` — reached 75% or 95% of an episode (``playback_events.jsonl``)
* ``captured`` — a highlight, or a note on the item (``highlights`` / ``notes``)
* ``saved`` — favourited (``favorites`` ``added_at``)
* ``opened`` — opened an episode / its show (``listen_events.jsonl``)
* ``subscribed`` — followed a show (``library`` ``added_at``)
* ``followed`` — followed a topic, theme, storyline or person (``interest_events.jsonl``)

Each entity also carries ``listeners`` (distinct users over the series), which floors use instead
of event counts. Corpus-wide aggregate by default; ``user_id`` yields one user's ("mine") series;
``since_week`` restricts the tally to a window (so ``listeners`` is distinct within it). Output
mirrors ``content_series`` — ``window_weeks`` + per-entity sparse ``weekly_counts``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from podcast_scraper.server import app_user_state

#: Weight of each signal, counted once per listener per entity per week (ADR-163 §5).
DEFAULT_WEIGHTS: dict[str, int] = {
    "finished": 3,
    "captured": 3,
    "saved": 2,
    "opened": 1,
    "subscribed": 1,
    "followed": 1,
}
#: Playback milestones that count as finishing an episode.
_FINISHED_MILESTONES = frozenset({75, 95})

# One user's signals: (kind, entity_id) → {ISO-week: {signal}}
_Signals = dict[tuple[str, str], dict[str, set[str]]]


def _week_of_ts(ts: Any) -> str | None:
    """Epoch seconds OR ISO-8601 string → ISO year-week ``YYYY-Www`` (``None`` on bad input).

    Accepts both the legacy epoch-int ts and the canonical ISO-8601 string (ADR-119
    emit_event envelope) so a mixed old/new listen log buckets correctly.
    """
    dt: datetime | None = None
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except (ValueError, OverflowError, OSError, TypeError):
        try:
            dt = datetime.fromisoformat(str(ts)).astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None
    iso = dt.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def _iter_user_ids(data_dir: Path, user_id: str | None) -> list[str]:
    """User ids to aggregate over: just ``user_id`` (mine) or every user dir (corpus-wide)."""
    if user_id is not None:
        return [user_id]
    users_dir = data_dir / "users"
    if not users_dir.is_dir():
        return []
    return sorted(p.name for p in users_dir.iterdir() if p.is_dir())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read an append-only JSONL log into dict records; skip blank/corrupt lines."""
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _mark(sig: _Signals, kind: str, entity_id: str, week: str | None, signal: str) -> None:
    """Note that this user gave ``signal`` for ``(kind, entity_id)`` in ``week`` (idempotent)."""
    if week and entity_id:
        sig[(kind, entity_id)][week].add(signal)


def _kind_of_token(token: str) -> str | None:
    """Interest-token prefix → entity kind (``None`` for unrecognized prefixes)."""
    for prefix, kind in (("person:", "person"), ("topic:", "topic"), ("thc:", "storyline")):
        if token.startswith(prefix):
            return kind
    # "theme", matching the /trending kind enum — engagement series join momentum on this
    # string, so the two must move together or the join silently yields nothing for `tc:`.
    return "theme" if token.startswith("tc:") else None


def _user_signals(data_dir: Path, uid: str) -> _Signals:
    """Every signal one user gave, by entity and week — each at most once per (entity, week)."""
    sig: _Signals = defaultdict(lambda: defaultdict(set))
    udir = data_dir / "users" / uid
    for ev in _read_jsonl(udir / "listen_events.jsonl"):
        wk = _week_of_ts(ev.get("ts"))
        _mark(sig, "episode", str(ev.get("slug") or ""), wk, "opened")
        if ev.get("feed_id"):
            _mark(sig, "show", str(ev.get("feed_id")), wk, "opened")
    for ev in _read_jsonl(udir / "playback_events.jsonl"):
        try:
            finished = int(ev.get("milestone") or 0) in _FINISHED_MILESTONES
        except (TypeError, ValueError):
            finished = False
        if finished:
            _mark(sig, "episode", str(ev.get("slug") or ""), _week_of_ts(ev.get("ts")), "finished")
    for sub in app_user_state.get_library(data_dir, uid):
        _mark(
            sig,
            "show",
            str(sub.get("feed_id") or ""),
            _week_of_ts(sub.get("added_at")),
            "subscribed",
        )
    for fav in app_user_state.get_favorites(data_dir, uid):
        kind, ref = str(fav.get("kind") or ""), str(fav.get("ref") or "")
        if kind in ("episode", "insight", "person", "topic"):
            _mark(sig, kind, ref, _week_of_ts(fav.get("added_at")), "saved")
    slug_of_highlight: dict[str, str] = {}
    for h in app_user_state.get_highlights(data_dir, uid):
        slug = str(h.get("episode_slug") or "")
        slug_of_highlight[str(h.get("id"))] = slug
        _mark(sig, "episode", slug, _week_of_ts(h.get("created_at")), "captured")
    for n in app_user_state.get_notes(data_dir, uid):
        target, tid = str(n.get("target") or ""), str(n.get("target_id") or "")
        wk = _week_of_ts(n.get("created_at"))
        if target == "highlight":
            _mark(sig, "episode", slug_of_highlight.get(tid, ""), wk, "captured")
        elif target in ("episode", "show", "topic", "person", "storyline"):
            _mark(sig, target, tid, wk, "captured")
    for ev in _read_jsonl(udir / "interest_events.jsonl"):
        token = str(ev.get("token") or "")
        tkind = _kind_of_token(token)
        if tkind:
            _mark(sig, tkind, token, _week_of_ts(ev.get("ts")), "followed")
    return sig


def _week_axis(active_weeks: set[str]) -> list[str]:
    """Contiguous ISO-week axis spanning the observed weeks (empty when none)."""
    if not active_weeks:
        return []

    def _monday(wk: str) -> datetime:
        year, week = wk.split("-W")
        return datetime.fromisocalendar(int(year), int(week), 1).replace(tzinfo=timezone.utc)

    lo, hi = min(map(_monday, active_weeks)), max(map(_monday, active_weeks))
    axis: list[str] = []
    seen: set[str] = set()
    cur = lo
    while cur <= hi:
        iso = cur.isocalendar()
        wk = f"{iso.year:04d}-W{iso.week:02d}"
        if wk not in seen:
            seen.add(wk)
            axis.append(wk)
        cur += timedelta(days=7)
    return axis


def engagement_series(
    data_dir: Path,
    *,
    user_id: str | None = None,
    since_week: str | None = None,
    weights: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Per-entity weekly engagement scores (corpus-wide, or one user for ``scope=mine``).

    Returns ``{window_weeks, entities: [{entity_id, kind, weekly_counts, total, listeners}]}``.
    ``weekly_counts[w]`` = the sum, over listeners, of the weights of the DISTINCT signals each gave
    the entity in week ``w``; ``listeners`` = distinct users who gave any signal (within
    ``since_week``…, when set).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    scores: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    who: dict[tuple[str, str], set[str]] = defaultdict(set)
    active: set[str] = set()
    for uid in _iter_user_ids(data_dir, user_id):
        for key, by_week in _user_signals(data_dir, uid).items():
            for week, signals in by_week.items():
                if since_week is not None and week < since_week:
                    continue
                active.add(week)
                scores[key][week] += sum(int(w.get(s, 0)) for s in signals)
                who[key].add(uid)
    entities: list[dict[str, Any]] = [
        {
            "entity_id": entity_id,
            "kind": kind,
            "weekly_counts": dict(sorted(counts.items())),
            "total": sum(counts.values()),
            "listeners": len(who[(kind, entity_id)]),
        }
        for (kind, entity_id), counts in scores.items()
    ]
    entities.sort(key=lambda r: (str(r["kind"]), -int(r["total"]), str(r["entity_id"])))
    return {"window_weeks": _week_axis(active), "entities": entities}


__all__ = ["DEFAULT_WEIGHTS", "engagement_series"]
