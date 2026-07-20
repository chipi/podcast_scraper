"""Engagement event series for the momentum layer (RFC-103 Phase 2).

The *engagement* event source (complements the enricher's ``content_series``): rolls the per-user
**timestamped** engagement logs into per-entity weekly counts, keyed by a namespaced ``(kind,
entity_id)``. Every saveable kind has a timestamped signal:

* episode — opens (``listen_events.jsonl``) + clicks (``ranking_events.jsonl``)
* show — opens (by ``feed_id``) + subscribes (``library`` ``added_at``)
* insight / person / topic — saves (``favorites`` ``added_at``)
* topic / cluster (``tc:``) / storyline (``thc:``) / person — follows (``interest_events.jsonl``)

Corpus-wide aggregate (all users) by default; a single ``user_id`` yields the per-user ("mine")
series. Output mirrors ``content_series`` — ``window_weeks`` (contiguous ISO-week axis) + per-entity
sparse ``weekly_counts`` — so the momentum capability runs one EWMA over either source.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from podcast_scraper.server import app_user_state

# Accumulator: (kind, entity_id) → {ISO-week: count}
_Acc = dict[tuple[str, str], dict[str, int]]


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


def _bump(acc: _Acc, active: set[str], kind: str, entity_id: str, week: str | None) -> None:
    """Add one engagement event for ``(kind, entity_id)`` in ``week`` (no-op on a bad week)."""
    if not week or not entity_id:
        return
    active.add(week)
    acc[(kind, entity_id)][week] += 1


def _kind_of_token(token: str) -> str | None:
    """Interest-token prefix → entity kind (``None`` for unrecognized prefixes)."""
    for prefix, kind in (("person:", "person"), ("topic:", "topic"), ("thc:", "storyline")):
        if token.startswith(prefix):
            return kind
    return "cluster" if token.startswith("tc:") else None


def _tally_user(data_dir: Path, uid: str, acc: _Acc, active: set[str]) -> None:
    """Fold one user's opens, clicks, subscribes, saves, and follows into the tallies in place."""
    udir = data_dir / "users" / uid
    # opens (episode + show) + clicks (episode)
    for ev in _read_jsonl(udir / "listen_events.jsonl"):
        wk = _week_of_ts(ev.get("ts"))
        _bump(acc, active, "episode", str(ev.get("slug") or ""), wk)
        if ev.get("feed_id"):
            _bump(acc, active, "show", str(ev.get("feed_id")), wk)
    for ev in _read_jsonl(udir / "ranking_events.jsonl"):
        if ev.get("kind") == "click":
            _bump(acc, active, "episode", str(ev.get("slug") or ""), _week_of_ts(ev.get("ts")))
    # subscribes (show) — library carries added_at
    for sub in app_user_state.get_library(data_dir, uid):
        _bump(acc, active, "show", str(sub.get("feed_id") or ""), _week_of_ts(sub.get("added_at")))
    # saves (episode / insight / person / topic) — favorites carry added_at
    for fav in app_user_state.get_favorites(data_dir, uid):
        kind, ref = str(fav.get("kind") or ""), str(fav.get("ref") or "")
        if kind in ("episode", "insight", "person", "topic"):
            _bump(acc, active, kind, ref, _week_of_ts(fav.get("added_at")))
    # follows (topic / cluster / storyline / person) — interest_events log
    for ev in _read_jsonl(udir / "interest_events.jsonl"):
        token = str(ev.get("token") or "")
        tkind = _kind_of_token(token)
        if tkind:
            _bump(acc, active, tkind, token, _week_of_ts(ev.get("ts")))


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


def engagement_series(data_dir: Path, *, user_id: str | None = None) -> dict[str, Any]:
    """Per-entity weekly engagement counts (corpus-wide, or one user for ``scope=mine``).

    Returns ``{window_weeks, entities: [{entity_id, kind, weekly_counts, total}]}`` — the momentum
    layer's engagement event source, parallel to the enricher's content series.
    """
    acc: _Acc = defaultdict(lambda: defaultdict(int))
    active: set[str] = set()
    for uid in _iter_user_ids(data_dir, user_id):
        _tally_user(data_dir, uid, acc, active)
    entities: list[dict[str, Any]] = [
        {
            "entity_id": entity_id,
            "kind": kind,
            "weekly_counts": dict(sorted(counts.items())),
            "total": sum(counts.values()),
        }
        for (kind, entity_id), counts in acc.items()
    ]
    entities.sort(key=lambda r: (str(r["kind"]), -int(r["total"]), str(r["entity_id"])))
    return {"window_weeks": _week_axis(active), "entities": entities}


__all__ = ["engagement_series"]
