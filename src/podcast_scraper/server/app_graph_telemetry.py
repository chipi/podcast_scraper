"""Graph analytics — append-only per-user event log of graph usage.

Captures **what users do** (node taps, rail navigation, trail loads, re-centres, search) and **how
the graph changes** (ego changes, trail grow/prune, redraws, layout, handoff outcomes), so graph
UX can be measured — e.g. which features get used, where sessions stall, whether the load+trail
actually helps or people re-centre instead.

One ``graph_events`` row per event (ADR-169) — append-only, never rewrites history (same contract
as the listen logs); they go with the account. The viewer batches events and posts them
fire-and-forget; aggregation reads the whole log. Each event carries at least an ``action``; the
rest of the payload is free-form so new event kinds don't need a schema change. Before the
database it was ``<data_dir>/users/<id>/graph_events.jsonl``; the importer reads those.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from podcast_scraper.db import read_with_retry, rw_engine
from podcast_scraper.db.schema import graph_events

#: The file the log was kept in before the database; the importer reads it.
EVENTS_FILE = "graph_events.jsonl"

#: The route's bucket for signed-out sessions; stored as NULL, since it is not an account.
ANON = "anon"


def _owner(user_id: str) -> str | None:
    return None if user_id == ANON else user_id


def parse_line(line: str) -> dict[str, Any] | None:
    """One JSONL line as an event, as the file store read it: a dict with an ``action``."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        rec = json.loads(stripped)
    except ValueError:
        return None
    return rec if isinstance(rec, dict) and rec.get("action") else None


def record_events(data_dir: Path, user_id: str, events: Sequence[Any]) -> int:
    """Append a batch of graph events for *user_id*; returns how many were written.

    Each event must be a dict with a truthy ``action``; anything else (non-dicts, actionless dicts)
    is skipped so one bad row can't drop the batch — hence the permissive ``Sequence[Any]``. One
    insert, so a batch's events stay together in order.
    """
    valid = [e for e in events if isinstance(e, dict) and e.get("action")]
    if not valid:
        return 0
    try:
        with rw_engine().begin() as conn:
            conn.execute(
                graph_events.insert(), [{"user_id": _owner(user_id), "event": ev} for ev in valid]
            )
    except IntegrityError:
        # The id is not an account (a session outliving its deleted account): the batch is
        # dropped, as the route promises the client never sees an error.
        return 0
    return len(valid)


def _read(where: Any = None) -> list[dict[str, Any]]:
    query = select(graph_events.c.event).order_by(
        graph_events.c.user_id.nulls_first(), graph_events.c.id
    )
    if where is not None:
        query = query.where(where)

    def read(engine: Any) -> list[dict[str, Any]]:
        with engine.connect() as conn:
            return [e for e in conn.execute(query).scalars() if isinstance(e, dict)]

    return read_with_retry(read)


def read_events(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """All of one user's graph events (chronological as written); ``anon`` reads the signed-out."""
    owner = _owner(user_id)
    return _read(
        graph_events.c.user_id.is_(None) if owner is None else graph_events.c.user_id == owner
    )


def read_all_events(data_dir: Path) -> list[dict[str, Any]]:
    """Every graph event, signed-out sessions included, concatenated per user in order."""
    return _read()


def users_with_events(data_dir: Path) -> int:
    """How many accounts have recorded any graph event."""

    def read(engine: Any) -> int:
        with engine.connect() as conn:
            return int(
                conn.execute(select(func.count(func.distinct(graph_events.c.user_id)))).scalar()
                or 0
            )

    return read_with_retry(read)


def _stats(xs: list[int]) -> dict[str, float]:
    """min / avg / max / p50 / p95 of a sample (all zero when empty)."""
    if not xs:
        return {"min": 0, "avg": 0.0, "max": 0, "p50": 0, "p95": 0}
    s = sorted(xs)

    def pct(p: float) -> int:
        return s[min(len(s) - 1, int(p * len(s)))]

    return {
        "min": s[0],
        "avg": round(sum(s) / len(s), 1),
        "max": s[-1],
        "p50": pct(0.5),
        "p95": pct(0.95),
    }


def _as_int(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def aggregate(events: Sequence[Any]) -> dict[str, Any]:
    """Summarise a flat list of graph events into usage / size-dynamics / breakage.

    Pure: usage = event counts by action + node taps by kind; size = min/avg/max/p50/p95 of the
    per-redraw node, edge and trail-size samples; breakage = ``graph_broke`` count by reason.
    """
    by_action: dict[str, int] = {}
    node_taps: dict[str, int] = {}
    break_reasons: dict[str, int] = {}
    nodes: list[int] = []
    edges: list[int] = []
    trail: list[int] = []
    for e in events:
        action = str(e.get("action") or "")
        if not action:
            continue
        by_action[action] = by_action.get(action, 0) + 1
        if action == "graph_node_tap":
            k = str(e.get("kind") or "unknown")
            node_taps[k] = node_taps.get(k, 0) + 1
        elif action == "graph_redraw":
            for src, dst in (
                (e.get("nodes"), nodes),
                (e.get("edges"), edges),
                (e.get("trail_size"), trail),
            ):
                n = _as_int(src)
                if n is not None:
                    dst.append(n)
        elif action == "graph_broke":
            r = str(e.get("reason") or "unknown")
            break_reasons[r] = break_reasons.get(r, 0) + 1
    return {
        "total_events": sum(by_action.values()),
        "by_action": by_action,
        "node_taps_by_kind": node_taps,
        "size": {
            "samples": len(nodes),
            "nodes": _stats(nodes),
            "edges": _stats(edges),
            "trail": _stats(trail),
        },
        "breakage": {"count": by_action.get("graph_broke", 0), "by_reason": break_reasons},
    }


def sessions(events: Sequence[Any]) -> list[dict[str, Any]]:
    """Group events by ``session_id`` → one summary per session, most-recent first.

    Each summary carries the user, start/end timestamps, event count and the min/max graph size
    seen (from the ``graph_redraw`` samples) — enough to pick a session to inspect or replay.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        sid = e.get("session_id")
        if isinstance(sid, str) and sid:
            grouped.setdefault(sid, []).append(e)
    out: list[dict[str, Any]] = []
    for sid, evs in grouped.items():
        evs.sort(key=lambda e: _as_int(e.get("ts")) or 0)
        node_counts = [
            n
            for n in (_as_int(e.get("nodes")) for e in evs if e.get("action") == "graph_redraw")
            if n is not None
        ]
        out.append(
            {
                "session_id": sid,
                "user_id": next(
                    (str(e.get("user_id")) for e in evs if e.get("user_id")), "unknown"
                ),
                "started": _as_int(evs[0].get("ts")) or 0,
                "ended": _as_int(evs[-1].get("ts")) or 0,
                "count": len(evs),
                "size_min": min(node_counts) if node_counts else 0,
                "size_max": max(node_counts) if node_counts else 0,
            }
        )
    out.sort(key=lambda s: -int(s["started"]))
    return out


def session_events(events: Sequence[Any], session_id: str) -> list[dict[str, Any]]:
    """The one session's events, ordered by timestamp (for the step-by-step timeline + replay)."""
    evs = [e for e in events if isinstance(e, dict) and e.get("session_id") == session_id]
    evs.sort(key=lambda e: _as_int(e.get("ts")) or 0)
    return evs


__all__ = [
    "record_events",
    "read_events",
    "read_all_events",
    "aggregate",
    "sessions",
    "session_events",
]
