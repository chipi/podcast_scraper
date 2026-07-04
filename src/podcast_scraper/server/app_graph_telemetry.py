"""Graph analytics — append-only per-user event log of graph usage.

Captures **what users do** (node taps, rail navigation, trail loads, re-centres, search) and **how
the graph changes** (ego changes, trail grow/prune, redraws, layout, handoff outcomes), so graph
UX can be measured — e.g. which features get used, where sessions stall, whether the load+trail
actually helps or people re-centre instead.

One JSON line per event in ``<data_dir>/users/<id>/graph_events.jsonl`` — append-only, never
rewrites history (same contract as the listen + ranking logs). The viewer batches events and posts
them fire-and-forget; aggregation reads the whole log offline. Each event carries at least an
``action``; the rest of the payload is free-form so new event kinds don't need a schema change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def _events_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / "graph_events.jsonl"


def record_events(data_dir: Path, user_id: str, events: Sequence[Any]) -> int:
    """Append a batch of graph events for *user_id*; returns how many were written.

    Each event must be a dict with a truthy ``action``; anything else (non-dicts, actionless dicts)
    is skipped so one bad row can't drop the batch — hence the permissive ``Sequence[Any]``.
    """
    valid = [e for e in events if isinstance(e, dict) and e.get("action")]
    if not valid:
        return 0
    path = _events_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for ev in valid:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return len(valid)


def read_events(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """All of one user's graph events (chronological as written); skips corrupt lines."""
    path = _events_path(data_dir, user_id)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("action"):
            out.append(rec)
    return out


def read_all_events(data_dir: Path) -> list[dict[str, Any]]:
    """Every user's graph events, concatenated (each user's own order preserved)."""
    users_dir = data_dir / "users"
    if not users_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for user_dir in sorted(users_dir.iterdir()):
        if user_dir.is_dir():
            out.extend(read_events(data_dir, user_dir.name))
    return out


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
