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


def record_events(data_dir: Path, user_id: str, events: Sequence[dict[str, Any]]) -> int:
    """Append a batch of graph events for *user_id*; returns how many were written.

    Each event must be a dict with a truthy ``action``; malformed entries are skipped so one bad
    row can't drop the batch.
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


__all__ = ["record_events", "read_events"]
