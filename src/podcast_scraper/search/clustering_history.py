"""What each storyline and theme run did, kept over time (RFC-128, #2349).

Both clustering artifacts are rewritten on every run, so on their own they cannot say how the
storylines and themes moved: how many there were last month, which were created, renamed, went
quiet or were absorbed. Each run appends one line here:

- ``enrichments/clustering_history.jsonl`` — one row per storyline run (``kind: storyline``,
  ``topic_theme_clusters``) and per theme build (``kind: theme``, ``search/topic_clusters.json``):
  its time, counts, health and this run's changes by id and name;
- ``enrichments/clustering_history.backfill.jsonl`` — storyline rows from before there were runs to
  record, one per month, from the history seed's replay (``search.storyline_history``); rewritten
  whole each time the seed is applied, so it never duplicates.

Read by ``GET /api/corpus/clustering/health`` and the viewer dashboard. Appending never fails a run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

HISTORY = Path("enrichments") / "clustering_history.jsonl"
BACKFILL = Path("enrichments") / "clustering_history.backfill.jsonl"
SEED = Path("enrichments") / "storyline_registry.seed.json"
STORYLINES = Path("enrichments") / "topic_theme_clusters.json"


def append(corpus_root: Path, row: Mapping[str, Any]) -> None:
    """Append one run's row. Best-effort: a write failure is logged, never raised."""
    path = Path(corpus_root) / HISTORY
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning("clustering history: could not append to %s: %s", path, exc)


def write_backfill(corpus_root: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Replace the storyline backfill (one row per replayed month)."""
    path = Path(corpus_root) / BACKFILL
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(r), sort_keys=True) + "\n" for r in rows)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a torn last line from a crash mid-append; the rest stands
        if isinstance(row, dict):
            out.append(row)
    return out


def read(corpus_root: Path, kind: str) -> list[dict[str, Any]]:
    """Every row of ``kind`` (``storyline`` or ``theme``), oldest first: the backfill, then the
    runs recorded since (a backfill row at or after the first recorded run is dropped)."""
    return read_files(Path(corpus_root) / HISTORY, Path(corpus_root) / BACKFILL, kind)


def read_files(history: Path, backfill: Path, kind: str) -> list[dict[str, Any]]:
    """:func:`read` from explicit files (a server route passes paths it has checked)."""
    runs = [r for r in _rows(history) if r.get("kind") == kind]
    runs.sort(key=lambda r: float(r.get("ts") or 0))
    first = float(runs[0].get("ts") or 0) if runs else float("inf")
    back = [r for r in _rows(backfill) if r.get("kind") == kind and float(r.get("ts") or 0) < first]
    back.sort(key=lambda r: float(r.get("ts") or 0))
    return back + runs


def seed_status(corpus_root: Path) -> dict[str, Any]:
    """``{"state": "none" | "pending" | "taken", "id"}`` of the storyline history seed."""
    return seed_status_files(Path(corpus_root) / SEED, Path(corpus_root) / STORYLINES)


def seed_status_files(seed_file: Path, storylines_file: Path) -> dict[str, Any]:
    """:func:`seed_status` from explicit files (a server route passes paths it has checked)."""
    try:
        seed = json.loads(seed_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seed = None
    taken: Optional[str] = None
    try:
        raw = json.loads(storylines_file.read_text(encoding="utf-8"))
        data = raw.get("data", raw) if isinstance(raw, dict) else {}
        registry = data.get("registry") if isinstance(data, dict) else None
        if isinstance(registry, dict):
            taken = registry.get("seeded_from")
    except (OSError, ValueError):
        pass
    if isinstance(seed, dict) and seed.get("id") and seed["id"] != taken:
        return {"state": "pending", "id": seed["id"]}
    if taken:
        return {"state": "taken", "id": taken}
    return {"state": "none", "id": None}


def _by_id(clusters: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(c.get("graph_compound_parent_id")): c
        for c in clusters
        if isinstance(c, Mapping) and c.get("graph_compound_parent_id")
    }


def changes(
    before: Iterable[Mapping[str, Any]],
    after: Iterable[Mapping[str, Any]],
    history_before: Mapping[str, Any] | None,
    history_after: Mapping[str, Any] | None,
) -> dict[str, list[dict[str, str]]]:
    """This run's changes, by id and name: created, renamed, went quiet, woke, absorbed, gone."""
    old, new = _by_id(before), _by_id(after)
    name = {**{k: str(v.get("canonical_label") or k) for k, v in old.items()}}
    name.update({k: str(v.get("canonical_label") or k) for k, v in new.items()})
    succ_before = {
        str(k): str(v) for k, v in ((history_before or {}).get("successors") or {}).items()
    }
    succ_after = {
        str(k): str(v) for k, v in ((history_after or {}).get("successors") or {}).items()
    }
    gone_before = set((history_before or {}).get("gone") or [])
    gone_after = set((history_after or {}).get("gone") or [])
    out: dict[str, list[dict[str, str]]] = {
        "created": [{"id": i, "name": name[i]} for i in sorted(set(new) - set(old))],
        "renamed": [
            {"id": i, "from": str(old[i].get("canonical_label")), "to": name[i]}
            for i in sorted(set(new) & set(old))
            if old[i].get("canonical_label") != new[i].get("canonical_label")
        ],
        "quiet": [
            {"id": i, "name": name[i]}
            for i in sorted(set(new) & set(old))
            if new[i].get("state") == "quiet" and old[i].get("state") != "quiet"
        ],
        "woke": [
            {"id": i, "name": name[i]}
            for i in sorted(set(new) & set(old))
            if new[i].get("state") == "active" and old[i].get("state") == "quiet"
        ],
        "absorbed": [
            {"id": i, "name": name.get(i, i), "into": v, "into_name": name.get(v, v)}
            for i, v in sorted(succ_after.items())
            if succ_before.get(i) != v and i in old
        ],
        "gone": [{"id": i, "name": name.get(i, i)} for i in sorted(gone_after - gone_before)],
    }
    return out


__all__ = [
    "BACKFILL",
    "HISTORY",
    "append",
    "changes",
    "read",
    "read_files",
    "seed_status",
    "seed_status_files",
    "write_backfill",
]
