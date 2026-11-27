"""The clustering's files as one published ``cluster_builds`` build (CRFC-005 §3.7; tracker
s-importer-enrich).

The files, as the clustering writes them today:

- ``search/topic_clusters.json``: themes (``tc:``), each a set of topics;
- ``enrichments/topic_theme_clusters.json``: storylines (``thc:``), each under a super-theme
  (``sth:``) written on the storyline, and the storyline registry;
- ``search/insight_clusters.json``: insight clusters (``ic:``), each a set of insights;
- ``enrichments/clustering_history.jsonl`` and its ``.backfill.jsonl``: one line per clustering
  run's changes.

A group's durable id is allocated once per natural id in ``cluster_ids``. A member is written by
its row: a topic by ``entities.natural_id``, an insight by ``insights.natural_id``. A member the
corpus has no row for is refused and counted per file, never guessed, so this runs after the KG and
GI imports. Episode lists and the quotes copied into the insight-cluster file are not stored:
``item_entities`` and ``insight_support`` give them. A missing file writes no group of its kind.

``storyline_registry.json`` (a copy of the registry) and its ``.seed.json`` are not read.
"""

from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from sqlalchemy import func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from podcast_scraper.db.schema import (
    cluster_builds,
    cluster_events,
    cluster_id_history,
    cluster_ids,
    cluster_members,
    clusters,
    entities,
    insights,
)

THEMES = "search/topic_clusters.json"
STORYLINES = "enrichments/topic_theme_clusters.json"
INSIGHT_CLUSTERS = "search/insight_clusters.json"
HISTORY = ("enrichments/clustering_history.backfill.jsonl", "enrichments/clustering_history.jsonl")

_KINDS = {"tc:": "theme", "thc:": "storyline", "sth:": "super_theme", "ic:": "insight"}
_STORYLINE_KEYS = {
    "members",
    "canonical_label",
    "graph_compound_parent_id",
    "member_count",
    "state",
    "aliases",
    "created_at",
    "last_active_at",
    "super_theme_id",
    "super_theme_label",
    "super_theme_is_long_tail",
    "cluster_type",
}
_THEME_KEYS = {
    "members",
    "canonical_label",
    "graph_compound_parent_id",
    "member_count",
    "cil_alias_target_topic_id",
}
_INSIGHT_KEYS = {"members", "cluster_id", "canonical_insight", "member_count", "episode_ids"}


@dataclass
class GroupingsWritten:
    """What :func:`import_groupings` wrote and refused."""

    build_id: Optional[int] = None
    groups: collections.Counter[str] = field(default_factory=collections.Counter)
    members: int = 0
    history: int = 0
    events: int = 0
    refused: list[tuple[str, str]] = field(default_factory=list)


def kind_of(natural_id: str) -> Optional[str]:
    """The group kind a natural id's prefix names, else ``None``."""
    for prefix, kind in _KINDS.items():
        if natural_id.startswith(prefix):
            return kind
    return None


def _instant(value: Any) -> Optional[datetime]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return None


def _read(root: Path, relpath: str, written: GroupingsWritten) -> Optional[dict[str, Any]]:
    path = root / relpath
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        written.refused.append((relpath, f"unreadable: {exc}"))
        return None
    if isinstance(doc, dict) and "enricher_id" in doc:
        doc = doc.get("data")
    if not isinstance(doc, dict) or not isinstance(doc.get("clusters"), list):
        written.refused.append((relpath, "holds no clusters list"))
        return None
    return doc


def _scalars(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if isinstance(v, (str, int, float, bool)) or v is None}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_by_natural(conn: Connection, table: Any, natural_ids: Iterable[str]) -> dict[str, int]:
    keys = sorted(set(natural_ids))
    found: dict[str, int] = {}
    pk = table.c.entity_id if table is entities else table.c.insight_id
    for start in range(0, len(keys), 5000):
        chunk = keys[start : start + 5000]
        found.update(
            conn.execute(select(table.c.natural_id, pk).where(table.c.natural_id.in_(chunk))).all()
        )
    return found


class _Writer:
    def __init__(self, conn: Connection, build_id: int, written: GroupingsWritten) -> None:
        self.conn = conn
        self.build_id = build_id
        self.written = written
        self.cluster_of: dict[str, int] = {}

    def group_id(self, natural_id: str, kind: str) -> int:
        if natural_id not in self.cluster_of:
            self.conn.execute(
                pg_insert(cluster_ids)
                .values(kind=kind, natural_id=natural_id)
                .on_conflict_do_nothing(index_elements=["natural_id"])
            )
            self.cluster_of[natural_id] = self.conn.execute(
                select(cluster_ids.c.cluster_id).where(cluster_ids.c.natural_id == natural_id)
            ).scalar_one()
        return self.cluster_of[natural_id]

    def group(self, natural_id: str, kind: str, label: str, **columns: Any) -> int:
        cluster_id = self.group_id(natural_id, kind)
        self.conn.execute(
            insert(clusters).values(
                build_id=self.build_id, cluster_id=cluster_id, kind=kind, label=label, **columns
            )
        )
        self.written.groups[kind] += 1
        return cluster_id

    def members(self, cluster_id: int, rows: list[dict[str, Any]]) -> None:
        if rows:
            self.conn.execute(
                insert(cluster_members),
                [dict(build_id=self.build_id, cluster_id=cluster_id, **r) for r in rows],
            )
            self.written.members += len(rows)


def _refuse_missing(written: GroupingsWritten, relpath: str, what: str, missing: list[str]) -> None:
    if missing:
        written.refused.append(
            (
                relpath,
                f"{len(missing)} members name a {what} the corpus has no row for,"
                f" e.g. {missing[0]}",
            )
        )


def _topic_rows(
    members: Iterable[Any], topic_ids: Mapping[str, int], missing: list[str]
) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for m in members:
        if not isinstance(m, dict) or not isinstance(m.get("topic_id"), str):
            continue
        entity_id = topic_ids.get(m["topic_id"])
        if entity_id is None:
            missing.append(m["topic_id"])
            continue
        strength = m.get("strength", m.get("similarity_to_centroid"))
        rows.setdefault(
            entity_id,
            dict(
                entity_id=entity_id,
                insight_id=None,
                strength=strength,
                lift_to_cluster=m.get("lift_to_cluster"),
                is_anchor=bool(m.get("anchor")),
            ),
        )
    return list(rows.values())


def _themes(w: _Writer, doc: Mapping[str, Any], topic_ids: Mapping[str, int]) -> None:
    missing: list[str] = []
    for c in doc["clusters"]:
        natural_id = c.get("graph_compound_parent_id")
        if not isinstance(natural_id, str) or kind_of(natural_id) != "theme":
            w.written.refused.append((THEMES, f"a cluster has no tc: id: {natural_id!r}"))
            continue
        anchor = c.get("cil_alias_target_topic_id")
        cluster_id = w.group(
            natural_id,
            "theme",
            str(c.get("canonical_label") or natural_id),
            member_count=c.get("member_count"),
            anchor_entity_id=topic_ids.get(anchor) if isinstance(anchor, str) else None,
            detail={k: v for k, v in c.items() if k not in _THEME_KEYS},
        )
        w.members(cluster_id, _topic_rows(c.get("members") or [], topic_ids, missing))
    _refuse_missing(w.written, THEMES, "topic", missing)


def _storylines(w: _Writer, doc: Mapping[str, Any], topic_ids: Mapping[str, int]) -> None:
    missing: list[str] = []
    for c in doc["clusters"]:
        sth = c.get("super_theme_id")
        if isinstance(sth, str) and kind_of(sth) == "super_theme" and sth not in w.cluster_of:
            w.group(
                sth,
                "super_theme",
                str(c.get("super_theme_label") or sth),
                detail={"long_tail": bool(c.get("super_theme_is_long_tail"))},
            )
    for c in doc["clusters"]:
        natural_id = c.get("graph_compound_parent_id")
        if not isinstance(natural_id, str) or kind_of(natural_id) != "storyline":
            w.written.refused.append((STORYLINES, f"a cluster has no thc: id: {natural_id!r}"))
            continue
        sth = c.get("super_theme_id")
        anchors = [m for m in c.get("members") or [] if isinstance(m, dict) and m.get("anchor")]
        anchor = anchors[0].get("topic_id") if anchors else None
        cluster_id = w.group(
            natural_id,
            "storyline",
            str(c.get("canonical_label") or natural_id),
            parent_id=w.cluster_of.get(sth) if isinstance(sth, str) else None,
            member_count=c.get("member_count"),
            anchor_entity_id=topic_ids.get(anchor) if isinstance(anchor, str) else None,
            state=c.get("state") if c.get("state") in ("active", "quiet") else None,
            aliases=[str(a) for a in c.get("aliases") or []],
            created_at=_instant(c.get("created_at")),
            last_active_at=_instant(c.get("last_active_at")),
            detail={k: v for k, v in c.items() if k not in _STORYLINE_KEYS},
        )
        w.members(cluster_id, _topic_rows(c.get("members") or [], topic_ids, missing))
    _refuse_missing(w.written, STORYLINES, "topic", missing)


def _insight_clusters(w: _Writer, doc: Mapping[str, Any], insight_ids: Mapping[str, int]) -> None:
    missing: list[str] = []
    for c in doc["clusters"]:
        natural_id = c.get("cluster_id")
        if not isinstance(natural_id, str) or kind_of(natural_id) != "insight":
            w.written.refused.append((INSIGHT_CLUSTERS, f"a cluster has no ic: id: {natural_id!r}"))
            continue
        cluster_id = w.group(
            natural_id,
            "insight",
            str(c.get("canonical_insight") or natural_id),
            member_count=c.get("member_count"),
            detail={k: v for k, v in c.items() if k not in _INSIGHT_KEYS}
            | {"canonical_insight": c.get("canonical_insight")},
        )
        rows: dict[int, dict[str, Any]] = {}
        for m in c.get("members") or []:
            if not isinstance(m, dict) or not isinstance(m.get("insight_id"), str):
                continue
            insight_id = insight_ids.get(m["insight_id"])
            if insight_id is None:
                missing.append(m["insight_id"])
                continue
            rows.setdefault(
                insight_id,
                dict(
                    entity_id=None,
                    insight_id=insight_id,
                    strength=m.get("similarity_to_centroid"),
                    lift_to_cluster=None,
                    is_anchor=False,
                ),
            )
        w.members(cluster_id, list(rows.values()))
    _refuse_missing(w.written, INSIGHT_CLUSTERS, "insight", missing)


def _history(w: _Writer, relpath: str, doc: Mapping[str, Any]) -> None:
    history = doc.get("id_history")
    if not isinstance(history, dict):
        return
    rows: dict[str, dict[str, Any]] = {}
    for old, new in (history.get("successors") or {}).items():
        if isinstance(old, str) and isinstance(new, str) and new:
            rows[old] = dict(old_natural_id=old, new_natural_id=new, change="successor")
    for old in history.get("gone") or []:
        if isinstance(old, str) and old not in rows:
            rows[old] = dict(old_natural_id=old, new_natural_id=None, change="gone")
    if rows:
        w.conn.execute(
            pg_insert(cluster_id_history)
            .values([dict(build_id=w.build_id, **r) for r in rows.values()])
            .on_conflict_do_nothing(index_elements=["build_id", "old_natural_id"])
        )
        w.written.history += len(rows)


def _event_kind(line: Mapping[str, Any]) -> Optional[str]:
    if isinstance(line.get("kind"), str) and line["kind"]:
        return str(line["kind"])
    for changed in (line.get("changes") or {}).values():
        for change in changed if isinstance(changed, list) else []:
            kind = kind_of(change.get("id", "")) if isinstance(change, dict) else None
            if kind is not None:
                return kind
    return None


def _events(conn: Connection, root: Path, written: GroupingsWritten) -> None:
    for relpath in HISTORY:
        path = root / relpath
        if not path.is_file():
            continue
        rows = []
        for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                line = json.loads(raw)
                at = _instant(line.get("ts")) if isinstance(line, dict) else None
            except ValueError as exc:
                written.refused.append((f"{relpath}:{n}", f"unreadable: {exc}"))
                continue
            kind = _event_kind(line)
            if at is None or kind is None:
                written.refused.append((f"{relpath}:{n}", "the line names no time or no kind"))
                continue
            changes = {k: v for k, v in line.items() if k not in ("ts", "kind")}
            rows.append(dict(at=at, kind=kind, changes=changes, source=Path(relpath).name))
        if rows:
            conn.execute(insert(cluster_events), rows)
            written.events += len(rows)


def import_groupings(conn: Connection, corpus_root: Path) -> GroupingsWritten:
    """Write the clustering files under ``corpus_root`` as one published build; nothing when a
    build is already published (a second import)."""
    root = Path(corpus_root)
    written = GroupingsWritten()
    published = conn.execute(
        select(cluster_builds.c.build_id).where(cluster_builds.c.state == "published")
    ).first()
    if published is not None:
        return written
    docs = {rel: _read(root, rel, written) for rel in (THEMES, STORYLINES, INSIGHT_CLUSTERS)}
    present = {rel: doc for rel, doc in docs.items() if doc is not None}
    if present:
        topics = [
            m.get("topic_id")
            for rel in (THEMES, STORYLINES)
            for c in (present.get(rel) or {}).get("clusters", [])
            for m in c.get("members") or []
            if isinstance(m, dict)
        ] + [
            c.get("cil_alias_target_topic_id")
            for c in (present.get(THEMES) or {}).get("clusters", [])
        ]
        topic_ids = _ids_by_natural(conn, entities, [t for t in topics if isinstance(t, str)])
        insight_keys = [
            m.get("insight_id")
            for c in (present.get(INSIGHT_CLUSTERS) or {}).get("clusters", [])
            for m in c.get("members") or []
            if isinstance(m, dict)
        ]
        insight_ids = _ids_by_natural(
            conn, insights, [i for i in insight_keys if isinstance(i, str)]
        )
        storylines = present.get(STORYLINES) or {}
        registry = storylines.get("registry")
        build_id = conn.execute(
            insert(cluster_builds)
            .values(
                rule_version="files",
                inputs={rel: {"sha256": _sha256(root / rel)} for rel in present},
                registry_state=registry if isinstance(registry, dict) else {},
                stats={
                    rel: _scalars({k: v for k, v in doc.items() if k != "registry"})
                    for rel, doc in present.items()
                },
                state="published",
                published_at=func.now(),
            )
            .returning(cluster_builds.c.build_id)
        ).scalar_one()
        written.build_id = build_id
        w = _Writer(conn, build_id, written)
        if THEMES in present:
            _themes(w, present[THEMES], topic_ids)
        if STORYLINES in present:
            _storylines(w, present[STORYLINES], topic_ids)
        if INSIGHT_CLUSTERS in present:
            _insight_clusters(w, present[INSIGHT_CLUSTERS], insight_ids)
        for rel, doc in present.items():
            _history(w, rel, doc)
    _events(conn, root, written)
    return written
