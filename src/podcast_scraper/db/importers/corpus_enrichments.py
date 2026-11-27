"""The enrichers' output files as one published ``enrichment_builds`` build per enricher (CRFC-005
§3.6, §3.7; tracker s-importer-enrich).

Every enricher writes an envelope (``enricher_id``, ``enricher_version``, ``computed_at``,
``status``, ``data``): corpus-scope ones under ``enrichments/``, episode-scope ones beside each
item's metadata under ``metadata/enrichments/<stem>.<enricher>.json``. An envelope's ``data`` is
written whole as an ``enrichment_outputs`` row, except for the parts read per entity or per pair:

- ``person_web`` and ``org_web``: one ``entity`` row per person or organisation, the rest as the
  corpus row;
- ``topic_similarity`` (directed, a to b), ``guest_coappearance`` and ``topic_cooccurrence_corpus``
  (symmetric, once with ``a < b``): ``entity_pairs`` rows, the rest as the corpus row.

An episode-scope output is read only beside the metadata the item import selected (the newest run
per episode), and written under the item its ``data`` names. A row naming an item or entity the
corpus has no row for is refused and counted per file, never guessed. An envelope whose status is
not ``ok`` is refused. Not read here: ``topic_theme_clusters`` (``importers.corpus_groupings``),
``temporal_velocity`` (``entity_weeks``, with the projections), the run summary, the registry
copies, the ``*.delta_cursor.json`` and ``*.vectors_cache.json`` files, and the raw page and image
directories.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import Connection

from podcast_scraper.db.schema import (
    content_items,
    enrichment_builds,
    enrichment_outputs,
    entities,
    entity_pairs,
)

CORPUS, EPISODE, ENTITY = "corpus", "episode", "entity"
_NOT_OUTPUTS = {
    "topic_theme_clusters.json",
    "temporal_velocity.json",
    "run_summary.json",
    "storyline_registry.json",
    "storyline_registry.seed.json",
}
_PER_ENTITY = {"person_web": ("persons", "person_id"), "org_web": ("orgs", "org_id")}
_SUFFIX = ".metadata.json"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass
class EnrichmentsWritten:
    """What :func:`import_enrichments` wrote and refused."""

    builds: int = 0
    outputs: collections.Counter[str] = field(default_factory=collections.Counter)
    pairs: collections.Counter[str] = field(default_factory=collections.Counter)
    refused: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _Envelope:
    relpath: str
    enricher_id: str
    version: Optional[str]
    computed_at: Optional[datetime]
    data: Any


def _read(root: Path, path: Path, written: EnrichmentsWritten) -> Optional[_Envelope]:
    relpath = path.relative_to(root).as_posix()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        written.refused.append((relpath, f"unreadable: {exc}"))
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("enricher_id"), str):
        written.refused.append((relpath, "not an enricher output"))
        return None
    if doc.get("status") != "ok":
        written.refused.append((relpath, f"status {doc.get('status')!r}"))
        return None
    computed = doc.get("computed_at")
    return _Envelope(
        relpath=relpath,
        enricher_id=doc["enricher_id"],
        version=str(doc["enricher_version"]) if doc.get("enricher_version") is not None else None,
        computed_at=(
            datetime.fromisoformat(computed.replace("Z", "+00:00"))
            if isinstance(computed, str) and computed
            else None
        ),
        data=doc.get("data"),
    )


def _corpus_files(root: Path) -> list[Path]:
    return [
        p
        for p in sorted((root / "enrichments").glob("*.json"))
        if p.name not in _NOT_OUTPUTS
        and not p.name.endswith((".delta_cursor.json", ".vectors_cache.json"))
    ]


def _episode_files(root: Path, metadata_relpaths: Iterable[str]) -> list[Path]:
    found = []
    for relpath in sorted(set(metadata_relpaths)):
        meta = root / relpath
        if not meta.name.endswith(_SUFFIX):
            continue
        stem = meta.name[: -len(_SUFFIX)]
        found.extend(sorted((meta.parent / "enrichments").glob(f"{stem}.*.json")))
    return found


def _lookup(conn: Connection, column: Any, pk: Any, keys: Iterable[str]) -> dict[str, int]:
    wanted = sorted(set(keys))
    found: dict[str, int] = {}
    for start in range(0, len(wanted), 5000):
        chunk = wanted[start : start + 5000]
        found.update(conn.execute(select(column, pk).where(column.in_(chunk))).all())
    return found


def _entity_ids(conn: Connection, keys: Iterable[str]) -> dict[str, int]:
    return _lookup(conn, entities.c.natural_id, entities.c.entity_id, keys)


def _missing(written: EnrichmentsWritten, relpath: str, what: str, missing: list[str]) -> None:
    if missing:
        written.refused.append(
            (
                relpath,
                f"{len(missing)} rows name {what} the corpus has no row for, e.g. {missing[0]}",
            )
        )


def _pairs(env: _Envelope) -> tuple[list[tuple[str, str, float, dict[str, Any]]], dict[str, Any]]:
    """The envelope's pairs as ``(a, b, score, detail)`` natural ids, and the rest of its data."""
    data = env.data if isinstance(env.data, dict) else {}
    pairs: list[tuple[str, str, float, dict[str, Any]]]
    if env.enricher_id == "topic_similarity":
        pairs = [
            (t["topic_id"], k["topic_id"], float(k["similarity"]), {})
            for t in data.get("topics") or []
            for k in t.get("top_k") or []
            if isinstance(t, dict) and isinstance(k, dict) and k.get("similarity") is not None
        ]
        return pairs, {k: v for k, v in data.items() if k != "topics"}
    if env.enricher_id == "guest_coappearance":
        pairs = [
            (p["person_a_id"], p["person_b_id"], float(p["episode_count"]), {})
            for p in data.get("pairs") or []
            if isinstance(p, dict)
        ]
        return pairs, {k: v for k, v in data.items() if k != "pairs"}
    if env.enricher_id == "topic_cooccurrence_corpus":
        measures = (
            "episode_count",
            "topic_a_episode_count",
            "topic_b_episode_count",
            "lift",
            "pmi",
            "npmi",
        )
        pairs = [
            (p["topic_a_id"], p["topic_b_id"], float(p["lift"]), {m: p.get(m) for m in measures})
            for p in data.get("pairs") or []
            if isinstance(p, dict) and p.get("lift") is not None
        ]
        return pairs, {k: v for k, v in data.items() if k != "pairs"}
    return [], data


def _write_pairs(
    conn: Connection, build_id: int, env: _Envelope, written: EnrichmentsWritten
) -> dict[str, Any]:
    pairs, rest = _pairs(env)
    if not pairs:
        return rest
    ids = _entity_ids(conn, [n for a, b, _, _ in pairs for n in (a, b)])
    symmetric = env.enricher_id != "topic_similarity"
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    missing: list[str] = []
    for a, b, score, detail in pairs:
        ia, ib = ids.get(a), ids.get(b)
        if ia is None or ib is None:
            missing.append(a if ia is None else b)
            continue
        if ia == ib:
            continue
        if symmetric and ia > ib:
            ia, ib = ib, ia
        rows.setdefault(
            (ia, ib),
            dict(
                build_id=build_id,
                enricher_id=env.enricher_id,
                a=ia,
                b=ib,
                score=score,
                detail=detail,
            ),
        )
    _missing(written, env.relpath, "an entity", missing)
    for start in range(0, len(rows), 5000):
        conn.execute(insert(entity_pairs), list(rows.values())[start : start + 5000])
    written.pairs[env.enricher_id] += len(rows)
    return rest


def _write_entities(
    conn: Connection, build_id: int, env: _Envelope, written: EnrichmentsWritten
) -> dict[str, Any]:
    data = env.data if isinstance(env.data, dict) else {}
    listed, key = _PER_ENTITY[env.enricher_id]
    records = [
        r for r in data.get(listed) or [] if isinstance(r, dict) and isinstance(r.get(key), str)
    ]
    ids = _entity_ids(conn, [r[key] for r in records])
    rows: dict[int, dict[str, Any]] = {}
    missing = []
    for r in records:
        entity_id = ids.get(r[key])
        if entity_id is None:
            missing.append(r[key])
            continue
        rows.setdefault(
            entity_id,
            dict(
                build_id=build_id,
                enricher_id=env.enricher_id,
                scope=ENTITY,
                item_id=None,
                entity_id=entity_id,
                output=r,
            ),
        )
    _missing(written, env.relpath, "an entity", missing)
    if rows:
        conn.execute(insert(enrichment_outputs), list(rows.values()))
        written.outputs[ENTITY] += len(rows)
    return {k: v for k, v in data.items() if k != listed}


def _build(conn: Connection, enricher_id: str, envs: list[_Envelope], state: str) -> int:
    newest = max(envs, key=lambda e: e.computed_at or _EPOCH)
    build_id: int = conn.execute(
        insert(enrichment_builds)
        .values(
            enricher_id=enricher_id,
            enricher_version=newest.version,
            computed_at=newest.computed_at,
            state=state,
            published_at=func.now() if state == "published" else None,
        )
        .returning(enrichment_builds.c.build_id)
    ).scalar_one()
    return build_id


def _episode_item(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for key in ("episode_id", "item_id"):
        if isinstance(data.get(key), str) and data[key]:
            return str(data[key])
    return None


def _envelopes(
    conn: Connection, root: Path, metadata_relpaths: Iterable[str], written: EnrichmentsWritten
) -> tuple[dict[str, list[tuple[str, _Envelope]]], dict[str, int]]:
    by_enricher: dict[str, list[tuple[str, _Envelope]]] = collections.defaultdict(list)
    for path in _corpus_files(root):
        env = _read(root, path, written)
        if env is not None:
            by_enricher[env.enricher_id].append((CORPUS, env))
    for path in _episode_files(root, metadata_relpaths):
        env = _read(root, path, written)
        if env is not None:
            by_enricher[env.enricher_id].append((EPISODE, env))
    episode_keys = [
        k
        for envs in by_enricher.values()
        for s, e in envs
        if s == EPISODE
        for k in [_episode_item(e.data)]
        if k
    ]
    items = _lookup(conn, content_items.c.natural_id, content_items.c.item_id, episode_keys)
    return by_enricher, items


def _write_enricher(
    conn: Connection,
    enricher_id: str,
    envs: list[tuple[str, _Envelope]],
    items: dict[str, int],
    written: EnrichmentsWritten,
    state: str,
) -> int:
    build_id = _build(conn, enricher_id, [e for _, e in envs], state)
    written.builds += 1
    rows: dict[tuple[str, Optional[int]], dict[str, Any]] = {}
    missing: list[str] = []
    for scope, env in envs:
        if scope == CORPUS:
            data: Any = env.data
            if enricher_id in _PER_ENTITY:
                data = _write_entities(conn, build_id, env, written)
            else:
                data = _write_pairs(conn, build_id, env, written)
            key: tuple[str, Optional[int]] = (CORPUS, None)
            if key in rows:
                written.refused.append((env.relpath, "a second corpus output of this enricher"))
                continue
            rows[key] = dict(scope=CORPUS, item_id=None, output=data)
            continue
        natural_id = _episode_item(env.data)
        item_id = items.get(natural_id) if natural_id else None
        if item_id is None:
            missing.append(natural_id or env.relpath)
            continue
        if (EPISODE, item_id) in rows:
            written.refused.append((env.relpath, "a second output of this enricher for its item"))
            continue
        rows[(EPISODE, item_id)] = dict(scope=EPISODE, item_id=item_id, output=env.data)
    _missing(written, f"{enricher_id} episode outputs", "an item", missing)
    values = [
        dict(build_id=build_id, enricher_id=enricher_id, entity_id=None, **r) for r in rows.values()
    ]
    for start in range(0, len(values), 2000):
        conn.execute(insert(enrichment_outputs), values[start : start + 2000])
    for r in rows.values():
        written.outputs[r["scope"]] += 1
    return build_id


def import_enrichments(
    conn: Connection, corpus_root: Path, metadata_relpaths: Iterable[str]
) -> EnrichmentsWritten:
    """Write the enricher outputs under ``corpus_root``; ``metadata_relpaths`` are the metadata
    files the item import selected. An enricher with a published build is left as it is (a second
    import)."""
    written = EnrichmentsWritten()
    by_enricher, items = _envelopes(conn, Path(corpus_root), metadata_relpaths, written)
    published = set(
        conn.execute(
            select(enrichment_builds.c.enricher_id).where(enrichment_builds.c.state == "published")
        ).scalars()
    )
    for enricher_id in sorted(set(by_enricher) - published):
        _write_enricher(conn, enricher_id, by_enricher[enricher_id], items, written, "published")
    return written


def publish_enrichments(
    conn: Connection, corpus_root: Path, metadata_relpaths: Iterable[str], enricher_id: str
) -> EnrichmentsWritten:
    """After one enricher's run on files: write its outputs as a new build, publish it in place of
    its published one (now ``retired``), and delete its retired builds but the newest, so undo can
    publish the previous build. Nothing is published when the enricher left no readable ``ok``
    output. Call it in the run's transaction."""
    written = EnrichmentsWritten()
    by_enricher, items = _envelopes(conn, Path(corpus_root), metadata_relpaths, written)
    if enricher_id not in by_enricher:
        return written
    build_id = _write_enricher(conn, enricher_id, by_enricher[enricher_id], items, written, "built")
    builds = enrichment_builds.c
    mine = builds.enricher_id == enricher_id
    conn.execute(
        update(enrichment_builds).where(mine, builds.state == "published").values(state="retired")
    )
    conn.execute(
        update(enrichment_builds)
        .where(builds.build_id == build_id)
        .values(state="published", published_at=func.now())
    )
    newest_retired = (
        select(func.max(builds.build_id)).where(mine, builds.state == "retired").scalar_subquery()
    )
    conn.execute(
        delete(enrichment_builds).where(
            mine, builds.state == "retired", builds.build_id < newest_retired
        )
    )
    return written
