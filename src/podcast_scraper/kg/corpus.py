"""File-based KG corpus: scan artifacts, roll-ups, co-occurrence, export bundles."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from podcast_scraper.utils.log_redaction import format_exception_for_log

from .io import read_artifact
from .schema import validate_artifact

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_INVALID_ARGS = 2
EXIT_NO_ARTIFACTS = 3
EXIT_VALIDATION_FAILED = 1


def scan_kg_artifact_paths(output_dir: Path) -> List[Path]:
    """List all .kg.json paths under output_dir (metadata/*.kg.json and rglob)."""
    out = Path(output_dir)
    if not out.is_dir():
        return []
    paths: List[Path] = []
    metadata_dir = out / "metadata"
    if metadata_dir.is_dir():
        paths.extend(metadata_dir.glob("*.kg.json"))
    for p in out.rglob("*.kg.json"):
        if p not in paths:
            paths.append(p)
    return sorted(set(paths))


def collect_kg_paths_from_inputs(paths: List[Path]) -> List[Path]:
    """Expand files and directories to a list of .kg.json paths."""
    result: List[Path] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")
        if p.is_file():
            if not p.name.endswith(".kg.json"):
                raise ValueError(f"Not a .kg.json file: {p}")
            result.append(p)
            continue
        for child in p.rglob("*.kg.json"):
            result.append(child)
    return sorted(set(result))


def load_kg_artifacts(
    paths: List[Path],
    *,
    validate: bool = False,
    strict: bool = False,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Load artifacts from paths; skip invalid with warning when not strict."""
    out: List[Tuple[Path, Dict[str, Any]]] = []
    for path in paths:
        try:
            data = read_artifact(path)
            if validate:
                validate_artifact(data, strict=strict)
            out.append((path, data))
        except Exception as e:
            if strict:
                raise
            logger.warning(
                "Skip invalid KG artifact %s: %s",
                path,
                format_exception_for_log(e),
            )
    return out


def inspect_summary(
    artifact: Dict[str, Any], *, artifact_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Build a JSON-serializable summary for kg inspect."""
    nodes = artifact.get("nodes") or []
    edges = artifact.get("edges") or []
    by_type: Dict[str, int] = defaultdict(int)
    for n in nodes:
        t = str(n.get("type", "?"))
        by_type[t] += 1
    topics: List[Dict[str, str]] = []
    entities: List[Dict[str, Any]] = []
    ep_title = None
    for n in nodes:
        nt = n.get("type")
        props = n.get("properties") or {}
        if nt == "Topic":
            topics.append(
                {
                    "id": str(n.get("id", "")),
                    "label": str(props.get("label", "")),
                    "slug": str(props.get("slug", "")),
                }
            )
        elif nt == "Entity":
            entities.append(
                {
                    "id": str(n.get("id", "")),
                    "name": str(props.get("name", "")),
                    "entity_kind": str(props.get("entity_kind", "")),
                    "role": props.get("role"),
                }
            )
        elif nt == "Episode":
            ep_title = props.get("title")
    ext = artifact.get("extraction") or {}
    summary = {
        "episode_id": artifact.get("episode_id"),
        "schema_version": artifact.get("schema_version"),
        "extraction": ext,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_type": dict(by_type),
        "topics": topics,
        "entities": entities,
        "episode_title": ep_title,
    }
    if artifact_path is not None:
        summary["artifact_path"] = str(artifact_path)
    return summary


def entity_rollup(
    loaded: List[Tuple[Path, Dict[str, Any]]],
    *,
    min_episodes: int = 1,
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Aggregate Entity nodes across episodes (RFC-056 entity roll-up pattern)."""
    # key -> {display, entity_kind, episode_ids set, paths, mention_count}
    agg: Dict[str, Dict[str, Any]] = {}
    for path, art in loaded:
        epi = art.get("episode_id") or ""
        ep_node = next((n for n in art.get("nodes", []) if n.get("type") == "Episode"), None)
        ep_title = None
        if ep_node:
            ep_title = (ep_node.get("properties") or {}).get("title")
        rel_path = str(path)
        if output_dir:
            try:
                rel_path = str(path.relative_to(output_dir))
            except ValueError:
                pass
        edges = art.get("edges") or []
        for n in art.get("nodes", []):
            if n.get("type") != "Entity":
                continue
            props = n.get("properties") or {}
            name = str(props.get("name", "")).strip()
            if not name:
                continue
            kind = str(props.get("entity_kind", "person"))
            key = f"{kind}:{name.lower()}"
            if key not in agg:
                agg[key] = {
                    "key": key,
                    "name": name,
                    "entity_kind": kind,
                    "episode_ids": set(),
                    "episodes": [],
                    "mention_count": 0,
                }
            bucket = agg[key]
            bucket["mention_count"] += sum(
                1 for e in edges if e.get("from") == n.get("id") and e.get("type") == "MENTIONS"
            )
            if epi and epi not in bucket["episode_ids"]:
                bucket["episode_ids"].add(epi)
                bucket["episodes"].append(
                    {
                        "episode_id": epi,
                        "title": ep_title or "",
                        "artifact_path": rel_path,
                    }
                )
    rows: List[Dict[str, Any]] = []
    for _k, b in sorted(agg.items(), key=lambda x: (-len(x[1]["episode_ids"]), x[1]["name"])):
        if len(b["episode_ids"]) < min_episodes:
            continue
        rows.append(
            {
                "entity_kind": b["entity_kind"],
                "name": b["name"],
                "episode_count": len(b["episode_ids"]),
                "mention_count": b["mention_count"],
                "episodes": b["episodes"],
            }
        )
    return rows


def topic_cooccurrence(
    loaded: List[Tuple[Path, Dict[str, Any]]],
    *,
    min_support: int = 1,
) -> List[Dict[str, Any]]:
    """Count Topic–Topic pairs that appear in the same episode (unordered pairs)."""
    pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_meta: Dict[Tuple[str, str], Tuple[str, str]] = {}

    for _path, art in loaded:
        topics = [n for n in art.get("nodes", []) if n.get("type") == "Topic"]
        ids = [str(t.get("id", "")) for t in topics if t.get("id")]
        labels: Dict[str, str] = {}
        for t in topics:
            tid = str(t.get("id", ""))
            props = t.get("properties") or {}
            labels[tid] = str(props.get("label", tid))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sorted((ids[i], ids[j]))
                key = (a, b)
                pair_count[key] += 1
                pair_meta[key] = (labels.get(a, a), labels.get(b, b))

    out: List[Dict[str, Any]] = []
    for (a, b), cnt in sorted(pair_count.items(), key=lambda x: -x[1]):
        if cnt < min_support:
            continue
        la, lb = pair_meta[(a, b)]
        out.append(
            {
                "topic_a_id": a,
                "topic_b_id": b,
                "topic_a_label": la,
                "topic_b_label": lb,
                "episode_count": cnt,
            }
        )
    return out


def export_ndjson(
    loaded: List[Tuple[Path, Dict[str, Any]]],
    *,
    output_dir: Optional[Path],
    stream_write: Callable[[str], None],
) -> None:
    """Write one JSON object per line; each includes _artifact_path."""
    for path, art in loaded:
        row = dict(art)
        rel = str(path)
        if output_dir:
            try:
                rel = str(path.relative_to(output_dir))
            except ValueError:
                pass
        row["_artifact_path"] = rel
        stream_write(json.dumps(row, ensure_ascii=False) + "\n")


def export_merged_json(
    loaded: List[Tuple[Path, Dict[str, Any]]],
    *,
    output_dir: Optional[Path],
) -> Dict[str, Any]:
    """Single JSON document with all artifacts (corpus bundle)."""
    artifacts: List[Dict[str, Any]] = []
    for path, art in loaded:
        copy = dict(art)
        rel = str(path)
        if output_dir:
            try:
                rel = str(path.relative_to(output_dir))
            except ValueError:
                pass
        copy["_artifact_path"] = rel
        artifacts.append(copy)
    total_nodes = sum(len(a.get("nodes") or []) for _, a in loaded)
    total_edges = sum(len(a.get("edges") or []) for _, a in loaded)
    return {
        "export_kind": "kg_corpus_bundle",
        "schema_version": "1.0",
        "artifact_count": len(artifacts),
        "node_count_total": total_nodes,
        "edge_count_total": total_edges,
        "artifacts": artifacts,
    }
