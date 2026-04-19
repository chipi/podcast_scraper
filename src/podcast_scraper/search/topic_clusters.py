"""Corpus-wide KG topic clustering from FAISS ``kg_topic`` embeddings."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

from podcast_scraper.graph_id_utils import slugify_label, topic_node_id_from_slug
from podcast_scraper.search.corpus_scope import (
    discover_metadata_files,
    episode_root_from_metadata_path,
)
from podcast_scraper.search.faiss_store import FaissVectorStore, VECTORS_FILE
from podcast_scraper.search.indexer import _kg_path, _load_metadata_file
from podcast_scraper.utils.path_validation import safe_resolve_directory

logger = logging.getLogger(__name__)

TOPIC_CLUSTERS_FILENAME = "topic_clusters.json"
# v2 renames fields so "graph compound" (viewer) vs "CIL alias target" (identity) are not confused.
TOPIC_CLUSTERS_SCHEMA_VERSION = "2"


def _cil_alias_target_topic_id(cluster: Mapping[str, Any]) -> Optional[str]:
    """``cil_alias_target_topic_id`` (v2) or legacy ``canonical_topic_id`` (v1)."""
    for key in ("cil_alias_target_topic_id", "canonical_topic_id"):
        v = cluster.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _graph_compound_parent_id(cluster: Mapping[str, Any]) -> Optional[str]:
    """``graph_compound_parent_id`` (v2) or legacy ``cluster_id`` (v1)."""
    for key in ("graph_compound_parent_id", "cluster_id"):
        v = cluster.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def topic_cluster_enrichment_by_topic_id(
    payload: Mapping[str, Any],
) -> Dict[str, Dict[str, str]]:
    """Build ``topic_id`` → cluster fields for search metadata (query-time join from JSON).

    Later clusters in the payload overwrite earlier entries for the same ``topic_id`` (matches
    viewer overlay when a topic appears in multiple clusters).
    """
    out: Dict[str, Dict[str, str]] = {}
    raw = payload.get("clusters")
    if not isinstance(raw, list):
        return out
    for cl in raw:
        if not isinstance(cl, Mapping):
            continue
        gpid = _graph_compound_parent_id(cl)
        if not gpid:
            continue
        cil = _cil_alias_target_topic_id(cl)
        label_raw = cl.get("canonical_label")
        canon_label = (
            str(label_raw).strip()
            if isinstance(label_raw, str) and str(label_raw).strip()
            else gpid
        )
        members = cl.get("members")
        if not isinstance(members, list):
            continue
        for m in members:
            if not isinstance(m, Mapping):
                continue
            tid = m.get("topic_id")
            if not isinstance(tid, str) or not tid.strip():
                continue
            entry: Dict[str, str] = {
                "graph_compound_parent_id": gpid,
                "canonical_label": canon_label,
            }
            if isinstance(cil, str) and cil.strip():
                entry["cil_alias_target_topic_id"] = cil.strip()
            out[tid.strip()] = entry
    return out


def load_topic_cluster_enrichment_map(corpus_root: Path) -> Dict[str, Dict[str, str]]:
    """Load ``search/topic_clusters.json`` and return enrichment map; empty if missing/invalid."""
    root_p = safe_resolve_directory(corpus_root)
    if root_p is None:
        return {}
    root_s = os.path.normpath(str(root_p))
    safe_prefix = root_s + os.sep
    joined = os.path.normpath(os.path.join(root_s, "search", TOPIC_CLUSTERS_FILENAME))
    if joined != root_s and not joined.startswith(safe_prefix):
        return {}
    # codeql[py/path-injection] -- joined under root_s (Type 1; CODEQL_DISMISSALS.md).
    if not os.path.isfile(joined):
        return {}
    try:
        # codeql[py/path-injection] -- joined sanitized above.
        with open(joined, encoding="utf-8") as fh:
            payload = cast(Dict[str, Any], json.loads(fh.read()))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("topic cluster enrichment: skip %s: %s", joined, exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    return topic_cluster_enrichment_by_topic_id(payload)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for L2-normalized rows (``n``, ``d``)."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2-D")
    return cast(np.ndarray, vectors @ vectors.T)


def cluster_indices_by_threshold(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy average-linkage merging using cosine similarity.

    Repeatedly merges the two clusters whose **mean** pairwise similarity between
    members is highest, while that mean is still ``>= threshold``. This avoids
    single-linkage chaining (where A–B and B–C links force A with C even when
    direct A–C similarity is low).

    Args:
        sim: Symmetric similarity matrix ``(n, n)`` with ones on diagonal.
        threshold: Minimum mean cosine similarity between two clusters to merge.

    Returns:
        Integer cluster label per row (0 .. k-1).
    """
    n = int(sim.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    clusters: List[Set[int]] = [{i} for i in range(n)]

    def mean_inter_cluster(ci: Set[int], cj: Set[int]) -> float:
        tot = 0.0
        cnt = 0
        for a in ci:
            for b in cj:
                tot += float(sim[a, b])
                cnt += 1
        return tot / max(cnt, 1)

    while len(clusters) > 1:
        best_i, best_j = 0, 1
        best_s = -2.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = mean_inter_cluster(clusters[i], clusters[j])
                if s > best_s:
                    best_s = s
                    best_i, best_j = i, j
        if best_s < threshold:
            break
        merged = clusters[best_i] | clusters[best_j]
        clusters.pop(best_j)
        clusters[best_i] = merged

    labels = np.zeros(n, dtype=np.int64)
    for li, c in enumerate(clusters):
        for idx in c:
            labels[idx] = li
    return labels


def pick_centroid_closest_label(
    member_indices: Sequence[int],
    vectors: np.ndarray,
) -> int:
    """Index of member whose embedding has highest mean cosine similarity to others."""
    idx = list(member_indices)
    if not idx:
        return 0
    if len(idx) == 1:
        return idx[0]
    sub = vectors[np.array(idx, dtype=np.int64)]
    centroid = np.mean(sub, axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 1e-12:
        centroid = centroid / norm
    best_i = idx[0]
    best_score = -1.0
    for i in idx:
        score = float(np.dot(vectors[i], centroid))
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def load_kg_topic_labels_from_corpus(output_root: Path) -> Dict[str, str]:
    """Map ``topic:…`` node id to display label from all ``*.kg.json`` under *output_root*."""
    out: Dict[str, str] = {}
    for meta_path in discover_metadata_files(output_root):
        doc = _load_metadata_file(meta_path)
        if not doc:
            continue
        episode_root = episode_root_from_metadata_path(meta_path)
        kg_path = _kg_path(episode_root, meta_path, doc)
        if not kg_path.is_file():
            continue
        try:
            kg = json.loads(kg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skip kg.json %s: %s", kg_path, exc)
            continue
        for n in kg.get("nodes") or []:
            if not isinstance(n, dict) or n.get("type") != "Topic":
                continue
            nid = n.get("id")
            if not isinstance(nid, str) or not nid.strip():
                continue
            props = n.get("properties")
            p = props if isinstance(props, dict) else {}
            label = p.get("label")
            if isinstance(label, str) and label.strip():
                out[nid] = label.strip()
            elif nid not in out:
                out[nid] = nid
    return out


@dataclass
class TopicVectorRow:
    """One clustered topic (unique ``source_id`` / ``topic:`` id)."""

    topic_id: str
    label: str
    episode_ids: List[str]
    vector: np.ndarray


def collect_topic_rows_from_faiss(
    store: FaissVectorStore,
    label_by_topic_id: Mapping[str, str],
) -> List[TopicVectorRow]:
    """Aggregate ``kg_topic`` FAISS rows by ``metadata['source_id']`` (topic node id)."""
    vectors_by_doc = store.export_vectors_by_doc_id()
    # source_id -> list of (episode_id, vec)
    bucket: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for doc_id, meta in store.metadata_by_doc_id.items():
        if meta.get("doc_type") != "kg_topic":
            continue
        sid = meta.get("source_id")
        if not isinstance(sid, str) or not sid.strip():
            continue
        ep = meta.get("episode_id")
        eid = ep.strip() if isinstance(ep, str) and ep.strip() else ""
        vec = vectors_by_doc.get(doc_id)
        if vec is None:
            logger.warning("Missing vector for doc_id %s (metadata present)", doc_id)
            continue
        bucket.setdefault(sid, []).append((eid, vec))

    rows: List[TopicVectorRow] = []
    for topic_id, pairs in sorted(bucket.items()):
        episode_ids = sorted({e for e, _ in pairs if e})
        mats = np.stack([v for _, v in pairs], axis=0)
        mean_v = np.mean(mats, axis=0)
        nrm = float(np.linalg.norm(mean_v))
        if nrm > 1e-12:
            mean_v = mean_v / nrm
        label = label_by_topic_id.get(topic_id, topic_id)
        rows.append(
            TopicVectorRow(
                topic_id=topic_id,
                label=label,
                episode_ids=episode_ids,
                vector=np.asarray(mean_v, dtype=np.float32),
            )
        )
    return rows


def build_topic_clusters_payload(
    rows: Sequence[TopicVectorRow],
    *,
    threshold: float,
    embedding_model: str,
) -> Dict[str, Any]:
    """Build ``topic_clusters.json`` body from aggregated topic rows."""
    if not rows:
        return {
            "schema_version": TOPIC_CLUSTERS_SCHEMA_VERSION,
            "model": embedding_model,
            "threshold": threshold,
            "clusters": [],
            "singletons": 0,
            "topic_count": 0,
            "cluster_count": 0,
        }

    ids = [r.topic_id for r in rows]
    mat = np.stack([r.vector for r in rows], axis=0)
    sim = cosine_similarity_matrix(mat)
    labels = cluster_indices_by_threshold(sim, threshold)

    by_label: MutableMapping[int, List[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        by_label.setdefault(int(lab), []).append(i)

    used_tc_slugs: Set[str] = set()
    clusters_out: List[Dict[str, Any]] = []
    singletons = 0

    for _lab, member_indices in sorted(by_label.items(), key=lambda x: x[0]):
        if len(member_indices) < 2:
            singletons += len(member_indices)
            continue
        best_idx = pick_centroid_closest_label(member_indices, mat)
        canonical_label = rows[best_idx].label
        base_slug = slugify_label(canonical_label)
        tc_slug = base_slug
        suffix = 0
        while tc_slug in used_tc_slugs:
            suffix += 1
            tc_slug = f"{base_slug}-{suffix}"
        used_tc_slugs.add(tc_slug)
        graph_compound_parent_id = f"tc:{tc_slug}"

        centroid = np.mean(mat[np.array(member_indices, dtype=np.int64)], axis=0)
        cn = float(np.linalg.norm(centroid))
        if cn > 1e-12:
            centroid = centroid / cn

        members_json: List[Dict[str, Any]] = []
        for mi in sorted(member_indices, key=lambda i: rows[i].topic_id):
            score = float(np.dot(mat[mi], centroid))
            members_json.append(
                {
                    "topic_id": rows[mi].topic_id,
                    "label": rows[mi].label,
                    "similarity_to_centroid": round(score, 6),
                    "episode_ids": list(rows[mi].episode_ids),
                }
            )

        cil_alias_target_topic_id = topic_node_id_from_slug(slugify_label(canonical_label))

        clusters_out.append(
            {
                "canonical_label": canonical_label,
                "cil_alias_target_topic_id": cil_alias_target_topic_id,
                "graph_compound_parent_id": graph_compound_parent_id,
                "member_count": len(member_indices),
                "members": members_json,
            }
        )

    return {
        "schema_version": TOPIC_CLUSTERS_SCHEMA_VERSION,
        "model": embedding_model,
        "threshold": threshold,
        "clusters": clusters_out,
        "singletons": singletons,
        "topic_count": len(ids),
        "cluster_count": len(clusters_out),
    }


def topic_id_aliases_from_clusters_payload(
    payload: Mapping[str, Any],
) -> Dict[str, str]:
    """Map variant ``topic:…`` ids to each cluster's CIL merge target.

    Built from the in-memory ``topic_clusters.json`` body. Uses
    ``cil_alias_target_topic_id`` (v2) or legacy ``canonical_topic_id`` (v1). Every
    member whose ``topic_id`` differs from that target becomes ``alias -> target``.
    Singleton clusters are not present in ``clusters`` and produce no aliases.

    Args:
        payload: Object returned by :func:`build_topic_clusters_payload` /
            :func:`build_topic_clusters_for_corpus`.

    Returns:
        New dict suitable for merging into ``cil_lift_overrides.json`` ``topic_id_aliases``.
    """
    out: Dict[str, str] = {}
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        return out
    for cl in clusters:
        if not isinstance(cl, dict):
            continue
        target = _cil_alias_target_topic_id(cl)
        if not target:
            continue
        members = cl.get("members")
        if not isinstance(members, list):
            continue
        for m in members:
            if not isinstance(m, dict):
                continue
            tid = m.get("topic_id")
            if not isinstance(tid, str) or not tid.strip():
                continue
            tid_s = tid.strip()
            if tid_s == target:
                continue
            out[tid_s] = target
    return out


def write_topic_clusters_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write *payload* as formatted JSON to *path*, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_topic_clusters_for_corpus(
    output_dir: str | Path,
    *,
    index_dir: Optional[Path] = None,
    threshold: float = 0.75,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load FAISS index, aggregate ``kg_topic`` vectors, cluster, return JSON payload."""
    root = Path(output_dir).resolve()
    idx = Path(index_dir).resolve() if index_dir is not None else (root / "search").resolve()
    if not (idx / VECTORS_FILE).is_file():
        raise FileNotFoundError(f"No FAISS index at {idx} (expected {VECTORS_FILE})")

    store = FaissVectorStore.load(idx)
    labels_map = load_kg_topic_labels_from_corpus(root)
    rows = collect_topic_rows_from_faiss(store, labels_map)
    model = store.embedding_model
    payload = build_topic_clusters_payload(rows, threshold=threshold, embedding_model=model)
    target = out_path if out_path is not None else idx / TOPIC_CLUSTERS_FILENAME
    write_topic_clusters_json(target, payload)
    logger.info(
        "Wrote %s (schema_version=%s topics=%s clusters=%s singleton_slots=%s)",
        target,
        payload.get("schema_version"),
        payload["topic_count"],
        payload["cluster_count"],
        payload["singletons"],
    )
    return payload


def load_validation_yaml(path: Path) -> Dict[str, Any]:
    """Load a topic-cluster validation YAML file and return its root mapping."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("validation yaml root must be a mapping")
    return raw


def evaluate_validation_against_topics(
    spec: Mapping[str, Any],
    topic_ids: Sequence[str],
    cluster_labels: Sequence[int],
) -> Tuple[bool, List[str]]:
    """Check expected merge pairs / distinct constraints. Returns (ok, error messages)."""
    id_to_c = {tid: int(cluster_labels[i]) for i, tid in enumerate(topic_ids)}
    errors: List[str] = []

    def _check_merge_pair(group: Mapping[str, Any], tids: List[Any]) -> None:
        if len(tids) != 2:
            errors.append(
                f"expected_merge_pairs[{group.get('id', '?')}]: need exactly two topic_ids"
            )
            return
        a, b = tids[0], tids[1]
        if not isinstance(a, str) or not isinstance(b, str):
            return
        if a not in id_to_c or b not in id_to_c:
            errors.append(
                f"expected_merge_pairs[{group.get('id', '?')}]: missing topic in corpus: "
                f"{[t for t in (a, b) if t not in id_to_c]}"
            )
            return
        if id_to_c[a] != id_to_c[b]:
            errors.append(
                f"expected_merge_pairs[{group.get('id', '?')}]: "
                f"{a!r} and {b!r} should share a cluster (got {id_to_c[a]} vs {id_to_c[b]})"
            )

    for group in spec.get("expected_merge_pairs") or []:
        if not isinstance(group, dict):
            continue
        tids = group.get("topic_ids") or []
        if isinstance(tids, list):
            _check_merge_pair(group, tids)

    for group in spec.get("expected_clusters") or []:
        if not isinstance(group, dict):
            continue
        tids = group.get("topic_ids") or []
        if not isinstance(tids, list) or len(tids) < 2:
            continue
        present = [t for t in tids if isinstance(t, str) and t in id_to_c]
        missing = [t for t in tids if isinstance(t, str) and t not in id_to_c]
        if missing:
            errors.append(
                f"expected_clusters[{group.get('id', '?')}]: missing topic ids in corpus: {missing}"
            )
            continue
        clusters = {id_to_c[t] for t in present}
        if len(clusters) != 1:
            errors.append(
                f"expected_clusters[{group.get('id', '?')}]: topics split across "
                f"clusters {sorted(clusters)} (want single cluster): {present}"
            )
    for pair in spec.get("expected_distinct") or []:
        if not isinstance(pair, dict):
            continue
        tids = pair.get("topic_ids") or []
        if not isinstance(tids, list) or len(tids) != 2:
            continue
        a, b = tids[0], tids[1]
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if a not in id_to_c or b not in id_to_c:
            continue
        if id_to_c[a] == id_to_c[b]:
            errors.append(
                f"expected_distinct[{pair.get('id', '?')}]: "
                f"{a!r} and {b!r} landed in same cluster (should differ)"
            )
    return (len(errors) == 0, errors)
