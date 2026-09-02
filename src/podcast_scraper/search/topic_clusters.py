"""Corpus-wide KG topic clustering from indexed ``kg_topic`` embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

from podcast_scraper.graph_id_utils import slugify_label, topic_node_id_from_slug
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend
from podcast_scraper.search.corpus_scope import (
    discover_metadata_files,
    episode_root_from_metadata_path,
)
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


def _load_topic_clusters_payload(corpus_root: Path) -> Optional[Dict[str, Any]]:
    """Path-safe load of ``search/topic_clusters.json`` → payload dict (None if missing/invalid)."""
    root_p = safe_resolve_directory(corpus_root)
    if root_p is None:
        return None
    root_s = os.path.normpath(str(root_p))
    safe_prefix = root_s + os.sep
    joined = os.path.normpath(os.path.join(root_s, "search", TOPIC_CLUSTERS_FILENAME))
    if joined != root_s and not joined.startswith(safe_prefix):
        return None
    # codeql[py/path-injection] -- joined under root_s (Type 1; CODEQL_DISMISSALS.md).
    if not os.path.isfile(joined):
        return None
    try:
        # codeql[py/path-injection] -- joined sanitized above.
        with open(joined, encoding="utf-8") as fh:
            payload = cast(Dict[str, Any], json.loads(fh.read()))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("topic clusters: skip %s: %s", joined, exc)
        return None
    return payload if isinstance(payload, dict) else None


def load_topic_cluster_enrichment_map(corpus_root: Path) -> Dict[str, Dict[str, str]]:
    """Load ``search/topic_clusters.json`` and return enrichment map; empty if missing/invalid."""
    payload = _load_topic_clusters_payload(corpus_root)
    if payload is None:
        return {}
    return topic_cluster_enrichment_by_topic_id(payload)


def consumer_topic_cluster_map(corpus_root: Path) -> Dict[str, Dict[str, Any]]:
    """Per-topic cluster info for the consumer ``/entities`` endpoint (RFC-102 / PRD-043 FR1).

    ``topic_id`` → ``{cluster_id, cluster_label, cluster_size}`` where ``cluster_id`` is the
    cluster's ``graph_compound_parent_id``, ``cluster_label`` its canonical label, and
    ``cluster_size`` the cross-corpus member count. Topics not in any multi-member cluster
    (singletons) are simply absent. Empty when ``topic_clusters.json`` is missing/invalid.
    """
    payload = _load_topic_clusters_payload(corpus_root)
    if payload is None:
        return {}
    raw = payload.get("clusters")
    if not isinstance(raw, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for cl in raw:
        if not isinstance(cl, Mapping):
            continue
        gpid = _graph_compound_parent_id(cl)
        if not gpid:
            continue
        label_raw = cl.get("canonical_label")
        label = str(label_raw).strip() if isinstance(label_raw, str) and label_raw.strip() else gpid
        members = cl.get("members")
        if not isinstance(members, list):
            continue
        size = len(members)
        for m in members:
            if not isinstance(m, Mapping):
                continue
            tid = m.get("topic_id")
            if isinstance(tid, str) and tid.strip():
                out[tid.strip()] = {
                    "cluster_id": gpid,
                    "cluster_label": label,
                    "cluster_size": size,
                }
    return out


def top_clusters_by_member_count(corpus_root: Path, top_n: int = 12) -> List[Dict[str, Any]]:
    """Top-N clusters by member count (desc) for the interests picker (PRD-043 FR4 / 3.5).

    Returns ``[{"id", "label", "size"}, ...]``; empty when the artifact is missing/invalid.
    ``size`` is the explicit ``member_count`` when present, else ``len(members)``. ``id`` is the
    cluster's ``graph_compound_parent_id`` (the stable interest key stored per-user).
    """
    payload = _load_topic_clusters_payload(corpus_root)
    if payload is None:
        return []
    raw = payload.get("clusters")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for cl in raw:
        if not isinstance(cl, Mapping):
            continue
        gpid = _graph_compound_parent_id(cl)
        if not gpid:
            continue
        mc = cl.get("member_count")
        members = cl.get("members")
        if isinstance(mc, int):
            size = mc
        elif isinstance(members, list):
            size = len(members)
        else:
            size = 0
        label_raw = cl.get("canonical_label")
        label = str(label_raw).strip() if isinstance(label_raw, str) and label_raw.strip() else gpid
        out.append({"id": gpid, "label": label, "size": size})
    out.sort(key=lambda c: c["size"], reverse=True)
    return out[: max(top_n, 0)]


def consumer_cluster_siblings(corpus_root: Path, topic_id: str) -> List[Dict[str, str]]:
    """Sibling topics sharing ``topic_id``'s cluster, excluding itself (PRD-043 FR3).

    Returns ``[{"id", "label"}, ...]`` drawn from the cluster's ``members`` (so each carries
    its own display label). Empty when the topic is a singleton, absent, or the artifact is
    missing/invalid. The first cluster containing ``topic_id`` wins (topic ids are unique
    across clusters by construction).
    """
    tid = topic_id.strip()
    if not tid:
        return []
    payload = _load_topic_clusters_payload(corpus_root)
    if payload is None:
        return []
    raw = payload.get("clusters")
    if not isinstance(raw, list):
        return []
    for cl in raw:
        if not isinstance(cl, Mapping):
            continue
        members = cl.get("members")
        if not isinstance(members, list):
            continue
        member_ids = {
            str(m.get("topic_id")).strip()
            for m in members
            if isinstance(m, Mapping) and isinstance(m.get("topic_id"), str)
        }
        if tid not in member_ids:
            continue
        siblings: List[Dict[str, str]] = []
        for m in members:
            if not isinstance(m, Mapping):
                continue
            mid_raw = m.get("topic_id")
            if not isinstance(mid_raw, str) or not mid_raw.strip():
                continue
            mid = mid_raw.strip()
            if mid == tid:
                continue
            label_raw = m.get("label")
            label = (
                label_raw.strip()
                if isinstance(label_raw, str) and label_raw.strip()
                else mid.split(":", 1)[-1]
            )
            siblings.append({"id": mid, "label": label})
        return siblings
    return []


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for L2-normalized rows (``n``, ``d``)."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2-D")
    return cast(np.ndarray, vectors @ vectors.T)


def cluster_indices_by_threshold(sim: np.ndarray, threshold: float) -> np.ndarray:
    """UPGMA (average-linkage) clustering using cosine similarity.

    Equivalent to the old greedy merge loop but uses scipy's O(n²) UPGMA
    implementation so it scales to corpus-size topic vectors without hanging.
    Math: mean-cosine-distance = 1 − mean-cosine-similarity; average linkage is
    monotonic, so cutting the dendrogram at (1 − threshold) yields exactly the
    partition where two clusters stop merging when their mean similarity falls
    below threshold.

    Args:
        sim: Symmetric similarity matrix ``(n, n)`` with ones on diagonal.
        threshold: Minimum mean cosine similarity between two clusters to merge.

    Returns:
        Integer cluster label per row (0 .. k-1).
    """
    n = int(sim.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    # Lazy import: scipy lives in the ``[search]`` extra, but this module is imported
    # transitively by search.capability / the MCP tools under the core ``[dev]`` env (CI
    # test-unit). A module-level scipy import would break every unit test that touches those
    # paths; importing here keeps the module light and only requires scipy when we actually
    # cluster (which only happens with the search stack installed).
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    # Convert similarity → distance; clip to [0, 2] to guard floating-point overshoot.
    dist_mat = np.clip(1.0 - sim, 0.0, 2.0)
    condensed = squareform(dist_mat, checks=False)
    # Finite-guard. ``checks=False`` skips scipy's own finiteness check, and a non-finite
    # distance makes ``linkage`` raise "must contain only finite values". The zero-vector path
    # is already guarded upstream (collect_topic_rows_from_lance skips the 1/‖v‖ divide when the
    # norm is ~0), so this only trips if an input embedding is itself NaN/inf — rare model poison.
    # Fall back to all-singletons rather than crash the (non-fatal) corpus finalize; log so the
    # bad input is visible instead of silently swallowed.
    if not np.all(np.isfinite(condensed)):  # pragma: no cover - defensive NaN/inf-poison guard
        logger.warning(
            "topic clustering: %d/%d non-finite distances (NaN/inf input embedding?) — "
            "falling back to all-singleton clusters",
            int(np.count_nonzero(~np.isfinite(condensed))),
            condensed.size,
        )
        return np.arange(n, dtype=np.int64)
    Z = linkage(condensed, method="average")
    raw = fcluster(Z, t=1.0 - threshold, criterion="distance")
    # scipy labels are 1-based; shift to 0-based.
    return np.asarray(raw, dtype=np.int64) - 1


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


def collect_topic_rows_from_lance(
    lance_dir: Path,
    label_by_topic_id: Mapping[str, str],
) -> List[TopicVectorRow]:
    """Aggregate ``kg_topic`` rows from the LanceDB ``aux`` table by ``source_id`` (#995)."""
    be = LanceDBBackend(str(lance_dir))
    tbl = be._open_if_exists("aux")
    if tbl is None:
        return []
    n = tbl.count_rows()
    raw = (
        tbl.search()
        .where("doc_type = 'kg_topic'")
        .limit(max(n, 1))
        .select(["source_id", "episode_id", "embedding"])
        .to_list()
    )
    # source_id -> list of (episode_id, vec)
    bucket: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for r in raw:
        sid = r.get("source_id")
        if not isinstance(sid, str) or not sid.strip():
            continue
        ep = r.get("episode_id")
        eid = ep.strip() if isinstance(ep, str) and ep.strip() else ""
        emb = r.get("embedding")
        if emb is None:
            continue
        bucket.setdefault(sid, []).append((eid, np.asarray(emb, dtype=np.float32)))

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


def fingerprint_topic_rows(rows: Sequence[TopicVectorRow]) -> str:
    """SHA-256 fingerprint of the sorted (topic_id, vector-bytes) pairs.

    Used by the skip-gate in :func:`build_topic_clusters_payload` to detect when
    the input topic rows are unchanged since the last clustering run.
    """
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r.topic_id):
        h.update(r.topic_id.encode())
        h.update(r.vector.tobytes())
    return h.hexdigest()


def build_topic_clusters_payload(
    rows: Sequence[TopicVectorRow],
    *,
    threshold: float,
    embedding_model: str,
    prior_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """Build ``topic_clusters.json`` body from aggregated topic rows.

    When *prior_fingerprint* matches the current rows' fingerprint, clustering is
    skipped and the returned dict carries ``skipped_unchanged: True`` so callers
    can mirror IndexRunStats' incrementality shape.
    """
    current_fp = fingerprint_topic_rows(rows)

    if prior_fingerprint is not None and prior_fingerprint == current_fp:
        logger.info("topic-clusters: skipped_unchanged (fingerprint=%s)", current_fp[:16])
        return {
            "schema_version": TOPIC_CLUSTERS_SCHEMA_VERSION,
            "model": embedding_model,
            "threshold": threshold,
            "skipped_unchanged": True,
            "fingerprint": current_fp,
        }

    if not rows:
        return {
            "schema_version": TOPIC_CLUSTERS_SCHEMA_VERSION,
            "model": embedding_model,
            "threshold": threshold,
            "clusters": [],
            "singletons": 0,
            "topic_count": 0,
            "cluster_count": 0,
            "fingerprint": current_fp,
        }

    ids = [r.topic_id for r in rows]
    mat = np.stack([r.vector for r in rows], axis=0)
    sim = cosine_similarity_matrix(mat)
    labels = cluster_indices_by_threshold(sim, threshold)

    by_label: MutableMapping[int, List[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        by_label.setdefault(int(lab), []).append(i)

    # Deterministic cluster ordering: sort groups by the sorted tuple of their
    # member topic_ids so slug assignment is stable across algorithm changes.
    sorted_groups: List[List[int]] = sorted(
        by_label.values(),
        key=lambda idxs: tuple(sorted(rows[i].topic_id for i in idxs)),
    )

    used_tc_slugs: Set[str] = set()
    clusters_out: List[Dict[str, Any]] = []
    singletons = 0

    for member_indices in sorted_groups:
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
        "fingerprint": current_fp,
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


#: Cosine-similarity floor for linking two ``kg_topic`` vectors into one cluster.
#:
#: 0.70, measured on the REAL corpus (see af6bed32 / RFC-075's production sweep) — NOT the 0.75
#: that was Pareto-optimal on the six-cluster v2 fixtures. Mirrors
#: ``PodcastScraperConfig.topic_cluster_threshold``; keep the two in step.
#:
#: This is the single source of truth for every caller INCLUDING the ``topic-clusters`` CLI. The
#: CLI previously carried its own literal ``0.75`` argparse default, which meant it passed 0.75
#: explicitly on every invocation and this default was never reached — so the config change did
#: not affect the one command that actually rebuilds ``search/topic_clusters.json``.
DEFAULT_TOPIC_CLUSTER_THRESHOLD = 0.70


def build_topic_clusters_for_corpus(
    output_dir: str | Path,
    *,
    index_dir: Optional[Path] = None,
    threshold: float = DEFAULT_TOPIC_CLUSTER_THRESHOLD,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the LanceDB index, aggregate ``kg_topic`` vectors, cluster, return JSON payload."""
    root = Path(output_dir).resolve()
    idx = Path(index_dir).resolve() if index_dir is not None else (root / "search").resolve()
    lance_dir = idx / "lance_index"
    if not (lance_dir.is_dir() and any(lance_dir.iterdir())):
        raise FileNotFoundError(f"No LanceDB index at {lance_dir}")

    labels_map = load_kg_topic_labels_from_corpus(root)
    rows = collect_topic_rows_from_lance(lance_dir, labels_map)
    model = str(
        (LanceDBBackend(str(lance_dir)).read_index_meta() or {}).get("embedding_model") or ""
    )

    # Load prior fingerprint from the existing output file (if present) for the skip-gate.
    target = out_path if out_path is not None else idx / TOPIC_CLUSTERS_FILENAME
    prior_fp: Optional[str] = None
    if target.is_file():
        try:
            existing_raw = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(existing_raw, dict):
                fp_val = existing_raw.get("fingerprint")
                if isinstance(fp_val, str) and fp_val:
                    prior_fp = fp_val
        except (OSError, json.JSONDecodeError):
            pass

    payload = build_topic_clusters_payload(
        rows, threshold=threshold, embedding_model=model, prior_fingerprint=prior_fp
    )

    if payload.get("skipped_unchanged"):
        logger.info("topic-clusters: skipped_unchanged for %s", target)
        return payload

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
