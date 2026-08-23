"""C1 regression tests: UPGMA replacement, deterministic slugs, skip-gate, wall-clock guard.

Covers:
  (a) Partition equivalence — scipy UPGMA produces the same partition as the old
      greedy loop on a tie-free synthetic fixture.
  (b) Deterministic slugs — same input yields the same ``tc:`` slugs across two
      calls (and across permuted label orderings).
  (c) Skip-gate — unchanged topic rows are skipped; changed rows trigger clustering.
  (d) Wall-clock guardrail (ADR-095 Tier-2) — corpus-scale fixture (≥2 k topics)
      completes within 30 s.
"""

from __future__ import annotations

import time
from typing import cast, List

import numpy as np
import pytest

from podcast_scraper.search.topic_clusters import (
    build_topic_clusters_payload,
    cluster_indices_by_threshold,
    fingerprint_topic_rows,
    TopicVectorRow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(d: int, idx: int, total: int) -> np.ndarray:
    """Unit vector in direction *idx* out of *total* equally-spaced 2-D directions."""
    angle = 2.0 * np.pi * idx / total
    v = np.zeros(d, dtype=np.float32)
    v[0] = np.cos(angle)
    v[1] = np.sin(angle)
    return v


def _make_rows(topic_ids: List[str], vecs: List[np.ndarray]) -> List[TopicVectorRow]:
    return [TopicVectorRow(tid, f"Label {tid}", ["ep1"], v) for tid, v in zip(topic_ids, vecs)]


# ---------------------------------------------------------------------------
# Reference: old O(n³) greedy loop (kept here as oracle for equivalence check)
# ---------------------------------------------------------------------------


def _old_cluster_indices_by_threshold(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Verbatim copy of the old implementation for equivalence tests."""
    from typing import Set

    n = int(sim.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    clusters: List[Set[int]] = [{i} for i in range(n)]

    def _mean(ci: Set[int], cj: Set[int]) -> float:
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
                s = _mean(clusters[i], clusters[j])
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


def _partition_sets(labels: np.ndarray) -> List[frozenset]:
    """Convert label array to a sorted list of frozensets (one per unique label)."""
    from collections import defaultdict

    groups: dict = defaultdict(set)
    for i, lab in enumerate(labels.tolist()):
        groups[int(lab)].add(i)
    return sorted(groups.values(), key=lambda s: min(s))


# ---------------------------------------------------------------------------
# (a) Partition equivalence
# ---------------------------------------------------------------------------


def _tie_free_sim_matrix(n: int, threshold: float) -> np.ndarray:
    """Build a pairwise similarity matrix with no threshold ties.

    Items 0..n//2-1 are the "high" cluster (sim > threshold between any two),
    items n//2..n-1 are the "low" cluster (sim much below threshold across groups).
    Within-group similarities are arranged so no two inter-group pairs share the
    same value (avoids the greedy-loop vs UPGMA tie-breaking divergence).
    """
    rng = np.random.default_rng(42)
    vecs = np.zeros((n, 64), dtype=np.float32)
    half = n // 2
    # High cluster: tight cone around e_0
    for i in range(half):
        v = np.zeros(64, dtype=np.float32)
        v[0] = 1.0
        noise = rng.normal(0, 0.04 + i * 0.001, 64).astype(np.float32)
        v += noise
        v /= np.linalg.norm(v)
        vecs[i] = v
    # Low cluster: tight cone around e_1 (orthogonal to e_0)
    for i in range(half, n):
        v = np.zeros(64, dtype=np.float32)
        v[1] = 1.0
        noise = rng.normal(0, 0.04 + i * 0.001, 64).astype(np.float32)
        v += noise
        v /= np.linalg.norm(v)
        vecs[i] = v
    return cast_float32_sim(vecs @ vecs.T)


def cast_float32_sim(m: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, np.clip(m, -1.0, 1.0).astype(np.float64))


def test_partition_equivalence_scipy_matches_old_greedy() -> None:
    """New UPGMA and old greedy produce identical partitions on a tie-free fixture."""
    n = 20
    threshold = 0.80
    sim = _tie_free_sim_matrix(n, threshold)

    new_labels = cluster_indices_by_threshold(sim, threshold)
    old_labels = _old_cluster_indices_by_threshold(sim, threshold)

    new_part = _partition_sets(new_labels)
    old_part = _partition_sets(old_labels)

    assert new_part == old_part, f"Partition mismatch:\n  scipy: {new_part}\n  old:   {old_part}"


def test_non_finite_distance_falls_back_to_singletons_not_crash() -> None:
    """A NaN/inf similarity (e.g. a NaN input embedding) must NOT crash linkage.

    ``squareform(checks=False)`` removed scipy's own finiteness guard, so a non-finite
    distance would raise "must contain only finite values". The guard falls back to
    all-singleton clusters (clusters are non-fatal) instead of crashing the corpus finalize.
    """
    sim = np.array(
        [[1.0, 0.9, np.nan], [0.9, 1.0, 0.2], [np.nan, 0.2, 1.0]],
        dtype=np.float64,
    )
    labels = cluster_indices_by_threshold(sim, 0.5)
    # Each row its own cluster — no crash, deterministic 0..n-1.
    assert labels.tolist() == [0, 1, 2]


def test_partition_equivalence_four_items() -> None:
    """4-item fixture: (0,1) close, (2,3) close, cross-pair orthogonal."""
    e0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    e1_raw = np.array([0.99, 0.1414, 0.0, 0.0], dtype=np.float64)
    e1 = e1_raw / np.linalg.norm(e1_raw)
    e2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    e3_raw = np.array([0.0, 0.0, 0.99, 0.1414], dtype=np.float64)
    e3 = e3_raw / np.linalg.norm(e3_raw)
    mat = np.stack([e0, e1, e2, e3], axis=0)
    sim = mat @ mat.T

    new_labels = cluster_indices_by_threshold(sim, 0.9)
    old_labels = _old_cluster_indices_by_threshold(sim, 0.9)

    assert _partition_sets(new_labels) == _partition_sets(old_labels)
    # Also assert structural shape (two groups)
    assert int(new_labels[0]) == int(new_labels[1])
    assert int(new_labels[2]) == int(new_labels[3])
    assert int(new_labels[0]) != int(new_labels[2])


def test_edge_n0() -> None:
    sim = np.zeros((0, 0), dtype=np.float64)
    assert cluster_indices_by_threshold(sim, 0.8).shape == (0,)


def test_edge_n1() -> None:
    sim = np.array([[1.0]], dtype=np.float64)
    labels = cluster_indices_by_threshold(sim, 0.8)
    assert labels.shape == (1,)
    assert int(labels[0]) == 0


# ---------------------------------------------------------------------------
# (b) Deterministic slugs
# ---------------------------------------------------------------------------


def _two_close_rows() -> List[TopicVectorRow]:
    """Two close vectors + one far; stable across calls."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b_raw = np.array([0.99, 0.1414], dtype=np.float32)
    b = b_raw / float(np.linalg.norm(b_raw))
    c = np.array([0.0, 1.0], dtype=np.float32)
    return [
        TopicVectorRow("topic:alpha", "Alpha", ["ep1"], a),
        TopicVectorRow("topic:beta", "Beta", ["ep1"], b),
        TopicVectorRow("topic:gamma", "Gamma", ["ep2"], c),
    ]


def test_deterministic_slugs_same_input_same_slugs() -> None:
    """Two calls with identical rows produce identical ``tc:`` slugs."""
    rows = _two_close_rows()
    p1 = build_topic_clusters_payload(rows, threshold=0.9, embedding_model="m")
    p2 = build_topic_clusters_payload(rows, threshold=0.9, embedding_model="m")
    slugs1 = [c["graph_compound_parent_id"] for c in p1["clusters"]]
    slugs2 = [c["graph_compound_parent_id"] for c in p2["clusters"]]
    assert slugs1 == slugs2


def test_deterministic_slugs_permuted_order_same_slugs() -> None:
    """Permuting row order does not change slug assignment."""
    rows = _two_close_rows()
    rows_rev = list(reversed(rows))
    p1 = build_topic_clusters_payload(rows, threshold=0.9, embedding_model="m")
    p2 = build_topic_clusters_payload(rows_rev, threshold=0.9, embedding_model="m")
    slugs1 = sorted(c["graph_compound_parent_id"] for c in p1["clusters"])
    slugs2 = sorted(c["graph_compound_parent_id"] for c in p2["clusters"])
    assert slugs1 == slugs2


# ---------------------------------------------------------------------------
# (c) Skip-gate
# ---------------------------------------------------------------------------


def test_skip_gate_unchanged_returns_skipped_flag() -> None:
    """Same rows + matching prior fingerprint → ``skipped_unchanged: True``."""
    rows = _two_close_rows()
    # First run: compute fingerprint
    fp = fingerprint_topic_rows(rows)
    result = build_topic_clusters_payload(
        rows, threshold=0.9, embedding_model="m", prior_fingerprint=fp
    )
    assert result.get("skipped_unchanged") is True
    assert "clusters" not in result


def test_skip_gate_changed_rows_runs_clustering() -> None:
    """Different rows despite a prior fingerprint → clustering executes."""
    rows = _two_close_rows()
    fp = fingerprint_topic_rows(rows)

    # Modify one vector — fingerprint no longer matches.
    changed = list(rows)
    changed[0] = TopicVectorRow(
        "topic:alpha", "Alpha", ["ep1"], np.array([0.5, 0.5], dtype=np.float32)
    )
    result = build_topic_clusters_payload(
        changed, threshold=0.9, embedding_model="m", prior_fingerprint=fp
    )
    assert result.get("skipped_unchanged") is not True
    assert "clusters" in result


def test_skip_gate_none_prior_fingerprint_always_runs() -> None:
    """No prior fingerprint (first run) → clustering always executes."""
    rows = _two_close_rows()
    result = build_topic_clusters_payload(rows, threshold=0.9, embedding_model="m")
    assert "clusters" in result
    assert result.get("skipped_unchanged") is not True


def test_skip_gate_fingerprint_in_payload() -> None:
    """Payload includes ``fingerprint`` key matching ``fingerprint_topic_rows``."""
    rows = _two_close_rows()
    result = build_topic_clusters_payload(rows, threshold=0.9, embedding_model="m")
    assert result.get("fingerprint") == fingerprint_topic_rows(rows)


def test_skip_gate_mismatched_fingerprint_runs() -> None:
    """A stale/wrong fingerprint does not suppress clustering."""
    rows = _two_close_rows()
    result = build_topic_clusters_payload(
        rows, threshold=0.9, embedding_model="m", prior_fingerprint="deadbeef"
    )
    assert "clusters" in result


# ---------------------------------------------------------------------------
# (d) Wall-clock guardrail (ADR-095 Tier-2)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_wallclock_corpus_scale_under_30s() -> None:
    """UPGMA on 2 000 synthetic topic vectors completes in < 30 s.

    This is the regression guard for the O(n³) recompute. A 678-episode
    corpus produces ~2 k unique topic vectors; 2 000 items is conservative.
    """
    rng = np.random.default_rng(0)
    n = 2000
    d = 64
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = (vecs / np.where(norms > 1e-12, norms, 1.0)).astype(np.float32)
    sim = (vecs @ vecs.T).astype(np.float64)

    start = time.monotonic()
    labels = cluster_indices_by_threshold(sim, threshold=0.80)
    elapsed = time.monotonic() - start

    assert labels.shape == (n,), "Expected one label per item"
    assert elapsed < 30.0, f"Clustering took {elapsed:.1f}s — exceeds 30s wall-clock guard"
