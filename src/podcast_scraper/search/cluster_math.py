"""Average-linkage clustering over L2-normalised embeddings, shared by every corpus clusterer.

Topic themes (private, ADR-162) and insight clusters (public) both group vectors this way; the math
lives here so neither has to import the other.
"""

from __future__ import annotations

import logging
from typing import cast, Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


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


#: Rows of the similarity matrix computed at once by :func:`cluster_labels_by_threshold`. The
#: working set is ``block x n`` float32 (~180 MB at 22k vectors), never ``n x n``.
CLUSTER_BLOCK_ROWS = 2048

#: Slack on the pair test that seeds components, so a pair a hair inside the threshold in the
#: full matrix cannot fall a hair outside it in a row block (BLAS may sum in another order). A
#: looser seed only makes components bigger, which never changes the result — see below.
_SEED_SLACK = 1e-6


def cluster_labels_by_threshold(
    vectors: np.ndarray, threshold: float, *, block_rows: int = CLUSTER_BLOCK_ROWS
) -> np.ndarray:
    """The same partition as ``cluster_indices_by_threshold(cosine_similarity_matrix(v), t)``,
    without ever holding the ``n x n`` matrix.

    The full matrix plus its distance copies and scipy's average linkage peaked at 6.8 GiB for
    the 21,977 topics on prod (2026-10-08) and grows with n²: run after every index update, it
    was the process the kernel OOM-killed on the 15 GB box (05:08 in the api, 08:38 in a
    pipeline run).

    Why it is exact: average linkage merges two clusters only when their MEAN distance is within
    the cut, so at least one pair across them is within it. Every final cluster therefore lies
    inside one connected component of the "pair within the cut" graph, and average linkage over
    a component depends only on that component's members. So: find the pairs a row block at a
    time, union them into components, and cluster each component on its own (most are single
    topics). On the prod topics: identical partition (18,464 clusters), 1.3 GiB peak, 2.0 s
    instead of 7.5 s.

    MEMORY IS BOUNDED BY THE LARGEST COMPONENT, not by n: inside one component the clustering is
    still quadratic. On prod the largest was 3,139 of 21,977 topics; it is logged on every run so
    a component growing toward n shows up before it costs memory again.

    One deliberate difference: a NaN/inf vector no longer turns EVERY cluster into a singleton
    (the full-matrix fallback); it stays a singleton itself and the rest cluster normally.
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2-D")
    n = int(vectors.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.int64)
    cut = 1.0 - threshold
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    step = max(1, int(block_rows))
    for start in range(0, n, step):
        dist = np.clip(1.0 - (vectors[start : start + step] @ vectors.T), 0.0, 2.0)
        rows, cols = np.nonzero(dist <= cut + _SEED_SLACK)
        for i, j in zip((rows + start).tolist(), cols.tolist()):
            if j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    components: Dict[int, List[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)
    labels = np.empty((n,), dtype=np.int64)
    next_label = 0
    logger.info(
        "clustering: %d vectors, %d components, largest %d",
        n,
        len(components),
        max(len(m) for m in components.values()),
    )
    for members in components.values():
        if len(members) == 1:
            labels[members[0]] = next_label
            next_label += 1
            continue
        idx = np.asarray(members, dtype=np.int64)
        local = cluster_indices_by_threshold(cosine_similarity_matrix(vectors[idx]), threshold)
        labels[idx] = local + next_label
        next_label += int(local.max()) + 1
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
