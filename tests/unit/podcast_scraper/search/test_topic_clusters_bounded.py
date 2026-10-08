"""Topic clustering without the n x n matrix: same partition, bounded memory.

Prod 2026-10-08: clustering 21,977 topics through the full similarity matrix peaked at 6.8 GiB
and was the process the kernel OOM-killed twice on the 15 GB server. ``cluster_labels_by_threshold``
must give exactly the old partition while never holding an n x n array.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from podcast_scraper.search import cluster_math as tc

pytestmark = [pytest.mark.unit]


def _clustered(seed: int, n: int = 300, dim: int = 24) -> np.ndarray:
    """Normalised vectors in loose groups, so thresholds cut through real structure."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(max(2, n // 12), dim))
    v = centres[rng.integers(0, len(centres), size=n)] + rng.normal(scale=0.6, size=(n, dim))
    v = v.astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _partition(labels) -> list:
    groups: dict = {}
    for i, lab in enumerate(np.asarray(labels).tolist()):
        groups.setdefault(lab, []).append(i)
    return sorted(tuple(g) for g in groups.values())


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("threshold", [0.5, 0.7, 0.85])
def test_same_partition_as_the_full_matrix(seed, threshold) -> None:
    v = _clustered(seed)
    old = tc.cluster_indices_by_threshold(tc.cosine_similarity_matrix(v), threshold)
    for block in (7, 64, tc.CLUSTER_BLOCK_ROWS):
        new = tc.cluster_labels_by_threshold(v, threshold, block_rows=block)
        assert _partition(new) == _partition(old), (seed, threshold, block)


def test_labels_are_contiguous_from_zero() -> None:
    labels = tc.cluster_labels_by_threshold(_clustered(3), 0.7)
    assert sorted(set(labels.tolist())) == list(range(len(set(labels.tolist()))))


def _sparse_topics(seed: int, n: int, dim: int = 64) -> np.ndarray:
    """Like prod topics: mostly unrelated, a few tight near-duplicate groups."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n // 3, dim))
    v = centres[rng.integers(0, len(centres), size=n)] + rng.normal(scale=0.15, size=(n, dim))
    v = v.astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_memory_stays_below_one_full_matrix() -> None:
    # The bound is the largest connected component, not n: on prod 3,139 of 21,977 topics.
    n = 3000
    v = _sparse_topics(5, n=n)
    full_matrix_bytes = n * n * 4

    tracemalloc.start()
    tc.cluster_indices_by_threshold(tc.cosine_similarity_matrix(v), 0.7)
    _cur, old_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    tc.cluster_labels_by_threshold(v, 0.7, block_rows=256)
    _cur, new_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert old_peak > full_matrix_bytes  # the old path really holds n x n
    assert new_peak < full_matrix_bytes / 3, (new_peak, full_matrix_bytes)


def test_edge_sizes() -> None:
    assert tc.cluster_labels_by_threshold(np.zeros((0, 4), dtype=np.float32), 0.7).shape == (0,)
    assert tc.cluster_labels_by_threshold(np.ones((1, 4), dtype=np.float32), 0.7).tolist() == [0]
    with pytest.raises(ValueError):
        tc.cluster_labels_by_threshold(np.ones(4, dtype=np.float32), 0.7)


def test_a_nan_vector_stays_alone_and_does_not_flatten_the_rest() -> None:
    # The full-matrix path falls back to ALL singletons on any non-finite distance; here only
    # the poisoned vector is alone and the rest cluster as before.
    v = _clustered(8)
    poisoned = v.copy()
    poisoned[0] = np.nan
    labels = tc.cluster_labels_by_threshold(poisoned, 0.7)
    assert int((labels == labels[0]).sum()) == 1
    rest = [g for g in _partition(labels) if 0 not in g]
    # ...and the others cluster exactly as they would without vector 0 at all.
    without = tc.cluster_labels_by_threshold(v[1:], 0.7)
    assert sorted(rest) == sorted(tuple(i + 1 for i in g) for g in _partition(without))
