"""The faster linkage must produce identical clusters to the old one (#1972).

``_average_linkage`` used to recompute a full ``|ci|x|cj|`` sum for every candidate pair on every
round — the O(n^4) that the 400-topic cap existed to contain. Above that cap it degraded to
all-singletons, and all-singletons means every cluster is dropped downstream: ZERO themes, while
the run still reports ``status=ok``. Prod sat at 199 of 400 with the corpus planned to grow ~4x.

Lance-Williams gives the identical answer incrementally. This test is the proof: it runs the
ORIGINAL algorithm alongside the new one on randomised weighted graphs and demands the same
clusters, including the same tie-breaking — theme labels depend on which pair merged first.
"""

from __future__ import annotations

import random
import time

import pytest

from podcast_scraper.enrichment.enrichers import topic_theme_clusters as ttc

pytestmark = pytest.mark.unit


def _reference_average_linkage(n, weight, threshold):
    """The pre-#1972 implementation, verbatim, as the oracle."""
    clusters = [{i} for i in range(n)]

    def mean_inter(ci, cj):
        tot = 0.0
        for a in ci:
            for b in cj:
                tot += weight(a, b)
        return tot / (len(ci) * len(cj))

    while len(clusters) > 1:
        best = -1.0
        bi, bj = -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = mean_inter(clusters[i], clusters[j])
                if s > best:
                    best = s
                    bi, bj = i, j
        if bi < 0 or best < threshold:
            break
        clusters[bi] |= clusters[bj]
        clusters.pop(bj)
    return clusters


def _as_sorted(clusters):
    return sorted(sorted(c) for c in clusters)


def _make_weight(w):
    def weight(a, b):
        if a == b:
            return 0.0
        return w[(a, b)] if a < b else w[(b, a)]

    return weight


@pytest.mark.parametrize("seed", range(25))
def test_matches_the_reference_on_random_graphs(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(2, 14)
    threshold = rng.choice([0.0, 0.1, 0.25, 0.5, 0.75])
    # Sparse and tie-heavy on purpose: ties are where an "equivalent" rewrite quietly diverges.
    w = {
        (i, j): rng.choice([0.0, 0.0, 0.25, 0.5, 0.5, 1.0])
        for i in range(n)
        for j in range(i + 1, n)
    }
    weight = _make_weight(w)

    assert _as_sorted(ttc._average_linkage(n, weight, threshold)) == _as_sorted(
        _reference_average_linkage(n, weight, threshold)
    ), f"diverged from the reference at seed={seed}, n={n}, threshold={threshold}"


def test_all_ties_still_resolve_deterministically() -> None:
    """A fully tied graph is the worst case for tie-break drift."""
    n = 8

    def weight(a, b):
        return 0.0 if a == b else 0.5

    assert _as_sorted(ttc._average_linkage(n, weight, 0.1)) == _as_sorted(
        _reference_average_linkage(n, weight, 0.1)
    )


def test_degenerate_inputs() -> None:
    assert ttc._average_linkage(0, lambda a, b: 0.0, 0.1) == []
    assert ttc._average_linkage(1, lambda a, b: 0.0, 0.1) == [{0}]
    # Nothing clears the threshold -> all singletons, same as before.
    assert _as_sorted(ttc._average_linkage(4, lambda a, b: 0.0, 0.5)) == [[0], [1], [2], [3]]


def test_the_cap_is_no_longer_the_binding_constraint() -> None:
    """A corpus 4x today's must not silently switch theme generation off."""
    assert ttc._MAX_LINKAGE_TOPICS >= 2000


def test_scales_past_the_old_cap_without_timing_out() -> None:
    """450 topics — impractical under the old O(n^4) — must complete promptly."""
    n = 450
    rng = random.Random(7)
    dense = {(i, j): rng.choice([0.0, 0.0, 0.0, 0.9]) for i in range(n) for j in range(i + 1, n)}
    weight = _make_weight(dense)

    t0 = time.time()
    out = ttc._average_linkage(n, weight, 0.5)
    elapsed = time.time() - t0
    assert elapsed < 60, f"linkage still too slow above the old cap: {elapsed:.1f}s"
    assert sum(len(c) for c in out) == n, "every topic must survive into exactly one cluster"
