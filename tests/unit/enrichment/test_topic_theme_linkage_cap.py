"""The average-linkage clusterer degrades to no-clustering above _MAX_LINKAGE_TOPICS.

Above the cap the O(n^4)-worst-case merge would burn a CPU core past the timeout
in an uncancellable sync thread, so it returns singletons instead (low/theme-linkage).
"""

from __future__ import annotations

import pytest

from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
    _average_linkage,
    _MAX_LINKAGE_TOPICS,
)

pytestmark = pytest.mark.unit


def test_above_cap_returns_singletons_without_clustering():
    n = _MAX_LINKAGE_TOPICS + 1
    # A uniformly-high weight would normally merge everything into one cluster;
    # above the cap it must short-circuit to n singletons instead.
    out = _average_linkage(n, lambda i, j: 1.0, threshold=0.1)
    assert out == [{i} for i in range(n)]


def test_below_cap_merges_connected_members():
    # 0-1 strongly linked, 2 isolated: average-linkage merges {0,1}, leaves {2}.
    def weight(i: int, j: int) -> float:
        return 1.0 if {i, j} == {0, 1} else 0.0

    out = _average_linkage(3, weight, threshold=0.5)
    assert {0, 1} in out and {2} in out and len(out) == 2
