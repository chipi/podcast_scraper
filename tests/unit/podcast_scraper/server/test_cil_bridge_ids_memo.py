"""The bridge ids of the cached bundles are computed once per bundle list, not per query."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import cil_queries

pytestmark = [pytest.mark.unit]

FIXTURE = Path("tests/fixtures/app-validation-corpus/v3")


@pytest.fixture(autouse=True)
def _clean():
    perf_cache.clear()
    yield
    perf_cache.clear()


def test_bridge_ids_of_the_cached_bundles_are_the_computed_ones_and_are_not_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = os.path.abspath(FIXTURE)
    bundles = cil_queries._cached_bridge_bundles(root, root)
    assert bundles, "fixture corpus has no bridges"
    calls: list[int] = []
    real = cil_queries._compute_bridge_all_ids

    def _counting(bridge: Any) -> Any:
        calls.append(1)
        return real(bridge)

    monkeypatch.setattr(cil_queries, "_compute_bridge_all_ids", _counting)
    for _path, bridge in bundles:
        assert cil_queries._bridge_all_ids(bridge) == real(bridge)
    # one call per bundle is the comparison's own ``real(bridge)``; the memo added none
    assert calls == []
    # A bridge the cache does not hold is computed, never remembered.
    stranger = {"identities": [{"id": " topic:x "}, {"id": 3}, "junk"]}
    assert cil_queries._bridge_all_ids(stranger) == frozenset({"topic:x"})
    assert id(stranger) not in cil_queries._BUNDLE_IDS


def test_a_rebuilt_bundle_list_replaces_the_old_entries() -> None:
    root = os.path.abspath(FIXTURE)
    first = cil_queries._cached_bridge_bundles(root, root)
    old_ids = {id(b) for _p, b in first}
    perf_cache.clear()
    second = cil_queries._cached_bridge_bundles(root, root)
    assert second is not first
    assert not ((old_ids - {id(b) for _p, b in second}) & set(cil_queries._BUNDLE_IDS))
