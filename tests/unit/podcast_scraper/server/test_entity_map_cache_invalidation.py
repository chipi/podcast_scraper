"""The canonical entity map must not be cached for the life of the process (advisor S6).

Two caches held the variant->canonical map with **no invalidation token at all** — keyed on the
corpus path alone:

* ``search/corpus_graph.get_corpus_graph`` (``_corpus_graphs``)
* ``server/cil_queries._cil_entity_id_map`` (``_cil_id_maps``)

``clear_corpus_graph_cache`` exists but has no callers anywhere under ``src/``, so neither map was
ever rebuilt — not after a corpus migration, not after a ``relabel_only`` re-enrich, not after an
ordinary ingest. Only a process restart.

That is one step worse than the mtime-token gap those siblings had: there is no token to be stale.
An operator who migrates the corpus and then reads who-said / positions / relational sees the
pre-migration answer indefinitely, and the surfaces disagree with ``kg.json`` with nothing to
indicate why.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from podcast_scraper import perf_cache

pytestmark = [pytest.mark.unit]


def _corpus(root: Path, mtime: float) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    stamp = root / "corpus_run_summary.json"
    stamp.write_text("{}", encoding="utf-8")
    os.utime(stamp, (mtime, mtime))


class TestTheCilEntityMapFollowsTheCorpus:
    def test_a_changed_corpus_yields_a_fresh_map(self, tmp_path: Path) -> None:
        from podcast_scraper.server import cil_queries

        cil_queries._cil_id_maps.clear()
        _corpus(tmp_path, 1000.0)
        cil_queries._cil_entity_id_map(str(tmp_path))
        first_keys = set(cil_queries._cil_id_maps)
        assert first_keys, "the first call must cache something"

        _corpus(tmp_path, 2000.0)
        cil_queries._cil_entity_id_map(str(tmp_path))
        assert (
            set(cil_queries._cil_id_maps) != first_keys
        ), "the cache key must carry the corpus token, so a migration invalidates it"

    def test_an_unchanged_corpus_is_served_from_cache(self, tmp_path: Path) -> None:
        # The cache must still BE a cache — this is a hot serving path.
        from podcast_scraper.server import cil_queries

        cil_queries._cil_id_maps.clear()
        _corpus(tmp_path, 1000.0)
        cil_queries._cil_entity_id_map(str(tmp_path))
        n = len(cil_queries._cil_id_maps)
        cil_queries._cil_entity_id_map(str(tmp_path))
        assert len(cil_queries._cil_id_maps) == n, "a second call on an unchanged corpus adds none"


class TestTheCorpusGraphFollowsTheCorpus:
    def test_a_changed_corpus_yields_a_fresh_graph(self, tmp_path: Path) -> None:
        from podcast_scraper.search import corpus_graph

        corpus_graph.clear_corpus_graph_cache()
        _corpus(tmp_path, 1000.0)
        corpus_graph.get_corpus_graph(tmp_path)
        first_keys = set(corpus_graph._corpus_graphs)
        assert first_keys

        _corpus(tmp_path, 2000.0)
        corpus_graph.get_corpus_graph(tmp_path)
        assert (
            set(corpus_graph._corpus_graphs) != first_keys
        ), "the graph cache key must carry the corpus token"

    def test_an_unchanged_corpus_is_served_from_cache(self, tmp_path: Path) -> None:
        from podcast_scraper.search import corpus_graph

        corpus_graph.clear_corpus_graph_cache()
        _corpus(tmp_path, 1000.0)
        a = corpus_graph.get_corpus_graph(tmp_path)
        b = corpus_graph.get_corpus_graph(tmp_path)
        assert a is b, "an unchanged corpus must return the SAME object, not a rebuild"


class TestTheTokenIsTheSharedOne:
    """Both caches key on ``perf_cache.corpus_mtime`` so they invalidate together.

    Two caches with two different notions of "the corpus changed" is how one surface starts
    disagreeing with another — the failure mode this whole arc keeps rediscovering.
    """

    def test_the_migration_ledger_is_part_of_the_token(self, tmp_path: Path) -> None:
        _corpus(tmp_path, 1000.0)
        before = perf_cache.corpus_mtime(tmp_path)
        ledger = tmp_path / "upgrade_ledger.json"
        ledger.write_text("{}", encoding="utf-8")
        os.utime(ledger, (2000.0, 2000.0))
        assert perf_cache.corpus_mtime(tmp_path) != before
