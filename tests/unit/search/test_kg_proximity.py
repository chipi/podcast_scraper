"""Unit tests for the KG-proximity signal + RetrievalLayer wiring (RFC-091, #859)."""

from __future__ import annotations

import json

import pytest

from podcast_scraper.search.backend import ScoredResult
from podcast_scraper.search.corpus_graph import CorpusGraph
from podcast_scraper.search.kg_proximity import KGProximitySearch
from podcast_scraper.search.retrieval import RetrievalLayer

pytestmark = pytest.mark.unit


def _write_chain_corpus(tmp_path):
    # person:alice -SPOKEN_BY- quote:q1 -SUPPORTED_BY- insight:i1 -HAS_INSIGHT- episode:e1
    gi = {
        "episode_id": "e1",
        "nodes": [
            {"id": "podcast:show1", "type": "Podcast", "properties": {}},
            {"id": "episode:e1", "type": "Episode", "properties": {}},
            {"id": "person:alice", "type": "Person", "properties": {"name": "Alice"}},
            {"id": "topic:ai", "type": "Topic", "properties": {"label": "AI"}},
            {
                "id": "insight:i1",
                "type": "Insight",
                "properties": {"text": "AI insight", "episode_id": "e1"},
            },
            {"id": "quote:q1", "type": "Quote", "properties": {"episode_id": "e1"}},
        ],
        "edges": [
            {"type": "HAS_EPISODE", "from": "podcast:show1", "to": "episode:e1"},
            {"type": "HAS_INSIGHT", "from": "episode:e1", "to": "insight:i1"},
            {"type": "ABOUT", "from": "insight:i1", "to": "topic:ai"},
            {"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"},
            {"type": "SPOKEN_BY", "from": "quote:q1", "to": "person:alice"},
        ],
    }
    (tmp_path / "e1.gi.json").write_text(json.dumps(gi))
    return tmp_path


# --- KGProximitySearch ----------------------------------------------------------


def test_proximity_returns_insight_with_hop_score(tmp_path):
    graph = CorpusGraph.build(_write_chain_corpus(tmp_path))
    res = KGProximitySearch(graph).search("person:alice")
    # alice→quote(1)→insight(2): the insight is the only result-typed node within 3 hops.
    assert len(res) == 1
    r = res[0]
    assert r.doc_id == "insight:i1" and r.signal == "kg" and r.source_tier == "insight"
    assert r.score == pytest.approx(1.0 / 3.0)  # 1/(hop+1), hop=2
    assert r.rank == 1


def test_proximity_skips_non_result_node_types(tmp_path):
    graph = CorpusGraph.build(_write_chain_corpus(tmp_path))
    res = KGProximitySearch(graph).search("person:alice")
    returned = {r.doc_id for r in res}
    assert "quote:q1" not in returned and "topic:ai" not in returned
    assert "episode:e1" not in returned and "person:alice" not in returned


def test_proximity_filters(tmp_path):
    graph = CorpusGraph.build(_write_chain_corpus(tmp_path))
    kg = KGProximitySearch(graph)
    assert kg.search("person:alice", filters={"episode_id": "e1"})  # matches
    assert kg.search("person:alice", filters={"episode_id": "e2"}) == []  # filtered out


def test_proximity_unknown_entity_returns_empty(tmp_path):
    graph = CorpusGraph.build(_write_chain_corpus(tmp_path))
    assert KGProximitySearch(graph).search("person:nobody") == []


def test_proximity_max_hops_bounds_reach(tmp_path):
    graph = CorpusGraph.build(_write_chain_corpus(tmp_path))
    # insight is 2 hops away; max_hops=1 cannot reach it.
    assert KGProximitySearch(graph, max_hops=1).search("person:alice") == []


# --- RetrievalLayer wiring ------------------------------------------------------


class _FakeBackend:
    def search_bm25(self, query):
        return [ScoredResult("seg1", 1.0, 1, {}, "bm25", "segment")]

    def search_vector(self, query):
        return [ScoredResult("ins0", 1.0, 1, {}, "vector", "insight")]


class _FakeKG:
    def __init__(self, results):
        self._results = results

    def search(self, entity_id, *, k=20, filters=None):
        return list(self._results)


class _FakeResolver:
    def __init__(self, entity_id):
        self._entity_id = entity_id

    def resolve(self, text):
        return self._entity_id


def test_retrieval_includes_kg_signal_when_entity_resolves():
    kg_hit = ScoredResult("ins_kg", 0.5, 1, {}, "kg", "insight")
    layer = RetrievalLayer(
        _FakeBackend(),
        kg_proximity=_FakeKG([kg_hit]),
        entity_resolver=_FakeResolver("person:alice"),
    )
    out = layer.retrieve("Sam Altman", [0.1, 0.2])
    assert "ins_kg" in {r.doc_id for r in out}  # KG hit entered the fused results


def test_retrieval_skips_kg_when_entity_unresolved():
    layer = RetrievalLayer(
        _FakeBackend(),
        kg_proximity=_FakeKG([ScoredResult("ins_kg", 0.5, 1, {}, "kg", "insight")]),
        entity_resolver=_FakeResolver(None),  # no entity
    )
    out = layer.retrieve("nothing here", [0.1])
    assert "ins_kg" not in {r.doc_id for r in out}


def test_retrieval_without_kg_components_unchanged():
    layer = RetrievalLayer(_FakeBackend())
    out = layer.retrieve("Sam Altman", [0.1])
    assert {r.doc_id for r in out} == {"seg1", "ins0"}
