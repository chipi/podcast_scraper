"""Unit tests for the cross-layer corpus graph (search/corpus_graph.py, #849 Slice B)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.corpus_graph import (
    clear_corpus_graph_cache,
    CorpusGraph,
    get_corpus_graph,
)

pytestmark = pytest.mark.unit


# A GI artifact and a KG artifact for the SAME episode, sharing person:alice,
# topic:ai, and episode:e1 — the cross-layer join surface.
_GI = {
    "schema_version": "1.0",
    "episode_id": "e1",
    "nodes": [
        {"id": "podcast:show1", "type": "Podcast", "properties": {"title": "Show One"}},
        {"id": "episode:e1", "type": "Episode", "properties": {"title": "Ep 1"}},
        {
            "id": "person:alice",
            "type": "Person",
            "properties": {"name": "Alice", "aliases": ["Al"]},
        },
        {"id": "topic:ai", "type": "Topic", "properties": {"label": "AI"}},
        {
            "id": "insight:i1",
            "type": "Insight",
            "properties": {"text": "AI insight", "episode_id": "e1", "grounded": True},
        },
        {
            "id": "quote:q1",
            "type": "Quote",
            "properties": {"text": "quote text", "episode_id": "e1"},
        },
    ],
    "edges": [
        {"type": "HAS_EPISODE", "from": "podcast:show1", "to": "episode:e1"},
        {"type": "HAS_INSIGHT", "from": "episode:e1", "to": "insight:i1"},
        {"type": "ABOUT", "from": "insight:i1", "to": "topic:ai"},
        {"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"},
        {"type": "SPOKEN_BY", "from": "quote:q1", "to": "person:alice"},
    ],
}

_KG = {
    "schema_version": "1.2",
    "episode_id": "e1",
    "nodes": [
        {"id": "episode:e1", "type": "Episode", "properties": {"title": "Ep 1"}},
        {
            "id": "person:alice",
            "type": "Entity",
            "properties": {"name": "Alice", "kind": "person", "role": "guest"},
        },
        {"id": "org:acme", "type": "Entity", "properties": {"name": "Acme", "kind": "org"}},
        {"id": "topic:ai", "type": "Topic", "properties": {"label": "AI", "slug": "ai"}},
        # Layer-prefixed id must canonicalize to topic:ai (strip_layer_prefixes).
        {"id": "kg:topic:ai", "type": "Topic", "properties": {"slug": "ai"}},
    ],
    "edges": [
        {"type": "MENTIONS", "from": "person:alice", "to": "episode:e1"},
        {"type": "MENTIONS", "from": "topic:ai", "to": "episode:e1"},
        {"type": "MENTIONS", "from": "org:acme", "to": "episode:e1"},
    ],
}


def _write_corpus(tmp_path: Path, gi=_GI, kg=_KG) -> Path:
    (tmp_path / "ep1.gi.json").write_text(json.dumps(gi))
    (tmp_path / "ep1.kg.json").write_text(json.dumps(kg))
    return tmp_path


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_corpus_graph_cache()
    yield
    clear_corpus_graph_cache()


# --- cross-layer join ----------------------------------------------------------


def test_person_node_unified_across_layers(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    node = g.get_node("person:alice")
    assert node is not None
    assert node.type == "person"
    assert node.layers == {"gi", "kg"}
    # Merged payload: GI name + aliases AND KG kind/role.
    assert node.payload["name"] == "Alice"
    assert node.payload["kind"] == "person"
    assert node.payload["role"] == "guest"
    assert node.payload["aliases"] == ["Al"]


def test_neighbors_span_both_layers(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    nbrs = set(g.neighbors("person:alice"))
    assert "quote:q1" in nbrs  # GIL SPOKEN_BY
    assert "episode:e1" in nbrs  # KG MENTIONS


def test_layer_prefixed_id_canonicalizes(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    assert g.get_node("topic:ai") is not None
    assert g.get_node("kg:topic:ai") is None  # canonicalized into topic:ai


# --- traversal -----------------------------------------------------------------


def test_bfs_reaches_insight_from_person(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    dist = g.bfs("person:alice", max_hops=3)
    # person:alice -SPOKEN_BY- quote:q1 -SUPPORTED_BY- insight:i1  (2 hops)
    # (also person -MENTIONS- episode -HAS_INSIGHT- insight = 2 hops)
    assert dist.get("insight:i1") == 2
    assert dist["person:alice"] == 0


def test_bfs_respects_max_hops(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    dist = g.bfs("person:alice", max_hops=1)
    assert "insight:i1" not in dist  # 2 hops away
    assert "episode:e1" in dist and dist["episode:e1"] == 1


def test_bfs_unknown_start_returns_empty(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    assert g.bfs("person:nobody") == {}


# --- access helpers ------------------------------------------------------------


def test_nodes_by_type_and_degree(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    assert g.nodes_by_type("insight") == ["insight:i1"]
    assert "podcast:show1" in g.nodes_by_type("podcast")
    # episode:e1 connects to: podcast, insight, person, topic, org -> degree 5
    assert g.degree("episode:e1") == 5
    assert g.get_node("missing") is None
    assert g.get_node(None) is None


def test_show_connectivity_present(tmp_path):
    # FROM_SHOW is satisfied natively: Podcast --HAS_EPISODE--> Episode.
    g = CorpusGraph.build(_write_corpus(tmp_path))
    assert "episode:e1" in g.neighbors("podcast:show1")


def test_empty_corpus_builds_empty_graph(tmp_path):
    g = CorpusGraph.build(tmp_path)
    assert len(g) == 0
    assert g.neighbors("anything") == []


# --- cache ---------------------------------------------------------------------


def test_get_corpus_graph_caches(tmp_path):
    _write_corpus(tmp_path)
    g1 = get_corpus_graph(tmp_path)
    g2 = get_corpus_graph(tmp_path)
    assert g1 is g2
    clear_corpus_graph_cache()
    assert get_corpus_graph(tmp_path) is not g1


def test_build_from_real_fixture_corpus():
    from tests._fixtures import fixtures_dir

    g = CorpusGraph.build(fixtures_dir("viewer-validation-corpus"))
    assert len(g) > 0


# --- Slice C: derived person -> insight shortcut -------------------------------


def test_derive_speaker_links_adds_one_hop_person_to_insight(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path), derive_speaker_links=True)
    assert "insight:i1" in g.neighbors("person:alice")
    assert g.bfs("person:alice", max_hops=1).get("insight:i1") == 1


def test_derive_off_by_default_keeps_insight_two_hops(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    assert "insight:i1" not in g.neighbors("person:alice")
    assert g.bfs("person:alice").get("insight:i1") == 2


def test_get_corpus_graph_caches_per_derive_flag(tmp_path):
    _write_corpus(tmp_path)
    plain = get_corpus_graph(tmp_path)
    derived = get_corpus_graph(tmp_path, derive_speaker_links=True)
    assert plain is not derived
    assert "insight:i1" not in plain.neighbors("person:alice")
    assert "insight:i1" in derived.neighbors("person:alice")


# --- #882: typed adjacency (relational-query substrate) -------------------------


def test_typed_neighbors_filters_by_edge_type(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    # insight:i1 has an ABOUT edge to topic:ai and a SUPPORTED_BY edge to quote:q1.
    assert g.typed_neighbors("insight:i1", "ABOUT") == ["topic:ai"]
    assert g.typed_neighbors("insight:i1", "SUPPORTED_BY") == ["quote:q1"]
    # HAS_INSIGHT is the episode edge — not present from the insight via ABOUT.
    assert "episode:e1" in g.typed_neighbors("insight:i1", "HAS_INSIGHT")


def test_typed_neighbors_is_undirected(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    # HAS_EPISODE declared podcast→episode; reachable from either endpoint.
    assert g.typed_neighbors("podcast:show1", "HAS_EPISODE") == ["episode:e1"]
    assert g.typed_neighbors("episode:e1", "HAS_EPISODE") == ["podcast:show1"]


def test_derive_speaker_links_emits_states_edge(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path), derive_speaker_links=True)
    # The synthesized person→insight shortcut carries the STATES type, so the
    # relational layer can distinguish "stated" from "mentioned".
    assert g.typed_neighbors("person:alice", "STATES") == ["insight:i1"]


def test_typed_neighbors_unknown_returns_empty(tmp_path):
    g = CorpusGraph.build(_write_corpus(tmp_path))
    assert g.typed_neighbors("person:nobody", "STATES") == []
    assert g.typed_neighbors("insight:i1", "NO_SUCH_TYPE") == []


# --- #852: identity_map collapses cross-episode variant nodes -------------------


def _write_variant_corpus(tmp_path):
    # Two episodes, same show: org:cargill (ep1) and org:cargil (ep2, a spelling variant).
    for ep, org in (("e1", "org:cargill"), ("e2", "org:cargil")):
        (tmp_path / f"{ep}.kg.json").write_text(
            json.dumps(
                {
                    "episode_id": ep,
                    "nodes": [
                        {
                            "id": f"episode:{ep}",
                            "type": "Episode",
                            "properties": {"podcast_id": "show:lots"},
                        },
                        {"id": org, "type": "Entity", "properties": {"name": org.split(":")[1]}},
                    ],
                    "edges": [{"type": "MENTIONS", "from": org, "to": f"episode:{ep}"}],
                }
            )
        )
    return tmp_path


def test_identity_map_collapses_variant_nodes(tmp_path):
    corpus = _write_variant_corpus(tmp_path)
    g = CorpusGraph.build(corpus, identity_map={"org:cargil": "org:cargill"})
    assert g.get_node("org:cargil") is None  # variant collapsed
    assert g.get_node("org:cargill") is not None
    # The remapped node reaches BOTH episodes (its own + the variant's).
    nbrs = set(g.neighbors("org:cargill"))
    assert {"episode:e1", "episode:e2"} <= nbrs


def test_no_identity_map_keeps_variants_separate(tmp_path):
    g = CorpusGraph.build(_write_variant_corpus(tmp_path))
    assert g.get_node("org:cargil") is not None
    assert g.get_node("org:cargill") is not None


def test_get_corpus_graph_canonicalizes_entities_by_default(tmp_path):
    # Prod path: get_corpus_graph builds + applies the entity canonical map (#852).
    corpus = _write_variant_corpus(tmp_path)
    g = get_corpus_graph(corpus)  # canonicalize_entities=True by default
    assert g.get_node("org:cargil") is None  # auto-merged into the canonical
    assert g.get_node("org:cargill") is not None
    clear_corpus_graph_cache()
    g_off = get_corpus_graph(corpus, canonicalize_entities=False)
    assert g_off.get_node("org:cargil") is not None  # faithful union when off
