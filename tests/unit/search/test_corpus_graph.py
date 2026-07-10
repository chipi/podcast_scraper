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
    "schema_version": "3.0",
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
    "schema_version": "3.0",
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


# --- #1056: feed-anchored host reconciliation ----------------------------------


def _kg_episode(
    podcast_id: str,
    episode_id: str,
    *,
    host_id: str,
    host_name: str,
    role: str = "host",
    title: str = "A Show",
):
    """A minimal KG artifact: one episode of *podcast_id* with one host voice."""
    return {
        "episode_id": episode_id,
        "nodes": [
            {"id": podcast_id, "type": "Podcast", "properties": {"title": title}},
            {"id": episode_id, "type": "Episode", "properties": {}},
            {"id": host_id, "type": "Person", "properties": {"name": host_name, "role": role}},
        ],
        "edges": [
            {"type": "HAS_EPISODE", "from": podcast_id, "to": episode_id},
            {"type": "MENTIONS", "from": host_id, "to": episode_id},
        ],
    }


def _graph_from(*artifacts) -> CorpusGraph:
    g = CorpusGraph()
    for art in artifacts:
        g._ingest(art, "kg")
    return g


def test_reconcile_names_unnamed_host_from_sibling_episode():
    # Network feed: host named "Katie Martin" in 2 episodes, a bare SPEAKER_03 in a third.
    g = _graph_from(
        _kg_episode(
            "podcast:unhedged",
            "episode:u1",
            host_id="person:katie-martin",
            host_name="Katie Martin",
        ),
        _kg_episode(
            "podcast:unhedged",
            "episode:u2",
            host_id="person:katie-martin",
            host_name="Katie Martin",
        ),
        _kg_episode(
            "podcast:unhedged", "episode:u3", host_id="person:speaker-03", host_name="SPEAKER_03"
        ),
    )
    g._reconcile_feed_hosts()
    # The unnamed voice is gone, merged into the recurring named host…
    assert g.get_node("person:speaker-03") is None
    katie = g.get_node("person:katie-martin")
    assert katie is not None and katie.payload["name"] == "Katie Martin"
    # …and it now reaches the previously-orphaned third episode.
    assert "episode:u3" in g.neighbors("person:katie-martin")


def test_reconcile_is_noop_without_flag(tmp_path):
    corpus = tmp_path
    (corpus / "u1.kg.json").write_text(
        json.dumps(
            _kg_episode(
                "podcast:unhedged",
                "episode:u1",
                host_id="person:katie-martin",
                host_name="Katie Martin",
            )
        )
    )
    (corpus / "u2.kg.json").write_text(
        json.dumps(
            _kg_episode(
                "podcast:unhedged",
                "episode:u2",
                host_id="person:katie-martin",
                host_name="Katie Martin",
            )
        )
    )
    (corpus / "u3.kg.json").write_text(
        json.dumps(
            _kg_episode(
                "podcast:unhedged",
                "episode:u3",
                host_id="person:speaker-03",
                host_name="SPEAKER_03",
            )
        )
    )
    plain = CorpusGraph.build(corpus)
    assert plain.get_node("person:speaker-03") is not None  # untouched by default
    reconciled = CorpusGraph.build(corpus, reconcile_hosts=True)
    assert reconciled.get_node("person:speaker-03") is None  # opt-in merges


def test_reconcile_skips_cohost_show_but_tags_recurring_voice():
    # Two recurring named hosts → ambiguous, so the unnamed voice is NOT merged,
    # but (recurring across 2 episodes) it gets an honest "recurring host" note.
    g = _graph_from(
        _kg_episode(
            "podcast:fx", "episode:f1", host_id="person:katie", host_name="Katie", title="FX Show"
        ),
        _kg_episode("podcast:fx", "episode:f2", host_id="person:katie", host_name="Katie"),
        _kg_episode("podcast:fx", "episode:f3", host_id="person:rob", host_name="Rob"),
        _kg_episode("podcast:fx", "episode:f4", host_id="person:rob", host_name="Rob"),
        _kg_episode(
            "podcast:fx", "episode:f5", host_id="person:speaker-07", host_name="SPEAKER_07"
        ),
        _kg_episode(
            "podcast:fx", "episode:f6", host_id="person:speaker-07", host_name="SPEAKER_07"
        ),
    )
    g._reconcile_feed_hosts()
    voice = g.get_node("person:speaker-07")
    assert voice is not None  # not merged (co-host ambiguity)
    note = voice.payload.get("recurring_host_note", "")
    assert note.startswith("recurring host of") and "not auto-named" in note


def test_reconcile_merges_episode_scoped_unnamed_hosts_into_recurring_named_host():
    # #1b: unnamed voices are episode-scoped (person:speaker-<ep>-NN), so a recurring unnamed
    # host is TWO singleton nodes under one show, not one shared node. The per-podcast aggregation
    # still merges both into the show's single recurring named host.
    g = _graph_from(
        _kg_episode(
            "podcast:unhedged",
            "episode:u1",
            host_id="person:katie-martin",
            host_name="Katie Martin",
        ),
        _kg_episode(
            "podcast:unhedged",
            "episode:u2",
            host_id="person:katie-martin",
            host_name="Katie Martin",
        ),
        _kg_episode(
            "podcast:unhedged", "episode:u3", host_id="person:speaker-u3-03", host_name="SPEAKER_03"
        ),
        _kg_episode(
            "podcast:unhedged", "episode:u4", host_id="person:speaker-u4-03", host_name="SPEAKER_03"
        ),
    )
    g._reconcile_feed_hosts()
    assert g.get_node("person:speaker-u3-03") is None  # both episode-scoped voices merged away
    assert g.get_node("person:speaker-u4-03") is None
    assert "episode:u3" in g.neighbors("person:katie-martin")
    assert "episode:u4" in g.neighbors("person:katie-martin")


def test_reconcile_tags_episode_scoped_recurring_unnamed_host_in_cohost_show():
    # Co-host show (two recurring named hosts) → episode-scoped unnamed voices aren't merged, but
    # their episodes union to >=2, so both get the honest "recurring host" note via aggregation.
    g = _graph_from(
        _kg_episode(
            "podcast:fx", "episode:f1", host_id="person:katie", host_name="Katie", title="FX Show"
        ),
        _kg_episode("podcast:fx", "episode:f2", host_id="person:katie", host_name="Katie"),
        _kg_episode("podcast:fx", "episode:f3", host_id="person:rob", host_name="Rob"),
        _kg_episode("podcast:fx", "episode:f4", host_id="person:rob", host_name="Rob"),
        _kg_episode(
            "podcast:fx", "episode:f5", host_id="person:speaker-f5-07", host_name="SPEAKER_07"
        ),
        _kg_episode(
            "podcast:fx", "episode:f6", host_id="person:speaker-f6-07", host_name="SPEAKER_07"
        ),
    )
    g._reconcile_feed_hosts()
    for vid in ("person:speaker-f5-07", "person:speaker-f6-07"):
        voice = g.get_node(vid)
        assert voice is not None  # co-host ambiguity → not merged
        note = voice.payload.get("recurring_host_note", "")
        assert note.startswith("recurring host of") and "not auto-named" in note


def test_reconcile_does_not_merge_same_label_across_shows_when_episode_scoped():
    # The core #1b guarantee: SPEAKER_00 in show A and show B are DISTINCT episode-scoped nodes,
    # so the diarization label number never collapses into one phantom cross-show person.
    g = _graph_from(
        _kg_episode(
            "podcast:a",
            "episode:a1",
            host_id="person:speaker-a1-00",
            host_name="SPEAKER_00",
            title="Show A",
        ),
        _kg_episode(
            "podcast:b",
            "episode:b1",
            host_id="person:speaker-b1-00",
            host_name="SPEAKER_00",
            title="Show B",
        ),
    )
    g._reconcile_feed_hosts()
    assert g.get_node("person:speaker-a1-00") is not None  # two separate voices, never merged
    assert g.get_node("person:speaker-b1-00") is not None


def test_reconcile_skips_voice_shared_across_feeds():
    # A bare person:speaker-00 node MENTIONS episodes in TWO different shows — it's not
    # feed-exclusive, so merging it into one show's host would wrongly absorb the other.
    g = _graph_from(
        _kg_episode("podcast:a", "episode:a1", host_id="person:amy", host_name="Amy"),
        _kg_episode("podcast:a", "episode:a2", host_id="person:amy", host_name="Amy"),
        _kg_episode("podcast:b", "episode:b1", host_id="person:ben", host_name="Ben"),
        _kg_episode("podcast:b", "episode:b2", host_id="person:ben", host_name="Ben"),
        _kg_episode("podcast:a", "episode:a3", host_id="person:speaker-00", host_name="SPEAKER_00"),
        _kg_episode("podcast:b", "episode:b3", host_id="person:speaker-00", host_name="SPEAKER_00"),
    )
    g._reconcile_feed_hosts()
    assert g.get_node("person:speaker-00") is not None  # ambiguous → left alone


def test_reconcile_requires_recurring_named_host():
    # The named host appears in only ONE episode → not "recurring", so no merge target.
    g = _graph_from(
        _kg_episode("podcast:solo", "episode:s1", host_id="person:dana", host_name="Dana"),
        _kg_episode(
            "podcast:solo", "episode:s2", host_id="person:speaker-02", host_name="SPEAKER_02"
        ),
    )
    g._reconcile_feed_hosts()
    assert g.get_node("person:speaker-02") is not None  # nothing trustworthy to merge into


def test_reconcile_never_touches_guests():
    # An unnamed GUEST voice is never reassigned a host's name.
    g = _graph_from(
        _kg_episode(
            "podcast:show", "episode:e1", host_id="person:host-name", host_name="Real Host"
        ),
        _kg_episode(
            "podcast:show", "episode:e2", host_id="person:host-name", host_name="Real Host"
        ),
        _kg_episode(
            "podcast:show",
            "episode:e3",
            host_id="person:speaker-09",
            host_name="SPEAKER_09",
            role="guest",
        ),
    )
    g._reconcile_feed_hosts()
    assert g.get_node("person:speaker-09") is not None  # guest left as-is
