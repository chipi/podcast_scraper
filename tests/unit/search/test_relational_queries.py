"""Unit tests for the relational-query layer (RFC-094 / #882).

These prove the queries traverse the **typed** edges correctly — in particular that a
person↔insight pair is not conflated between ``STATES`` (the person stated it) and
``MENTIONS`` (the insight mentions the person). A hand-built fake graph keeps the test
independent of artifact loading; ``test_corpus_graph.py`` covers ``typed_neighbors``
against the real ``CorpusGraph``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pytest

from podcast_scraper.search.corpus_graph import Node
from podcast_scraper.search.relational_queries import (
    co_speakers,
    cross_show_synthesis,
    entities_in,
    entities_in_topic,
    episode_related_insights,
    episodes_of,
    insights_about,
    node_label,
    positions_of,
    related_insights,
    related_topics,
    RelatedNode,
    rerank_by_relevance,
    shared_topics,
    topics_of,
    topics_of_insight,
    who_said,
)

pytestmark = pytest.mark.unit


def _rn(node_id: str) -> RelatedNode:
    return RelatedNode(id=node_id, type="insight", text=node_id)


def test_rerank_by_relevance_scored_first_then_original_order() -> None:
    items = [_rn("i1"), _rn("i2"), _rn("i3"), _rn("i4")]
    # i3 and i1 are scored (i3 higher); i2, i4 unscored keep their original order after.
    out = rerank_by_relevance(items, {"i1": 0.5, "i3": 0.9})
    assert [r.id for r in out] == ["i3", "i1", "i2", "i4"]


def test_rerank_empty_map_is_unchanged() -> None:
    items = [_rn("i1"), _rn("i2")]
    assert [r.id for r in rerank_by_relevance(items, {})] == ["i1", "i2"]


def test_rerank_is_stable_for_equal_scores() -> None:
    items = [_rn("a"), _rn("b"), _rn("c")]
    out = rerank_by_relevance(items, {"a": 1.0, "b": 1.0, "c": 1.0})
    assert [r.id for r in out] == ["a", "b", "c"]


def test_node_label_returns_display_text() -> None:
    nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
        "person:p": ("person", {"name": "Jane Doe"}),
        "topic:t": ("topic", {"label": "Inflation"}),
    }
    g = FakeGraph(nodes, [])
    assert node_label(g, "person:p") == "Jane Doe"
    assert node_label(g, "topic:t") == "Inflation"
    assert node_label(g, "missing") == ""


def test_unnamed_recurring_host_shows_note_not_raw_speaker_id() -> None:
    # #1056 Opt1: a host the roster couldn't name keeps a bare "SPEAKER_07" name but
    # carries a reconciliation note — the projection must surface the note as display
    # text (and structurally) so the agent/viewer never sees raw "SPEAKER_07".
    note = "recurring host of FX Show — not auto-named"
    nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
        "person:speaker-07": ("person", {"name": "SPEAKER_07", "recurring_host_note": note}),
        "person:named": ("person", {"name": "Katie Martin", "recurring_host_note": ""}),
    }
    g = FakeGraph(nodes, [])
    assert node_label(g, "person:speaker-07") == note  # not "SPEAKER_07"
    assert node_label(g, "person:named") == "Katie Martin"  # real names untouched


class FakeGraph:
    """Minimal ``GraphLike``: typed undirected edges over hand-declared nodes."""

    def __init__(
        self,
        nodes: Dict[str, Tuple[str, Dict[str, object]]],
        edges: List[Tuple[str, str, str]],
    ) -> None:
        self._nodes = {nid: Node(id=nid, type=t, payload=p) for nid, (t, p) in nodes.items()}
        self._typed: Dict[str, List[Tuple[str, str]]] = {}
        for a, b, etype in edges:
            self._typed.setdefault(a, []).append((b, etype))
            self._typed.setdefault(b, []).append((a, etype))  # undirected

    def get_node(self, node_id: Optional[str]) -> Optional[Node]:
        return self._nodes.get(node_id) if node_id else None

    def typed_neighbors(self, node_id: str, edge_type: str) -> List[str]:
        return sorted({nbr for nbr, et in self._typed.get(node_id, ()) if et == edge_type})


@pytest.fixture()
def graph() -> FakeGraph:
    """Two shows, two speakers, one shared topic; insight:1 also *mentions* org:acme.

    insight:1 — stated by alice, about ai, mentions acme, in episode e1 (show1).
    insight:2 — stated by bob,   about ai,                in episode e2 (show2).
    """
    nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
        "person:alice": ("person", {"name": "Alice"}),
        "person:bob": ("person", {"name": "Bob"}),
        "org:acme": ("org", {"name": "Acme"}),
        "topic:ai": ("topic", {"label": "AI"}),
        "insight:1": ("insight", {"text": "Alice's position on AI"}),
        "insight:2": ("insight", {"text": "Bob's position on AI"}),
        "episode:e1": ("episode", {"show_id": "show1"}),
        "episode:e2": ("episode", {"show_id": "show2"}),
        "podcast:show1": ("podcast", {"name": "Show One"}),
        "podcast:show2": ("podcast", {"name": "Show Two"}),
    }
    edges: List[Tuple[str, str, str]] = [
        ("person:alice", "insight:1", "STATES"),
        ("person:bob", "insight:2", "STATES"),
        ("insight:1", "topic:ai", "ABOUT"),
        ("insight:2", "topic:ai", "ABOUT"),
        ("insight:1", "org:acme", "MENTIONS"),
        ("episode:e1", "insight:1", "HAS_INSIGHT"),
        ("episode:e2", "insight:2", "HAS_INSIGHT"),
        ("podcast:show1", "episode:e1", "HAS_EPISODE"),
        ("podcast:show2", "episode:e2", "HAS_EPISODE"),
    ]
    return FakeGraph(nodes, edges)


def test_positions_of_returns_only_stated_insights(graph: FakeGraph) -> None:
    result = positions_of(graph, "person:alice")
    assert [r.id for r in result] == ["insight:1"]
    # Alice did not state insight:2 — STATES must not leak via the shared topic.
    assert "insight:2" not in [r.id for r in result]


def test_insights_about_uses_mentions_edge(graph: FakeGraph) -> None:
    result = insights_about(graph, "org:acme")
    assert [r.id for r in result] == ["insight:1"]


def test_entities_in_returns_mentioned_not_speaker(graph: FakeGraph) -> None:
    # The disambiguation that motivated typed edges: insight:1's speaker is alice
    # (STATES), but its *mentioned* entity is acme (MENTIONS). entities_in must
    # return only the mentioned entity, never the speaker.
    result = entities_in(graph, "insight:1")
    assert [r.id for r in result] == ["org:acme"]
    assert "person:alice" not in [r.id for r in result]


def test_episodes_of_walks_has_episode(graph: FakeGraph) -> None:
    assert [r.id for r in episodes_of(graph, "podcast:show1")] == ["episode:e1"]


def test_who_said_buckets_insights_by_speaker(graph: FakeGraph) -> None:
    result = who_said(graph, "topic:ai")
    assert set(result) == {"person:alice", "person:bob"}
    assert [r.id for r in result["person:alice"]] == ["insight:1"]
    assert [r.id for r in result["person:bob"]] == ["insight:2"]


def test_cross_show_synthesis_one_insight_per_distinct_show(graph: FakeGraph) -> None:
    # Show key is the canonical podcast node id (consistent with episodes_of).
    result = cross_show_synthesis(graph, "topic:ai")
    assert set(result) == {"podcast:show1", "podcast:show2"}
    assert [r.id for r in result["podcast:show1"]] == ["insight:1"]
    assert [r.id for r in result["podcast:show2"]] == ["insight:2"]


def test_show_resolved_via_episode_when_payload_lacks_show_id(graph: FakeGraph) -> None:
    # insight nodes carry no show_id; the show is resolved insight→episode→podcast.
    result = cross_show_synthesis(graph, "topic:ai")
    assert result["podcast:show1"][0].id == "insight:1"


def test_related_insights_via_shared_topic_and_entity(graph: FakeGraph) -> None:
    # insight:1 and insight:2 share topic:ai → siblings; the seed is excluded.
    result = related_insights(graph, "insight:1")
    assert [r.id for r in result] == ["insight:2"]
    assert "insight:1" not in [r.id for r in result]


def test_related_insights_dedupes_across_shared_topic_and_entity() -> None:
    # insight:a and insight:b share BOTH a topic and an entity — returned once.
    nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
        "insight:a": ("insight", {}),
        "insight:b": ("insight", {}),
        "topic:t": ("topic", {}),
        "org:o": ("org", {}),
    }
    edges = [
        ("insight:a", "topic:t", "ABOUT"),
        ("insight:b", "topic:t", "ABOUT"),
        ("insight:a", "org:o", "MENTIONS"),
        ("insight:b", "org:o", "MENTIONS"),
    ]
    g = FakeGraph(nodes, edges)
    assert [r.id for r in related_insights(g, "insight:a")] == ["insight:b"]


def test_entities_in_topic_ranks_by_mention_frequency() -> None:
    # topic:t insights: i1 mentions org:acme + person:p; i2 mentions org:acme.
    # org:acme is mentioned by 2 insights, person:p by 1 → org:acme ranks first.
    nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
        "topic:t": ("topic", {}),
        "insight:1": ("insight", {}),
        "insight:2": ("insight", {}),
        "org:acme": ("org", {"name": "Acme"}),
        "person:p": ("person", {"name": "P"}),
    }
    edges = [
        ("topic:t", "insight:1", "ABOUT"),
        ("topic:t", "insight:2", "ABOUT"),
        ("insight:1", "org:acme", "MENTIONS"),
        ("insight:1", "person:p", "MENTIONS"),
        ("insight:2", "org:acme", "MENTIONS"),
    ]
    g = FakeGraph(nodes, edges)
    assert [r.id for r in entities_in_topic(g, "topic:t")] == ["org:acme", "person:p"]


def test_entities_in_topic_empty_on_missing(graph: FakeGraph) -> None:
    assert entities_in_topic(graph, "topic:nope") == []


def test_episode_related_insights_excludes_own_insights(graph: FakeGraph) -> None:
    # episode:e1 owns insight:1 (via HAS_INSIGHT). insight:1 shares topic:ai with
    # insight:2 → insight:2 is related; insight:1 (own) is excluded.
    result = episode_related_insights(graph, "episode:e1")
    assert [r.id for r in result] == ["insight:2"]
    assert "insight:1" not in [r.id for r in result]


def test_episode_related_insights_accepts_bare_id(graph: FakeGraph) -> None:
    assert [r.id for r in episode_related_insights(graph, "e1")] == ["insight:2"]


def test_queries_are_total_on_missing_ids(graph: FakeGraph) -> None:
    assert positions_of(graph, "person:nobody") == []
    assert insights_about(graph, "org:nobody") == []
    assert entities_in(graph, "insight:nope") == []
    assert episodes_of(graph, "podcast:nope") == []
    assert related_insights(graph, "insight:nope") == []
    assert episode_related_insights(graph, "episode:nope") == []
    assert who_said(graph, "topic:nope") == {}
    assert cross_show_synthesis(graph, "topic:nope") == {}


def test_k_limit_caps_positions(graph: FakeGraph) -> None:
    nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
        "person:p": ("person", {}),
        **{f"insight:{i}": ("insight", {}) for i in range(5)},
    }
    edges = [("person:p", f"insight:{i}", "STATES") for i in range(5)]
    g = FakeGraph(nodes, edges)
    assert len(positions_of(g, "person:p", k=3)) == 3


# --- connectivity helpers (#1054) ---


def test_topics_of_walks_person_to_topics(graph: FakeGraph) -> None:
    # alice STATES insight:1 ABOUT topic:ai -> her topics include ai (closes the dead-end).
    assert [t.id for t in topics_of(graph, "person:alice")] == ["topic:ai"]


def test_topics_of_empty_for_unknown_person(graph: FakeGraph) -> None:
    assert topics_of(graph, "person:nobody") == []


def test_co_speakers_finds_people_on_shared_topics(graph: FakeGraph) -> None:
    # alice + bob both speak ABOUT topic:ai -> each is the other's co-speaker; self excluded.
    assert [p.id for p in co_speakers(graph, "person:alice")] == ["person:bob"]
    assert "person:alice" not in [p.id for p in co_speakers(graph, "person:alice")]


def test_topics_of_insight(graph: FakeGraph) -> None:
    assert [t.id for t in topics_of_insight(graph, "insight:1")] == ["topic:ai"]


def test_shared_topics_is_the_intersection(graph: FakeGraph) -> None:
    # alice + bob both engage topic:ai → it's their shared topic.
    assert [t.id for t in shared_topics(graph, "person:alice", "person:bob")] == ["topic:ai"]


def test_related_topics_via_shared_insight() -> None:
    # one insight is ABOUT both topic:a and topic:b → they're related.
    nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
        "topic:a": ("topic", {"label": "A"}),
        "topic:b": ("topic", {"label": "B"}),
        "topic:c": ("topic", {"label": "C"}),
        "insight:1": ("insight", {}),
        "insight:2": ("insight", {}),
    }
    edges = [
        ("insight:1", "topic:a", "ABOUT"),
        ("insight:1", "topic:b", "ABOUT"),
        ("insight:2", "topic:a", "ABOUT"),
        ("insight:2", "topic:c", "ABOUT"),
    ]
    g = FakeGraph(nodes, edges)
    # a co-occurs with b (insight:1) and c (insight:2); self excluded.
    assert sorted(t.id for t in related_topics(g, "topic:a")) == ["topic:b", "topic:c"]
    assert "topic:a" not in [t.id for t in related_topics(g, "topic:a")]



# ---------------------------------------------------------------------------
# RFC-097 v3.0 typed MENTIONS family — queries traverse all three variants.
# ---------------------------------------------------------------------------


class TestTypedMentionsFamilyQueries:
    """Queries treat ``MENTIONS`` + ``MENTIONS_PERSON`` + ``MENTIONS_ORG`` as
    one semantic family (RFC-097 v3.0). New corpora emit only the typed
    variants; legacy / mid-migration corpora may still carry the generic
    ``MENTIONS``. The query layer must surface results from all three.
    """

    def _typed_graph(self) -> FakeGraph:
        nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
            "person:alice": ("person", {"name": "Alice"}),
            "person:bob": ("person", {"name": "Bob"}),
            "org:acme": ("org", {"name": "Acme"}),
            "topic:ai": ("topic", {"label": "AI"}),
            "insight:1": ("insight", {"text": "i1"}),
            "insight:2": ("insight", {"text": "i2"}),
            "insight:3": ("insight", {"text": "i3"}),
        }
        # i1 mentions alice (typed PERSON) + acme (typed ORG)
        # i2 mentions bob (typed PERSON)
        # i3 mentions alice (LEGACY generic — mid-migration corpus)
        edges: List[Tuple[str, str, str]] = [
            ("insight:1", "person:alice", "MENTIONS_PERSON"),
            ("insight:1", "org:acme", "MENTIONS_ORG"),
            ("insight:2", "person:bob", "MENTIONS_PERSON"),
            ("insight:3", "person:alice", "MENTIONS"),  # legacy form
            ("insight:1", "topic:ai", "ABOUT"),
            ("insight:2", "topic:ai", "ABOUT"),
            ("insight:3", "topic:ai", "ABOUT"),
        ]
        return FakeGraph(nodes, edges)

    def test_insights_about_traverses_typed_person_edges(self) -> None:
        """insights_about(person) returns insights via MENTIONS_PERSON."""
        from podcast_scraper.search.relational_queries import insights_about

        g = self._typed_graph()
        result = insights_about(g, "person:alice")
        result_ids = {r.id for r in result}
        # both insight:1 (MENTIONS_PERSON) and insight:3 (legacy MENTIONS) match
        assert result_ids == {"insight:1", "insight:3"}

    def test_insights_about_traverses_typed_org_edges(self) -> None:
        """insights_about(org) returns insights via MENTIONS_ORG."""
        from podcast_scraper.search.relational_queries import insights_about

        g = self._typed_graph()
        assert [r.id for r in insights_about(g, "org:acme")] == ["insight:1"]

    def test_entities_in_returns_both_typed_variants(self) -> None:
        """entities_in(insight) returns mentioned entities regardless of typed variant."""
        from podcast_scraper.search.relational_queries import entities_in

        g = self._typed_graph()
        result = entities_in(g, "insight:1")
        result_ids = {r.id for r in result}
        assert result_ids == {"person:alice", "org:acme"}

    def test_entities_in_topic_ranks_across_typed_family(self) -> None:
        """entities_in_topic counts mentions across the full MENTIONS family.

        Alice is mentioned twice (one typed PERSON via insight:1, one legacy via
        insight:3); Acme + Bob once each. Most-mentioned first.
        """
        from podcast_scraper.search.relational_queries import entities_in_topic

        g = self._typed_graph()
        result = entities_in_topic(g, "topic:ai")
        assert [r.id for r in result] == [
            "person:alice",  # 2 mentions (one typed + one legacy)
            "org:acme",  # 1
            "person:bob",  # 1
        ]

    def test_typed_family_dedupes_same_neighbor_in_multiple_variants(self) -> None:
        """Defensive: if a graph somehow lists the same edge under both legacy
        ``MENTIONS`` AND typed ``MENTIONS_PERSON`` (e.g. mid-migration data),
        the result still includes the neighbor exactly once.
        """
        from podcast_scraper.search.relational_queries import entities_in

        nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
            "person:alice": ("person", {"name": "Alice"}),
            "insight:1": ("insight", {"text": "i1"}),
        }
        edges: List[Tuple[str, str, str]] = [
            ("insight:1", "person:alice", "MENTIONS"),
            ("insight:1", "person:alice", "MENTIONS_PERSON"),
        ]
        g = FakeGraph(nodes, edges)
        result = entities_in(g, "insight:1")
        assert [r.id for r in result] == ["person:alice"]
        assert len(result) == 1  # not duplicated

    def test_related_insights_walks_typed_mentions(self) -> None:
        """related_insights walks shared-entity siblings via typed MENTIONS too."""
        from podcast_scraper.search.relational_queries import related_insights

        # i1 mentions alice (typed); i3 mentions alice (legacy).
        # related_insights(i1) should surface i3 as a typed-vs-legacy sibling.
        g = self._typed_graph()
        result = related_insights(g, "insight:1")
        result_ids = {r.id for r in result}
        # Both i2 (shared ABOUT topic) and i3 (shared alice mention) are siblings
        assert "insight:3" in result_ids
