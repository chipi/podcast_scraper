"""#1058 chunk 5 — Tier-3 connectivity over the multi-show fixture.

Loads the chunk-4 multi-show fixture via the real ``CorpusGraph`` and
asserts every relational connectivity surface returns non-empty data
for at least one valid input. Together with the per-artifact
fixture-contract tests in ``test_multi_show_fixture.py``, this proves
the live server path (CorpusGraph → relational_queries) renders every
surface the viewer + relational query layer expects under #1058.

This is the "real-server" half of #1058's closure: chunks 1–3 build
the pipeline path so airgapped corpora carry the data; chunk 4 builds
the pre-seeded data layer; this chunk asserts the same data flows
through the real query code paths the API + viewer consume.

If a test here fails, EITHER:

1. The fixture lost a connectivity surface — regenerate via
   ``build_fixture.py`` after updating the contract.
2. A relational_queries function changed signature / semantics — move
   the test in lockstep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.search import relational_queries as rq
from podcast_scraper.search.corpus_graph import CorpusGraph

pytestmark = pytest.mark.integration

_FIXTURE_FEEDS = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "connectivity-multi-show" / "feeds"
)


@pytest.fixture(scope="module")
def graph() -> CorpusGraph:
    if not _FIXTURE_FEEDS.is_dir():
        pytest.skip(
            "multi-show fixture missing — regenerate via "
            "tests/fixtures/connectivity-multi-show/build_fixture.py"
        )
    return CorpusGraph.build(
        _FIXTURE_FEEDS.parent,
        derive_speaker_links=True,
        reconcile_hosts=True,
    )


class TestRelationalQueriesNonEmpty:
    """Every connectivity surface returns non-empty data for at
    least one valid input from the fixture."""

    def test_positions_of_for_a_cross_show_host(self, graph: CorpusGraph) -> None:
        # Alice Hayes hosts show-a + show-c, guests on show-b → positions
        # land across multiple shows.
        results = rq.positions_of(graph, "person:alice-hayes", k=20)
        assert results, "positions_of returned empty for the cross-show host"

    def test_topics_of_for_a_cross_show_host(self, graph: CorpusGraph) -> None:
        topics = rq.topics_of(graph, "person:alice-hayes", k=20)
        assert topics, "topics_of returned empty for the cross-show host"

    def test_co_speakers_intra_episode(self, graph: CorpusGraph) -> None:
        co = rq.co_speakers(graph, "person:alice-hayes", k=20)
        # Alice co-speaks with at least one of {Bob, Maya, Dan} across the
        # fixture's six episodes.
        assert co, "co_speakers returned empty for the cross-show host"
        co_ids = {n.id for n in co}
        assert co_ids & {
            "person:bob-chen",
            "person:maya-okonkwo",
            "person:dan-tran",
        }, f"co_speakers returned no known co-host: {co_ids}"

    def test_related_topics_via_concept_cluster(self, graph: CorpusGraph) -> None:
        """Across shows, two per-show Topics that point at the same
        concept-Topic become each other's related_topics via the
        shared concept. Exercises the RELATED_TO chunk-3 edges
        materialised into the fixture."""
        # show-a's "AI safety" Topic should surface show-b's "AI
        # alignment" / show-c's "alignment problem" as related.
        related = rq.related_topics(graph, "topic:show-a-ai-safety", k=20)
        related_ids = {n.id for n in related}
        # Either of the cross-show member Topics is acceptable evidence.
        assert related_ids & {
            "topic:show-b-ai-alignment",
            "topic:show-c-alignment-problem",
            "concept:topic-ai-safety",
        }, f"related_topics missed cross-show neighbours: {related_ids}"

    def test_cross_show_synthesis_via_concept(self, graph: CorpusGraph) -> None:
        synthesis = rq.cross_show_synthesis(graph, "concept:topic-ai-safety", per_show=2)
        # The concept-Topic should fold insights from ≥2 shows.
        assert synthesis, "cross_show_synthesis returned empty for the concept-Topic"
        assert len(synthesis) >= 2, f"cross_show_synthesis only saw one show: {list(synthesis)}"

    def test_who_said_for_a_topic(self, graph: CorpusGraph) -> None:
        speakers = rq.who_said(graph, "topic:show-a-ai-safety", k=20)
        assert speakers, "who_said returned empty for a per-show Topic"

    def test_insights_about_an_organization(self, graph: CorpusGraph) -> None:
        insights = rq.insights_about(graph, "org:acme-labs", k=20)
        assert insights, "insights_about returned empty for an Organization"


class TestEntityNeighborhood:
    def test_person_to_topics_path_lands(self, graph: CorpusGraph) -> None:
        """``Person → STATES → Insight → ABOUT → Topic`` traversal
        lands at least one Topic — the path the Person Profile and
        Position Tracker render. This is the same end-to-end
        connectivity contract Tier-2 asserts on the production-shaped
        fixture; here we assert it again on the multi-show fixture so
        the cross-show case is covered too."""
        topics = rq.topics_of(graph, "person:alice-hayes", k=20)
        topic_ids = {n.id for n in topics}
        # Alice's neighborhood across show-a + show-c should land
        # multiple distinct topic ids.
        assert len(topic_ids) >= 2, f"Alice's neighborhood is too thin: {topic_ids}"
