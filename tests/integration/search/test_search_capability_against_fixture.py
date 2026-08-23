"""Tier-3 search capability over the synthetic app-validation corpus (RFC-095 corpus).

The search-layer parallel to ``tests/integration/connectivity/
test_relational_queries_against_fixture.py`` (the graph tier-3): that file asserts every
``relational_queries`` surface returns real data against a fixture; this one asserts every
*search* surface — the two-tier hybrid index, tier stamping, query classification,
chunk→insight lift stats, the result-set operators (cluster / consensus), and two-subject
compare — returns real data against ``tests/fixtures/app-validation-corpus/v3``.

It exercises the ``structured_corpus_search`` SSOT (shared by ``GET /api/search`` and the MCP
``search_corpus`` tool) directly — the capability layer, not the MCP wrappers (those are
covered end-to-end in ``tests/integration/test_mcp_pivot_chain_e2e.py``).

The two-tier LanceDB index is BUILT here at setup (offline, cached MiniLM) — not committed —
so there's no binary lance blob in git and no lance-format-version coupling. The module skips
cleanly where the embedding model isn't available (model-less unit CI); it runs fully in the
ML tier / locally.

If a test here fails, EITHER the synthetic corpus lost a search-relevant artifact (regenerate
via ``scripts/build_app_validation_corpus.py`` + ``make enrich``) OR a search capability
changed semantics — move the test in lockstep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.search import operators
from podcast_scraper.search.capability import doc_types_for_tier, structured_corpus_search
from podcast_scraper.search.compare import compare_subjects, SubjectRef
from podcast_scraper.search.router import QUERY_TYPES
from tests.integration.conftest import requires

pytestmark = pytest.mark.integration

_CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus" / "v3"

# Two queries that resolve against the synthetic corpus's committed topics/insights.
_Q = "risk management"
_Q2 = "systems thinking"


@pytest.fixture(scope="module")
def corpus(app_validation_search_index: Path) -> Path:
    """The synthetic corpus with its two-tier index built.

    The build itself lives in ``tests/integration/conftest.py`` behind a cross-process lock —
    this module and ``test_mcp_pivot_chain_e2e.py`` each had their own copy of it, which under
    xdist meant two workers could run ``index-two-tier`` into the same LanceDB directory, and a
    third module read the sidecar without ever declaring it needed one.
    """
    return app_validation_search_index


def _search(corpus: Path, query: str, **kw) -> dict:
    out: dict = structured_corpus_search(corpus, query, **kw)
    if out.get("error") == "embed_failed":
        pytest.skip("query embedding unavailable in this environment")
    return out


class TestStructuredSearchAgainstFixture:
    """Every search surface returns real data for a query that resolves in the corpus."""

    @requires("lancedb")  # searches the built LanceDB fixture index
    def test_two_tier_search_stamps_both_tiers(self, corpus: Path) -> None:
        """PRD-033 FR1.1: a ``both``-tier search returns hits from BOTH the insight tier and
        the segment tier, each stamped with its ``source_tier``."""
        out = _search(corpus, _Q, doc_types=doc_types_for_tier("both"), top_k=15)
        assert out.get("error") is None
        res = out["results"]
        assert res, "both-tier search returned no hits on the synthetic corpus"
        tiers = {r["source_tier"] for r in res}
        assert {"insight", "segment"} <= tiers, f"both-tier search missed a tier: {tiers}"
        # every hit carries a known tier stamp (insight / segment / aux kg-surface).
        assert tiers <= {"insight", "segment", "aux"}, f"unexpected tier stamp: {tiers}"

    @requires("lancedb")  # searches the built LanceDB fixture index
    def test_insight_tier_filter_returns_only_insights(self, corpus: Path) -> None:
        out = _search(corpus, _Q, doc_types=doc_types_for_tier("insight"), top_k=10)
        res = out["results"]
        assert res, "insight-tier search returned no hits"
        assert {r["source_tier"] for r in res} == {"insight"}

    @requires("lancedb")  # searches the built LanceDB fixture index
    def test_segment_tier_filter_returns_only_segments(self, corpus: Path) -> None:
        out = _search(corpus, _Q, doc_types=doc_types_for_tier("segment"), top_k=10)
        res = out["results"]
        assert res, "segment-tier search returned no hits"
        assert {r["source_tier"] for r in res} == {"segment"}

    def test_query_type_is_classified(self, corpus: Path) -> None:
        """FR1.4: the response is stamped with a detected query_type from the known set."""
        out = _search(corpus, _Q, top_k=5)
        assert out["query_type"] in QUERY_TYPES

    @requires("lancedb")  # searches the built LanceDB fixture index
    def test_lift_stats_are_wellformed(self, corpus: Path) -> None:
        """RFC-061: a two-tier search reports chunk→insight lift stats — present + well-typed
        even when no lift fires (the synthetic corpus doesn't share GIL transcript offsets, so
        we assert the SHAPE, not that lift > 0)."""
        out = _search(corpus, _Q, doc_types=doc_types_for_tier("both"), top_k=10)
        stats = out["lift_stats"]
        assert isinstance(stats, dict)
        assert isinstance(stats.get("transcript_hits_returned"), int)
        assert isinstance(stats.get("lift_applied"), int)
        assert stats["transcript_hits_returned"] >= 0 and stats["lift_applied"] >= 0

    @requires("lancedb")  # searches the built LanceDB fixture index
    def test_grounded_only_filter_runs_and_returns(self, corpus: Path) -> None:
        out = _search(corpus, _Q, grounded_only=True, top_k=10)
        assert out.get("error") is None
        assert out["results"], "grounded_only search returned no hits on the synthetic corpus"

    @requires("lancedb")  # searches the built LanceDB fixture index
    def test_topic_filter_runs_and_returns(self, corpus: Path) -> None:
        out = _search(corpus, _Q2, topic="topic:systems-thinking", top_k=15)
        assert out.get("error") is None
        assert out["results"], "topic-filtered search returned no hits"

    @requires("lancedb")  # searches the built LanceDB fixture index
    def test_speaker_filter_runs_and_returns(self, corpus: Path) -> None:
        out = _search(corpus, _Q, speaker="person:maya", top_k=15)
        assert out.get("error") is None
        assert out["results"], "speaker-filtered search returned no hits"


class TestSearchOperatorsAgainstFixture:
    """The result-set operators (cluster / consensus) fold real corpus enrichments over hits."""

    @requires("lancedb")  # searches the built LanceDB fixture index
    def test_cluster_hits_forms_a_real_theme_group(self, corpus: Path) -> None:
        out = _search(corpus, _Q, top_k=20)
        hits = out["results"]
        assert hits
        groups = operators.cluster_hits(hits, corpus)
        real = [g for g in groups if g.get("cluster_kind") != "ungrouped" and g.get("cluster_id")]
        assert real, f"cluster_hits produced no real theme group: {groups}"
        g = real[0]
        # indices point back into the caller's hit list; the label/size are consistent.
        assert g["hit_indices"] and all(0 <= i < len(hits) for i in g["hit_indices"])
        assert g["size"] == len(g["hit_indices"])
        assert g.get("label")

    def test_consensus_pairs_are_cross_person(self, corpus: Path) -> None:
        out = _search(corpus, _Q, top_k=20)
        pairs = operators.consensus_pairs_for_hits(out["results"], corpus, max_pairs=10)
        assert pairs, "consensus_pairs_for_hits returned no cross-person pair on the corpus"
        p = pairs[0]
        assert p["person_a_id"] != p["person_b_id"], "consensus pair is not cross-person"
        assert p.get("topic_id", "").startswith("topic:")
        assert p.get("insight_a_text") and p.get("insight_b_text")


@requires("lancedb")  # searches the built LanceDB fixture index
class TestCompareAgainstFixture:
    """Two-subject compare assembles a briefing pack per side + a judge summary."""

    def test_compare_two_topics_yields_packs_and_judge(self, corpus: Path) -> None:
        outcome = compare_subjects(
            corpus,
            SubjectRef(kind="topic", id="topic:risk-management"),
            SubjectRef(kind="topic", id="topic:systems-thinking"),
            q="",
            insight_types=["claim"],  # the RFC-072 filter narrows both sides symmetrically.
        )
        assert outcome.error is None, f"compare errored: {outcome.detail}"
        assert outcome.pack_a is not None and outcome.pack_b is not None
        assert outcome.pack_a.rendered and outcome.pack_a.top_insight_id
        assert outcome.pack_b.rendered
        # judge_summary is a rendered string (deterministic compare), not a dict.
        assert isinstance(outcome.judge_summary, str) and outcome.judge_summary
