"""E2E: the MCP cross-surface pivot chain over the committed synthetic corpus (RFC-095).

Standardizes the pivot-chain e2e on ``tests/fixtures/app-validation-corpus/v3`` (the UI
tier-3 corpus). The two-tier search index is BUILT here at setup (offline, cached MiniLM) —
not committed — so there is no binary lance blob in git and no lance-format-version coupling.
Skips cleanly where the embedding model isn't available (model-less unit CI); runs fully in
the ML tier / locally.

Proves referential parity: each surface's output id feeds the next tool's input, so an agent
pivots search -> graph -> insight across the surfaces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.conftest import requires

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def indexed_corpus(app_validation_search_index: Path) -> Path:
    """The synthetic corpus with its two-tier index built.

    Skips the module when the index can't be built (embedding model missing/offline) — the
    non-search MCP tools are covered by tests/unit/mcp; this module is the search+pivot half.
    The build is the shared, cross-process-locked one in ``tests/integration/conftest.py``: this
    module and ``search/test_search_capability_against_fixture.py`` used to hold two independent
    copies of it, so under xdist both could write the same LanceDB directory at once.
    """
    return app_validation_search_index


@requires("lancedb")  # the pivot chain runs corpus search
def test_pivot_chain_ids_flow_across_surfaces(indexed_corpus: Path) -> None:
    from podcast_scraper.mcp.context import CorpusContext
    from podcast_scraper.mcp.tools import (
        cil,
        composites,
        connectivity,
        relational,
        search as search_tool,
    )

    ctx = CorpusContext.from_path(indexed_corpus)

    # centrality -> a topic id
    leaders = cil.topic_perspective_leaders(ctx, limit=3).get("leaders") or []
    assert leaders, "expected at least one perspective-leader topic in the synthetic corpus"
    topic_id = leaders[0].get("topic_id") or leaders[0].get("id")
    assert topic_id and topic_id.startswith("topic:")

    # temporal pivot: topic_id -> conversation arc
    arc = cil.topic_conversation_arc(ctx, topic_id).get("arc")
    assert arc is not None

    # search -> a hit that carries a pivot handle (the referential-parity artifact)
    res = search_tool.search_corpus(ctx, "risk management", tier="insight", top_k=5)
    if res.get("error") == "embed_failed":
        pytest.skip("query embedding unavailable in this environment")
    hits = res.get("results") or []
    assert hits, "expected search hits on the indexed synthetic corpus"
    pivot = hits[0].get("pivot") or {}
    assert pivot.get("id") and pivot.get("expand_with"), "every hit must carry a pivot handle"

    # THE BRIDGE: a search insight hit's pivot.id resolves in the graph (search -> graph)
    if pivot.get("kind") == "insight":
        detail = relational.insight_detail(ctx, pivot["id"]).get("detail")
        assert detail is not None and detail.get("id") == pivot["id"]

    # graph pivot: the topic id expands via multi-hop proximity
    ego = connectivity.ego_network(ctx, topic_id, max_hops=2, k=10)
    assert ego.get("ok") is True

    # composite: one call fuses the surfaces, keeping ids
    dossier = composites.entity_dossier(ctx, topic_id, k=5)
    assert dossier["kind"] == "topic"
    assert dossier.get("neighborhood") is not None


def test_episode_scoped_tools_return_data(indexed_corpus: Path) -> None:
    """The episode-scoped tools that were untestable until the synthetic corpus carried
    diarization diagnostics + current-schema GI/enrichments (the gap that drove the corpus
    realignment): speaker roster, per-episode insights, enrichment signals, episode digest."""
    from podcast_scraper.mcp.context import CorpusContext
    from podcast_scraper.mcp.tools import composites, enrichment, gi

    ctx = CorpusContext.from_path(indexed_corpus)
    metas = sorted(indexed_corpus.glob("feeds/*/*/metadata/*.metadata.json"))
    assert metas, "no episodes in the fixture"
    rel = str(metas[0].relative_to(indexed_corpus))

    # episode_speaker_roster — reads .speakers.diagnostics.json (net-new; no HTTP route).
    roster = enrichment.episode_speaker_roster(ctx, rel)
    diag = roster.get("diagnostics")
    assert diag is not None, "expected diarization diagnostics on the realigned corpus"
    summary = diag.get("summary", {})
    assert summary.get("num_speakers", 0) >= 1
    assert isinstance(summary.get("exposed"), list) and summary["exposed"]
    # talk-share is a fraction; the roster carries host/guest roles.
    assert any(e.get("role") in {"host", "guest"} for e in summary["exposed"])

    # episode_insights — salience-ranked grounded insights (current-schema GI).
    insights = gi.episode_insights(ctx, rel).get("insights")
    assert isinstance(insights, list) and insights

    # episode_enrichment_signals — per-episode RFC-088 envelopes.
    signals = enrichment.episode_enrichment_signals(ctx, rel).get("signals")
    assert isinstance(signals, dict) and signals

    # episode_digest — the composite that fuses all of the above in one call.
    digest = composites.episode_digest(ctx, rel)
    assert digest.get("insights") and digest.get("speaker_roster") is not None


@requires("lancedb")  # the pivot chain runs corpus search
def test_search_operators_and_compare(indexed_corpus: Path) -> None:
    """Search result-set operators + two-subject compare — the search-heavy tools that need
    a real index (built at setup) to exercise cluster_hits / consensus / compare_subjects."""
    from podcast_scraper.mcp.context import CorpusContext
    from podcast_scraper.mcp.tools import gi, operators

    ctx = CorpusContext.from_path(indexed_corpus)

    # cluster_search: run the search then group the hits by cluster.
    cl = operators.cluster_search(ctx, "risk management", top_k=15)
    if cl.get("error") == "embed_failed":
        pytest.skip("query embedding unavailable in this environment")
    assert cl.get("error") is None
    assert cl.get("hit_count", 0) >= 1
    assert isinstance(cl.get("groups"), list)

    # consensus_search: cross-speaker consensus pairs over the surfaced topics (may be empty).
    co = operators.consensus_search(ctx, "risk management", top_k=15)
    assert co.get("error") is None
    assert isinstance(co.get("consensus_pairs"), list)

    # compare_subjects: two topics -> a briefing pack per side + judge summary; insight_types
    # narrows both sides symmetrically (the only insight_type filter in MCP).
    cmp = gi.compare_subjects(
        ctx, "topic:risk-management", "topic:systems-thinking", insight_types=["claim"]
    )
    assert "error" not in cmp or cmp.get("error") is None
    assert cmp.get("pack_a") and cmp.get("pack_b")
