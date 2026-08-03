"""Behavioural smoke for the cross-surface refresh tools (RFC-095, 2026-08).

Exercises the new tool functions against a committed fixture corpus so they have CI
regression coverage (the full prod-v2 pivot-chain e2e is local-only —
scripts/mcp_e2e_pivot_chain.py). Asserts shape + that ids/handles are present, not exact
values (fixture-dependent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools import cil, composites, connectivity, enrichment, gi, trending

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "connectivity-multi-show"


@pytest.fixture
def ctx() -> CorpusContext:
    return CorpusContext.from_path(_FIXTURE)


def test_corpus_trending_returns_kinds(ctx: CorpusContext) -> None:
    out = trending.corpus_trending(ctx, limit=3)
    assert out["error"] is None
    assert "topic" in out["kinds"]  # every kind key present even when empty


def test_perspective_leaders_shape(ctx: CorpusContext) -> None:
    out = cil.topic_perspective_leaders(ctx, limit=3)
    assert isinstance(out["leaders"], list)


def test_topic_clusters_envelope(ctx: CorpusContext) -> None:
    # any topic id or none: the tool must always return the uniform ok-envelope
    out = connectivity.topic_clusters(ctx, "topic:does-not-exist")
    assert out["ok"] is True
    assert set(out["data"]) == {"semantic", "theme"}


def test_ego_network_unknown_entity_is_clean_error(ctx: CorpusContext) -> None:
    out = connectivity.ego_network(ctx, "person:nobody", max_hops=2, k=5)
    assert out["ok"] is False  # not-in-corpus is a clean error, not a crash


def test_explore_insights_shape(ctx: CorpusContext) -> None:
    out = gi.explore_insights(ctx, limit=3)
    assert "insights" in out


def test_insight_detail_non_insight_returns_none(ctx: CorpusContext) -> None:
    out = cil  # noqa: F841 — keep import used if the assert below changes
    from podcast_scraper.mcp.tools import relational

    res = relational.insight_detail(ctx, "topic:not-an-insight")
    assert res["detail"] is None


def test_corpus_enrichment_signals_envelope(ctx: CorpusContext) -> None:
    out = enrichment.corpus_enrichment_signals(ctx)
    assert out["scope"] == "corpus"
    assert isinstance(out["signals"], dict)  # empty on a fixture without enrichments/


def test_entity_dossier_topic_branch(ctx: CorpusContext) -> None:
    out = composites.entity_dossier(ctx, "topic:whatever", k=3)
    # kind-dispatched: topic branch carries the topic keys even when empty
    assert out["entity_id"] == "topic:whatever"
    assert out["kind"] == "topic"
