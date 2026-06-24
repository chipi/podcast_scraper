"""Unit tests for the ``corpus_briefing_pack`` MCP tool (RFC-093).

The pack-builder itself is exercised by
``tests/unit/search/test_context_pack.py``; these tests only assert
the MCP wrapper does the right things on top of it (empty query
short-circuit, dict→ScoredResult adapter, error pass-through, output
shape).
"""

from __future__ import annotations

import pytest

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools.briefing_pack import (
    _dict_to_scored_result,
    corpus_briefing_pack,
)

pytestmark = pytest.mark.unit


def test_empty_query_short_circuits(tmp_path) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    out = corpus_briefing_pack(ctx, "   ")
    assert out["error"] == "empty_query"
    assert out["rendered"] == ""


def test_search_error_passes_through(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)

    def fake_struct(root, query, **kwargs):
        return {
            "query_type": "semantic",
            "results": [],
            "error": "no_index",
            "detail": "no index found",
            "lift_stats": None,
        }

    monkeypatch.setattr(
        "podcast_scraper.search.capability.structured_corpus_search",
        fake_struct,
    )
    out = corpus_briefing_pack(ctx, "climate")
    assert out["error"] == "no_index"
    assert out["detail"] == "no index found"
    assert out["rendered"] == ""


def test_dict_to_scored_result_adapter() -> None:
    """The wrapper converts the search-tool dict shape back to the
    ScoredResult dataclass the pack builder expects."""
    row = {
        "doc_id": "insight:abc",
        "score": 0.87,
        "rank": 3,
        "text": "Some grounded claim.",
        "source_tier": "insight",
        "metadata": {
            "episode_id": "ep1",
            "show_id": "show-a",
            "confidence": 0.9,
        },
        "signal": "rrf",
        "supporting_quotes": ["q1"],
        "lifted": True,
    }
    out = _dict_to_scored_result(row)
    assert out.doc_id == "insight:abc"
    assert out.score == pytest.approx(0.87)
    assert out.rank == 3
    assert out.source_tier == "insight"
    assert out.payload["text"] == "Some grounded claim."
    assert out.payload["episode_id"] == "ep1"
    assert out.payload["show_id"] == "show-a"
    assert out.payload["supporting_quotes"] == ["q1"]
    assert out.payload["lifted"] is True


def test_end_to_end_returns_litm_pack(tmp_path, monkeypatch) -> None:
    """Full happy path: stub search returns two results, the wrapper
    converts + builds the pack + returns the expected envelope."""
    ctx = CorpusContext.from_path(tmp_path)

    def fake_struct(root, query, **kwargs):
        return {
            "query_type": "entity_lookup",
            "results": [
                {
                    "doc_id": "insight:i1",
                    "score": 0.92,
                    "rank": 1,
                    "text": "Grounded claim about AI safety.",
                    "source_tier": "insight",
                    "signal": "rrf",
                    "metadata": {
                        "episode_id": "ep1",
                        "show_id": "show-a",
                        "confidence": 0.92,
                    },
                },
                {
                    "doc_id": "segment:s1",
                    "score": 0.71,
                    "rank": 2,
                    "text": "Raw transcript quote supporting the claim.",
                    "source_tier": "segment",
                    "signal": "bm25",
                    "metadata": {"episode_id": "ep1", "show_id": "show-a"},
                },
            ],
            "error": None,
            "detail": None,
            "lift_stats": None,
        }

    monkeypatch.setattr(
        "podcast_scraper.search.capability.structured_corpus_search",
        fake_struct,
    )

    out = corpus_briefing_pack(ctx, "ai safety", top_k=5, max_tokens=2000)

    assert out["query"] == "ai safety"
    assert out["query_type"] == "entity_lookup"
    assert "error" not in out  # success path: no error key
    assert "rendered" in out and out["rendered"]
    # LITM section order in the rendered text.
    rendered = out["rendered"]
    assert (
        rendered.index("[CRITICAL GROUNDING]")
        < rendered.index("[SUPPORTING EVIDENCE]")
        < rendered.index("[CAVEATS]")
    )
    assert out["top_insight_id"] == "insight:i1"
    assert out["supporting_segment_ids"] == ["segment:s1"]
    assert out["coverage_summary"]["episode_count"] == 1
    assert out["result_count"] == 2
    assert out["max_tokens"] == 2000


def test_top_k_clamped(tmp_path, monkeypatch) -> None:
    ctx = CorpusContext.from_path(tmp_path)
    captured: dict = {}

    def fake_struct(root, query, **kwargs):
        captured.update(kwargs)
        return {"query_type": "semantic", "results": [], "error": None}

    monkeypatch.setattr(
        "podcast_scraper.search.capability.structured_corpus_search",
        fake_struct,
    )
    corpus_briefing_pack(ctx, "climate", top_k=999)
    assert captured["top_k"] == 100  # clamped to [1, 100] like search_corpus
