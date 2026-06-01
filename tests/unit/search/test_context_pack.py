"""Unit tests for LITM briefing packs (RFC-093, #861)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.backend import CompoundResult, ScoredResult
from podcast_scraper.search.context_pack import build_briefing_pack

pytestmark = pytest.mark.unit


def _ins(doc_id, text="", show="A", ep="e1", conf=None):
    payload = {"text": text, "show_id": show, "episode_id": ep}
    if conf is not None:
        payload["confidence"] = conf
    return ScoredResult(doc_id, 0.9, 1, payload, "vector", "insight")


def _seg(doc_id, text="", show="A", ep="e1"):
    return ScoredResult(
        doc_id, 0.8, 1, {"text": text, "show_id": show, "episode_id": ep}, "bm25", "segment"
    )


def test_top_insight_and_supporting_segments():
    results = [
        _ins("i1", "grounded claim", conf=0.9),
        _seg("s1", "raw quote one"),
        _seg("s2", "raw quote two"),
    ]
    pack = build_briefing_pack("q", "entity_lookup", results)
    assert pack.top_insight.doc_id == "i1"
    assert [s.doc_id for s in pack.supporting_segments] == ["s1", "s2"]
    assert pack.coverage_summary["episode_count"] == 1
    assert pack.coverage_summary["show_ids"] == ["A"]


def test_compound_contributes_both_tiers():
    seg = _seg("s1", "quote")
    ins = _ins("i1", "claim", conf=0.7)
    comp = CompoundResult(doc_id="s1", score=0.9, rank=1, segment=seg, insight=ins)
    pack = build_briefing_pack("q", "semantic", [comp])
    assert pack.top_insight.doc_id == "i1"  # insight extracted from the compound
    assert pack.supporting_segments[0].doc_id == "s1"  # segment too


def test_confidence_p50():
    results = [_ins("i1", conf=0.5), _ins("i2", conf=0.7), _ins("i3", conf=0.9)]
    pack = build_briefing_pack("q", "semantic", results)
    assert pack.confidence_p50 == 0.7


def test_empty_results_pack():
    pack = build_briefing_pack("q", "semantic", [])
    assert pack.top_insight is None and pack.supporting_segments == []
    assert pack.coverage_summary["episode_count"] == 0


def test_render_litm_order_and_stable_null_fields():
    pack = build_briefing_pack(
        "q", "semantic", [_ins("i1", "claim")], canonical_entity={"name": "Alice"}
    )
    text = pack.render()
    # critical before supporting before caveats.
    assert (
        text.index("[CRITICAL GROUNDING]")
        < text.index("[SUPPORTING EVIDENCE]")
        < text.index("[CAVEATS]")
    )
    assert "Entity: Alice" in text
    assert pack.top_contradiction is None and pack.coverage_gaps == []  # stable until inputs exist


def test_token_budget_trims_segments():
    long = "word " * 200
    results = [_ins("i1", "claim")] + [_seg(f"s{i}", long) for i in range(5)]
    pack = build_briefing_pack("q", "semantic", results, max_tokens=50)
    assert pack.token_count <= 50
    assert len(pack.supporting_segments) < 5  # trimmed to fit
