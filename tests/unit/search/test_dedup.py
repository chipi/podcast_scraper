"""Unit tests for compound-result deduplication (RFC-090 §3.4)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.backend import CompoundResult, ScoredResult
from podcast_scraper.search.dedup import deduplicate

pytestmark = pytest.mark.unit


def _seg(doc_id, score, rank, linked=None):
    return ScoredResult(
        doc_id, score, rank, {"linked_insight_ids": linked or []}, "bm25", "segment"
    )


def _ins(doc_id, score, rank, src_seg=None):
    return ScoredResult(doc_id, score, rank, {"source_segment_id": src_seg}, "vector", "insight")


def test_compound_via_insight_source_segment_id():
    seg = _seg("s1", 0.6, 3)
    ins = _ins("i1", 0.9, 1, src_seg="s1")
    out = deduplicate([seg, ins])
    assert len(out) == 1
    comp = out[0]
    assert isinstance(comp, CompoundResult)
    assert comp.doc_id == "s1" and comp.segment is seg and comp.insight is ins
    assert comp.score == 0.9 and comp.rank == 1  # max score, min rank


def test_compound_via_segment_linked_insight_ids():
    seg = _seg("s1", 0.8, 2, linked=["i1"])
    ins = _ins("i1", 0.5, 4)  # no source_segment_id → uses the reverse link
    out = deduplicate([seg, ins])
    assert len(out) == 1 and isinstance(out[0], CompoundResult)
    assert out[0].score == 0.8 and out[0].rank == 2


def test_unlinked_results_pass_through():
    seg = _seg("s1", 0.7, 1)
    ins = _ins("i1", 0.6, 2)  # no link either direction
    out = deduplicate([seg, ins])
    assert len(out) == 2 and all(isinstance(r, ScoredResult) for r in out)
    assert [r.doc_id for r in out] == ["s1", "i1"]  # sorted by score desc


def test_one_segment_consumed_once():
    seg = _seg("s1", 0.5, 5, linked=["i1", "i2"])
    i1 = _ins("i1", 0.9, 1, src_seg="s1")
    i2 = _ins("i2", 0.8, 2, src_seg="s1")
    out = deduplicate([seg, i1, i2])
    compounds = [r for r in out if isinstance(r, CompoundResult)]
    assert len(compounds) == 1  # segment consumed by the first insight only
    assert any(r.doc_id == "i2" and isinstance(r, ScoredResult) for r in out)  # i2 left standalone
