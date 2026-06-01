"""Unit tests for the two-tier SearchBackend contracts (RFC-090 §3.1–3.2, #855)."""

from __future__ import annotations

from typing import Dict, List

import pytest

from podcast_scraper.search.backend import (
    CompoundResult,
    InsightDocument,
    ScoredResult,
    SearchBackend,
    SearchQuery,
    SegmentDocument,
)

pytestmark = pytest.mark.unit


def test_segment_document_defaults():
    seg = SegmentDocument(
        id="ep1_chunk_0", text="hello", show_id="s", episode_id="ep1", start_time=0.0, end_time=1.0
    )
    assert seg.source_tier == "segment"
    assert seg.embedding == [] and seg.linked_insight_ids == [] and seg.speaker_id is None


def test_insight_document_defaults():
    ins = InsightDocument(
        id="insight:1",
        text="claim",
        show_id="s",
        episode_id="ep1",
        entity_type="opinion",
        confidence=0.9,
        derived=False,
    )
    assert ins.source_tier == "insight"
    assert ins.embedding == [] and ins.source_segment_id is None


def test_search_query_defaults():
    q = SearchQuery(text="sam altman", embedding=[0.1, 0.2])
    assert q.k == 20 and q.tier == "all" and q.filters == {}


def test_compound_result_wraps_both_tiers():
    seg = ScoredResult("ep1_chunk_0", 0.8, 1, {}, "vector", "segment")
    ins = ScoredResult("insight:1", 0.9, 1, {}, "vector", "insight")
    comp = CompoundResult(doc_id="ep1_chunk_0", score=0.9, rank=1, segment=seg, insight=ins)
    assert comp.source_tier == "compound" and comp.signal == "rrf"
    assert comp.segment is seg and comp.insight is ins


class _FakeBackend:
    """Minimal in-memory backend used to verify protocol conformance (no deps)."""

    def __init__(self) -> None:
        self.segments: Dict[str, SegmentDocument] = {}
        self.insights: Dict[str, InsightDocument] = {}

    def search_bm25(self, query: SearchQuery) -> List[ScoredResult]:
        return []

    def search_vector(self, query: SearchQuery) -> List[ScoredResult]:
        return []

    def upsert_segment(self, doc: SegmentDocument) -> None:
        self.segments[doc.id] = doc

    def upsert_insight(self, doc: InsightDocument) -> None:
        self.insights[doc.id] = doc

    def delete(self, doc_id: str, tier) -> None:
        self.segments.pop(doc_id, None)
        self.insights.pop(doc_id, None)

    def create_indices(self) -> None:
        pass

    def health(self) -> Dict:
        return {"status": "ok", "segments": len(self.segments), "insights": len(self.insights)}


def test_fake_backend_satisfies_protocol():
    backend = _FakeBackend()
    assert isinstance(backend, SearchBackend)  # runtime_checkable structural check
    backend.upsert_segment(SegmentDocument("ep1_chunk_0", "t", "s", "ep1", 0.0, 1.0))
    assert backend.health()["segments"] == 1
