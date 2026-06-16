"""Unit tests for RRF fusion + dedup + router + RetrievalLayer (RFC-090 §3.3–3.6, #856)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.backend import ScoredResult
from podcast_scraper.search.dedup import deduplicate
from podcast_scraper.search.fusion import rrf_fuse
from podcast_scraper.search.retrieval import RetrievalLayer
from podcast_scraper.search.router import (
    classify_query,
    signal_weights_for,
    tier_weights_for,
)

pytestmark = pytest.mark.unit


def _sr(doc_id, rank, signal, tier, score=1.0, payload=None):
    return ScoredResult(doc_id, score, rank, payload or {}, signal, tier)


# --- RRF fusion -----------------------------------------------------------------


def test_rrf_doc_in_both_lists_ranks_highest():
    bm25 = [_sr("a", 1, "bm25", "insight"), _sr("b", 2, "bm25", "segment")]
    vector = [_sr("b", 1, "vector", "segment"), _sr("c", 2, "vector", "insight")]
    fused = rrf_fuse([bm25, vector])
    assert fused[0].doc_id == "b"  # appears in both → highest fused score
    assert all(r.signal == "rrf" for r in fused)
    assert [r.rank for r in fused] == [1, 2, 3]


def test_rrf_signal_weights_boost():
    bm25 = [_sr("a", 1, "bm25", "segment")]
    vector = [_sr("z", 1, "vector", "segment")]
    fused = rrf_fuse([bm25, vector], signal_weights={"bm25": 2.0, "vector": 0.1})
    assert fused[0].doc_id == "a"  # bm25 weighted far higher


def test_rrf_skips_empty_lists():
    assert rrf_fuse([[], [_sr("a", 1, "vector", "segment")], []])[0].doc_id == "a"


# --- compound dedup -------------------------------------------------------------


def test_dedup_compound_via_source_segment_id():
    res = [
        _sr("seg1", 1, "vector", "segment", score=0.7),
        _sr("ins1", 2, "vector", "insight", score=0.9, payload={"source_segment_id": "seg1"}),
    ]
    out = deduplicate(res)
    assert len(out) == 1
    comp = out[0]
    assert comp.source_tier == "compound" and comp.doc_id == "seg1"
    assert comp.score == 0.9 and comp.rank == 1  # max score, min rank
    assert comp.segment.doc_id == "seg1" and comp.insight.doc_id == "ins1"


def test_dedup_compound_via_linked_insight_ids():
    res = [
        _sr("seg1", 1, "bm25", "segment", payload={"linked_insight_ids": ["ins1"]}),
        _sr("ins1", 2, "bm25", "insight"),
    ]
    out = deduplicate(res)
    assert len(out) == 1 and out[0].source_tier == "compound"


def test_dedup_passthrough_when_unlinked():
    res = [
        _sr("seg1", 1, "vector", "segment", score=0.5),
        _sr("ins9", 2, "vector", "insight", score=0.8, payload={"source_segment_id": "other"}),
    ]
    out = deduplicate(res)
    assert {r.doc_id for r in out} == {"seg1", "ins9"}
    assert out[0].doc_id == "ins9"  # sorted by score desc


# --- router ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Sam Altman", "entity_lookup"),
        ("what was the exact quote about scaling", "raw_evidence"),
        ("how has AI reasoning changed over time", "temporal_tracking"),
        ("Acquired vs The Journal on AI", "cross_show_synthesis"),
        ("tell me about scaling laws", "semantic"),
    ],
)
def test_classify_query(text, expected):
    assert classify_query(text) == expected


def test_weights_lookup_and_fallback():
    assert signal_weights_for("raw_evidence")["bm25"] == 1.5
    assert signal_weights_for("entity_lookup")["bm25"] == 1.4
    assert tier_weights_for("raw_evidence")["segment"] == 1.3
    assert signal_weights_for("unknown") == signal_weights_for("semantic")


# --- RetrievalLayer -------------------------------------------------------------


class _FakeBackend:
    def __init__(self, bm25, vector):
        self._bm25, self._vector = bm25, vector

    def search_bm25(self, query):
        return list(self._bm25)

    def search_vector(self, query):
        return list(self._vector)


def test_retrieval_hybrid_fuses_and_dedups():
    bm25 = [_sr("seg1", 1, "bm25", "segment", payload={"linked_insight_ids": ["ins1"]})]
    vector = [_sr("ins1", 1, "vector", "insight")]
    layer = RetrievalLayer(_FakeBackend(bm25, vector))
    out = layer.retrieve("Sam Altman", [0.1, 0.2])
    # seg1 + ins1 are linked → one compound result.
    assert len(out) == 1 and out[0].source_tier == "compound"


def test_retrieval_single_signal_skips_fusion():
    bm25 = [_sr("seg1", 1, "bm25", "segment"), _sr("seg2", 2, "bm25", "segment")]
    layer = RetrievalLayer(_FakeBackend(bm25, []))
    out = layer.retrieve("x", [0.0], signals="bm25")
    assert [r.doc_id for r in out] == ["seg1", "seg2"]  # deduped passthrough, order kept


def test_retrieval_classify_helper():
    layer = RetrievalLayer(_FakeBackend([], []))
    assert layer.classify("Sam Altman") == "entity_lookup"


class _FakeHybridBackend:
    """Backend exposing native ``search_hybrid`` (ADR-099 Stage 2)."""

    def __init__(self, hybrid):
        self._hybrid = hybrid
        self.bm25_called = False
        self.vector_called = False

    def search_hybrid(self, query):
        return list(self._hybrid)

    def search_bm25(self, query):
        self.bm25_called = True
        return []

    def search_vector(self, query):
        self.vector_called = True
        return []


def test_retrieval_uses_native_hybrid_when_available():
    # ADR-099 Stage 2 (#995): the default hybrid signal uses the backend's in-engine
    # search_hybrid and does NOT fan out to the Python bm25+vector+RRF path.
    hits = [_sr("a", 1, "hybrid", "segment"), _sr("b", 2, "hybrid", "insight")]
    be = _FakeHybridBackend(hits)
    out = RetrievalLayer(be).retrieve("query", [0.1, 0.2])  # signals="hybrid" (default)
    assert [r.doc_id for r in out] == ["a", "b"]
    assert not be.bm25_called and not be.vector_called  # native path, no fan-out


def test_retrieval_explicit_bm25_signal_bypasses_native_hybrid():
    # An explicit single-signal request still uses the fan-out methods even when
    # search_hybrid exists (native hybrid is only for the default hybrid signal).
    be = _FakeHybridBackend([])
    RetrievalLayer(be).retrieve("query", [0.1], signals="bm25")
    assert be.bm25_called and not be.vector_called
