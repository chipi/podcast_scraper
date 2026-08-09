"""Integration tests for the hybrid serving bridge (RFC-090 Phase 2 wire-live).

Real LanceDB index + real MiniLM query embedding; asserts candidate mapping, tier
scoping, and the no_index signal.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.providers.ml import embedding_loader  # noqa: E402
from podcast_scraper.search import hybrid_search as hs  # noqa: E402
from podcast_scraper.search.backend import InsightDocument, SegmentDocument  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _emb(text):
    return list(embedding_loader.encode(text, _MODEL, return_numpy=False))


@pytest.fixture
def corpus(tmp_path):
    idx = tmp_path / "corpus" / "search" / "lance_index"
    idx.parent.mkdir(parents=True)
    b = LanceDBBackend(str(idx), embed_dim=384)
    b.upsert_segment(
        SegmentDocument(
            id="chunk:1",
            text="raw transcript about AI scaling laws",
            show_id="A",
            episode_id="ep1",
            start_time=2.0,
            end_time=5.0,
            embedding=_emb("raw transcript about AI scaling laws"),
        )
    )
    b.upsert_insight(
        InsightDocument(
            id="insight:1",
            text="Altman argues AI scaling continues",
            show_id="A",
            episode_id="ep1",
            entity_type="insight",
            confidence=0.8,
            derived=True,
            embedding=_emb("Altman argues AI scaling continues"),
        )
    )
    b.create_indices()
    return tmp_path / "corpus"


def test_maps_both_tiers_with_doc_type_and_timestamps(corpus):
    rows = hs.hybrid_candidates(corpus, "AI scaling", top_k=5)
    by_id = {r.doc_id: r for r in rows}
    assert by_id["insight:1"].metadata["doc_type"] == "insight"
    seg = by_id["chunk:1"]
    assert seg.metadata["doc_type"] == "transcript"  # segment → transcript vocab
    assert seg.metadata["timestamp_start_ms"] == 2000  # seconds → ms
    assert seg.metadata["episode_id"] == "ep1"


def test_tier_scoping(corpus):
    assert [
        r.doc_id for r in hs.hybrid_candidates(corpus, "AI", top_k=5, doc_types=["insight"])
    ] == ["insight:1"]
    assert [
        r.doc_id for r in hs.hybrid_candidates(corpus, "AI", top_k=5, doc_types=["transcript"])
    ] == ["chunk:1"]


def test_missing_index_returns_none(tmp_path):
    # No LanceDB index → None (no_index signal), not an empty list.
    assert hs.hybrid_candidates(tmp_path / "nope", "AI", top_k=5) is None


def test_grounded_flag_roundtrips_through_lancedb(tmp_path):
    """#19 end-to-end: the insight ``derived`` field round-trips through REAL LanceDB storage →
    hybrid_candidates → grounded metadata, so grounded_only can drop ungrounded insights (the unit
    test used a hand-built result; this proves the live-index path the prod bug hit)."""
    idx = tmp_path / "corpus" / "search" / "lance_index"
    idx.parent.mkdir(parents=True)
    b = LanceDBBackend(str(idx), embed_dim=384)
    b.upsert_insight(
        InsightDocument(
            id="insight:grounded",
            text="Altman argues AI scaling continues",
            show_id="A",
            episode_id="ep1",
            entity_type="insight",
            confidence=0.8,
            derived=True,
            embedding=_emb("Altman argues AI scaling continues"),
        )
    )
    b.upsert_insight(
        InsightDocument(
            id="insight:ungrounded",
            text="AI scaling is a speculative claim",
            show_id="A",
            episode_id="ep1",
            entity_type="insight",
            confidence=0.4,
            derived=False,
            embedding=_emb("AI scaling is a speculative claim"),
        )
    )
    b.create_indices()

    rows = hs.hybrid_candidates(tmp_path / "corpus", "AI scaling", top_k=10, doc_types=["insight"])
    grounded = {r.doc_id: r.metadata.get("grounded") for r in rows}
    assert grounded.get("insight:grounded") is True
    assert grounded.get("insight:ungrounded") is False
    # The grounded_only filter (explore.py: `if grounded_only and not ins.grounded`) keeps only the
    # grounded row — before #19 the flag was absent so EVERY insight was dropped.
    kept = [r.doc_id for r in rows if r.metadata.get("grounded")]
    assert kept == ["insight:grounded"]
