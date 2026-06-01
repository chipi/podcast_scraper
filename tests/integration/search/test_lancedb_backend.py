"""Tests for the LanceDB two-tier backend (RFC-090 §3.7, #855).

Uses a real embedded LanceDB in a temp dir with tiny 4-dim vectors (fast).
"""

from __future__ import annotations

import pytest

from podcast_scraper.search.backend import (
    InsightDocument,
    SearchBackend,
    SearchQuery,
    SegmentDocument,
)

pytestmark = pytest.mark.integration

lancedb = pytest.importorskip("lancedb")

from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402


def _seg(i, text, show, emb):
    return SegmentDocument(
        id=i, text=text, show_id=show, episode_id="ep1", start_time=0.0, end_time=1.0, embedding=emb
    )


@pytest.fixture
def backend(tmp_path):
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    b.upsert_segment(_seg("s1", "Sam Altman talks about OpenAI", "A", [0.1, 0.2, 0.3, 0.4]))
    b.upsert_segment(_seg("s2", "Tim Cook and Apple earnings", "B", [0.9, 0.1, 0.0, 0.1]))
    b.upsert_insight(
        InsightDocument(
            id="insight:1",
            text="Altman argues AI scaling continues",
            show_id="A",
            episode_id="ep1",
            entity_type="claim",
            confidence=0.9,
            derived=False,
            embedding=[0.1, 0.2, 0.3, 0.45],
        )
    )
    b.create_indices()
    return b


def test_satisfies_protocol(backend):
    assert isinstance(backend, SearchBackend)


def test_bm25_named_entity(backend):
    # BM25 retrieves the proper noun across both tiers.
    res = backend.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="all"))
    ids = {r.doc_id for r in res}
    assert "s1" in ids and "insight:1" in ids
    assert all(r.signal == "bm25" for r in res)
    assert res[0].rank == 1


def test_vector_search_orders_by_similarity(backend):
    res = backend.search_vector(
        SearchQuery(text="", embedding=[0.1, 0.2, 0.3, 0.4], tier="segment")
    )
    assert res and res[0].doc_id == "s1"  # closest segment
    assert all(r.signal == "vector" and r.source_tier == "segment" for r in res)
    # similarity score is higher-is-better.
    assert res[0].score >= res[-1].score


def test_tier_filter_segment_only(backend):
    res = backend.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="segment"))
    assert {r.source_tier for r in res} == {"segment"}


def test_payload_excludes_embedding(backend):
    res = backend.search_bm25(SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="all"))
    assert "embedding" not in res[0].payload
    assert res[0].payload.get("show_id") == "A"


def test_filters_where_show(backend):
    res = backend.search_bm25(
        SearchQuery(text="Altman", embedding=[0, 0, 0, 0], tier="segment", filters={"show_id": "B"})
    )
    assert res == []  # s1 is show A; filtered out


def test_health_and_delete(backend):
    h = backend.health()
    assert h["status"] == "ok" and h["segments"] == 2 and h["insights"] == 1
    backend.delete("s1", "segment")
    assert backend.health()["segments"] == 1
