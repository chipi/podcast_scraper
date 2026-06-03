"""Tests for the FAISS → two-tier LanceDB migration (RFC-090 Stage 4, #858).

Builds a tiny real FAISS store (4-dim), migrates it, and asserts the two tiers
land with reused embeddings and reusable indices.
"""

from __future__ import annotations

import pytest

from podcast_scraper.search.backend import SearchQuery

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")
pytest.importorskip("faiss")

from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402
from podcast_scraper.search.faiss_store import FaissVectorStore  # noqa: E402
from podcast_scraper.search.migration import migrate_faiss_to_lance  # noqa: E402


def _build_faiss(tmp_path):
    store = FaissVectorStore(4, index_dir=tmp_path / "faiss")
    store.upsert(
        "insight:1",
        [0.1, 0.2, 0.3, 0.4],
        {
            "doc_type": "insight",
            "text": "Altman on AI scaling",
            "episode_id": "ep1",
            "feed_id": "showA",
            "grounded": True,
        },
    )
    store.upsert(
        "chunk:1",
        [0.9, 0.1, 0.0, 0.1],
        {
            "doc_type": "transcript",
            "text": "raw transcript chunk",
            "episode_id": "ep1",
            "feed_id": "showA",
            "timestamp_start_ms": 2000,
            "timestamp_end_ms": 5000,
        },
    )
    # A non-two-tier doc type that must be skipped.
    store.upsert(
        "kg_entity:1",
        [0.5, 0.5, 0.0, 0.0],
        {"doc_type": "kg_entity", "text": "Sam Altman", "episode_id": "ep1"},
    )
    store.persist()
    return tmp_path / "faiss"


def test_migration_maps_tiers_including_aux(tmp_path):
    faiss_dir = _build_faiss(tmp_path)
    stats = migrate_faiss_to_lance(faiss_dir, tmp_path / "lance")
    assert stats.insights == 1
    assert stats.segments == 1
    assert stats.aux == 1  # kg_entity → aux tier (full coverage), not skipped
    assert stats.skipped == 0
    assert stats.embed_dim == 4

    backend = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    health = backend.health()
    assert health["insights"] == 1 and health["segments"] == 1 and health["aux"] == 1

    # Migration records the FAISS index's model so the query path matches it.
    meta = backend.read_index_meta()
    assert meta is not None and meta["embedding_model"]


def test_migration_preserves_searchable_payload(tmp_path):
    faiss_dir = _build_faiss(tmp_path)
    migrate_faiss_to_lance(faiss_dir, tmp_path / "lance")
    backend = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)

    # BM25 over the migrated insight tier finds the reused text + timestamp mapping.
    hits = backend.search_bm25(SearchQuery(text="Altman scaling", embedding=[], tier="insight"))
    assert any(h.doc_id == "insight:1" for h in hits)

    seg_hits = backend.search_bm25(
        SearchQuery(text="transcript chunk", embedding=[], tier="segment")
    )
    seg = next(h for h in seg_hits if h.doc_id == "chunk:1")
    assert seg.payload["start_time"] == 2.0 and seg.payload["end_time"] == 5.0  # ms→s


def test_migration_limit_per_tier(tmp_path):
    faiss_dir = _build_faiss(tmp_path)
    stats = migrate_faiss_to_lance(faiss_dir, tmp_path / "lance", limit_per_tier=0)
    assert stats.segments == 0 and stats.insights == 0
