"""Aux-tier coverage: kg/quote/summary indexed + searchable (RFC-090 full coverage)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.providers.ml import embedding_loader  # noqa: E402
from podcast_scraper.search import two_tier_indexer as tti  # noqa: E402
from podcast_scraper.search.backend import AuxDocument, SearchQuery  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _emb(t):
    return list(embedding_loader.encode(t, _MODEL, return_numpy=False))


def test_backend_upsert_search_delete_aux(tmp_path):
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=384)
    b.upsert_aux(
        AuxDocument(
            id="kg_topic:1",
            text="interest rates",
            show_id="A",
            episode_id="e1",
            doc_type="kg_topic",
            embedding=_emb("interest rates"),
        )
    )
    b.create_indices()

    assert b.health()["aux"] == 1
    hits = b.search_bm25(SearchQuery(text="interest rates", embedding=[], tier="aux"))
    assert any(h.doc_id == "kg_topic:1" and h.source_tier == "aux" for h in hits)
    # "all" tier spans aux too.
    assert any(
        h.doc_id == "kg_topic:1"
        for h in b.search_bm25(SearchQuery(text="interest rates", embedding=[], tier="all"))
    )

    b.delete("kg_topic:1", "all")  # all-tier delete must hit aux
    assert b.health()["aux"] == 0


def test_indexer_routes_aux_doc_types(tmp_path, monkeypatch):
    rows = [
        ("kg_topic:1", "oil markets", {"doc_type": "kg_topic", "episode_id": "e1", "feed_id": "s"}),
        (
            "quote:1",
            "he said the dollar",
            {"doc_type": "quote", "episode_id": "e1", "feed_id": "s"},
        ),
        ("summary:1", "episode recap", {"doc_type": "summary", "episode_id": "e1", "feed_id": "s"}),
        (
            "kg_entity:1",
            "Jerome Powell",
            {"doc_type": "kg_entity", "episode_id": "e1", "feed_id": "s"},
        ),
    ]
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    monkeypatch.setattr(
        tti, "discover_metadata_files", lambda root: [corpus / "metadata" / "e1.json"]
    )
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "e1"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: corpus)
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)

    stats = tti.build_two_tier_index(corpus, corpus / "search" / "lance_index")
    assert stats.aux == 4 and stats.segments == 0 and stats.insights == 0
    assert LanceDBBackend(str(corpus / "search" / "lance_index")).health()["aux"] == 4


def test_indexer_routes_episode_metadata_doc_types_into_aux_tier(tmp_path, monkeypatch):
    """2026-07-22: episode-level metadata surfaces (``episode_title`` /
    ``episode_description`` / ``summary_short``) added to ``_AUX_DOC_TYPES``
    so the rows land in the aux table instead of being silently dropped
    by the else branch in ``build_two_tier_index``."""
    rows = [
        (
            "episode_title:sha256_x:ep-a",
            "Are AI stocks the new railroad bonds?",
            {
                "doc_type": "episode_title",
                "matched_field": "title",
                "episode_id": "ep-a",
                "feed_id": "s",
            },
        ),
        (
            "episode_description:sha256_x:ep-a",
            "Transformative technologies create speculative frenzies.",
            {
                "doc_type": "episode_description",
                "matched_field": "description",
                "episode_id": "ep-a",
                "feed_id": "s",
            },
        ),
        (
            "summary_short:sha256_x:ep-a",
            "Financial history provides valuable lessons for markets.",
            {
                "doc_type": "summary_short",
                "matched_field": "summary",
                "episode_id": "ep-a",
                "feed_id": "s",
            },
        ),
    ]
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    monkeypatch.setattr(
        tti, "discover_metadata_files", lambda root: [corpus / "metadata" / "e1.json"]
    )
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "ep-a"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: corpus)
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)

    stats = tti.build_two_tier_index(corpus, corpus / "search" / "lance_index")
    assert stats.aux == 3 and stats.segments == 0 and stats.insights == 0
    assert LanceDBBackend(str(corpus / "search" / "lance_index")).health()["aux"] == 3
