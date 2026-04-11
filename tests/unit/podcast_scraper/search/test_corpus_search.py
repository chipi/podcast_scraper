"""Unit tests for ``run_corpus_search`` (shared HTTP + CLI search)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.search.corpus_search import dedupe_kg_surface_rows, run_corpus_search
from podcast_scraper.search.protocol import SearchResult


@pytest.mark.unit
def test_run_corpus_search_empty_query(tmp_path: Path) -> None:
    out = run_corpus_search(tmp_path, "  \t")
    assert out.error == "empty_query"


@pytest.mark.unit
def test_run_corpus_search_no_index(tmp_path: Path) -> None:
    out = run_corpus_search(tmp_path, "hello")
    assert out.error == "no_index"
    assert "search" in (out.detail or "")


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.FaissVectorStore.load", side_effect=OSError("boom"))
def test_run_corpus_search_load_failed(mock_load: MagicMock, tmp_path: Path) -> None:
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")
    out = run_corpus_search(tmp_path, "q")
    assert out.error == "load_failed"
    mock_load.assert_called_once()


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
def test_run_corpus_search_embed_failed_wrong_shape(mock_enc: MagicMock, tmp_path: Path) -> None:
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")

    mock_store = MagicMock()
    mock_store.stats.return_value.embedding_model = "minilm-l6"

    mock_enc.return_value = [[0.1, 0.2]]  # not a flat float list

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        out = run_corpus_search(tmp_path, "q", embedding_model="minilm-l6")
    assert out.error == "embed_failed"


@pytest.mark.unit
@patch(
    "podcast_scraper.search.corpus_search.embedding_loader.encode",
    side_effect=RuntimeError("embed"),
)
def test_run_corpus_search_embed_failed_exception(mock_enc: MagicMock, tmp_path: Path) -> None:
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")

    mock_store = MagicMock()
    mock_store.stats.return_value.embedding_model = "minilm-l6"

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        out = run_corpus_search(tmp_path, "q")
    assert out.error == "embed_failed"
    mock_enc.assert_called_once()


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
def test_run_corpus_search_success_multi_type_filter(mock_enc: MagicMock, tmp_path: Path) -> None:
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")

    emb: List[float] = [0.0, 1.0, 0.0]
    mock_enc.return_value = emb

    hits = [
        SearchResult("a", 0.9, {"doc_type": "insight", "episode_id": "e1", "text": "t1"}),
        SearchResult("b", 0.8, {"doc_type": "quote", "episode_id": "e1", "text": "t2"}),
        SearchResult("c", 0.7, {"doc_type": "summary", "episode_id": "e1", "text": "t3"}),
    ]
    mock_store = MagicMock()
    mock_store.ntotal = 3
    mock_store.stats.return_value.embedding_model = "m1"
    mock_store.search.return_value = hits

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        out = run_corpus_search(
            tmp_path,
            "climate",
            doc_types=["insight", "quote"],
            top_k=1,
        )
    assert out.error is None
    assert len(out.results) == 1
    assert out.results[0]["doc_id"] == "a"


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
def test_run_corpus_search_backfills_source_metadata_relative_path(
    mock_enc: MagicMock, tmp_path: Path
) -> None:
    """Stale FAISS metadata without Library path still get it from corpus *.metadata.json."""
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True)
    (meta_dir / "ep1.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": "e1", "title": "Backfill Episode Title"},
                "feed": {
                    "feed_id": "https://example.com/feed",
                    "title": "Backfill Feed Title",
                },
            }
        ),
        encoding="utf-8",
    )
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")
    mock_enc.return_value = [0.0, 1.0, 0.0]
    hits = [
        SearchResult(
            "a",
            0.9,
            {
                "doc_type": "summary",
                "episode_id": "e1",
                "feed_id": "https://example.com/feed",
                "text": "hello world",
            },
        ),
    ]
    mock_store = MagicMock()
    mock_store.ntotal = 1
    mock_store.stats.return_value.embedding_model = "m1"
    mock_store.search.return_value = hits

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        out = run_corpus_search(tmp_path, "climate", top_k=5)
    assert out.error is None
    assert len(out.results) == 1
    m0 = out.results[0]["metadata"]
    assert m0["source_metadata_relative_path"] == "metadata/ep1.metadata.json"
    assert m0["episode_title"] == "Backfill Episode Title"
    assert m0["feed_title"] == "Backfill Feed Title"


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
def test_run_corpus_search_backfills_when_vector_hit_lacks_feed_id(
    mock_enc: MagicMock, tmp_path: Path
) -> None:
    """Episode-only scope key resolves metadata file when FAISS row has no feed_id."""
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True)
    (meta_dir / "ep1.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": "e1", "title": "Solo Ep Title"},
                "feed": {"feed_id": "https://example.com/feed", "title": "Solo Feed Title"},
            }
        ),
        encoding="utf-8",
    )
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")
    mock_enc.return_value = [0.0, 1.0, 0.0]
    hits = [
        SearchResult(
            "a",
            0.9,
            {
                "doc_type": "summary",
                "episode_id": "e1",
                "text": "hello world",
            },
        ),
    ]
    mock_store = MagicMock()
    mock_store.ntotal = 1
    mock_store.stats.return_value.embedding_model = "m1"
    mock_store.search.return_value = hits

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        out = run_corpus_search(tmp_path, "climate", top_k=5)
    assert out.error is None
    m0 = out.results[0]["metadata"]
    assert m0["source_metadata_relative_path"] == "metadata/ep1.metadata.json"
    assert m0["episode_title"] == "Solo Ep Title"
    assert m0["feed_title"] == "Solo Feed Title"


@pytest.mark.unit
def test_dedupe_kg_surface_rows_merges_same_surface_three_episodes() -> None:
    rows = [
        {
            "doc_id": "a",
            "score": 0.9,
            "text": "The Wall Street Journal",
            "metadata": {"doc_type": "kg_entity", "episode_id": "e1"},
        },
        {
            "doc_id": "b",
            "score": 0.8,
            "text": "The Wall Street Journal",
            "metadata": {"doc_type": "kg_entity", "episode_id": "e2"},
        },
        {
            "doc_id": "c",
            "score": 0.7,
            "text": "The Wall Street Journal",
            "metadata": {"doc_type": "kg_entity", "episode_id": "e3"},
        },
    ]
    out = dedupe_kg_surface_rows(rows)
    assert len(out) == 1
    meta = out[0]["metadata"]
    assert meta["kg_surface_match_count"] == 3
    assert set(meta["kg_surface_episode_ids"]) == {"e1", "e2", "e3"}
    assert out[0]["score"] == 0.9


@pytest.mark.unit
def test_dedupe_kg_surface_rows_interleaved_insight() -> None:
    rows = [
        {
            "doc_id": "a",
            "score": 0.9,
            "text": "Acme Corp",
            "metadata": {"doc_type": "kg_entity", "episode_id": "e1"},
        },
        {
            "doc_id": "i",
            "score": 0.5,
            "text": "insight text",
            "metadata": {"doc_type": "insight", "episode_id": "e1"},
        },
        {
            "doc_id": "b",
            "score": 0.8,
            "text": "Acme Corp",
            "metadata": {"doc_type": "kg_entity", "episode_id": "e2"},
        },
    ]
    out = dedupe_kg_surface_rows(rows)
    assert len(out) == 2
    assert out[0]["metadata"]["doc_type"] == "kg_entity"
    assert out[0]["metadata"]["kg_surface_match_count"] == 2
    assert out[1]["metadata"]["doc_type"] == "insight"


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
def test_run_corpus_search_dedupe_kg_surfaces_true_collapses(
    mock_enc: MagicMock, tmp_path: Path
) -> None:
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")
    mock_enc.return_value = [0.0, 1.0, 0.0]
    hits = [
        SearchResult(
            "a",
            0.9,
            {
                "doc_type": "kg_entity",
                "episode_id": "e1",
                "text": "Acme Corp",
            },
        ),
        SearchResult(
            "b",
            0.85,
            {
                "doc_type": "kg_entity",
                "episode_id": "e2",
                "text": "Acme Corp",
            },
        ),
        SearchResult(
            "c",
            0.8,
            {
                "doc_type": "kg_entity",
                "episode_id": "e3",
                "text": "Acme Corp",
            },
        ),
    ]
    mock_store = MagicMock()
    mock_store.ntotal = 3
    mock_store.stats.return_value.embedding_model = "m1"
    mock_store.search.return_value = hits

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        out = run_corpus_search(tmp_path, "q", top_k=10, dedupe_kg_surfaces=True)
    assert out.error is None
    assert len(out.results) == 1
    assert out.results[0]["metadata"]["kg_surface_match_count"] == 3


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
def test_run_corpus_search_dedupe_kg_surfaces_false_keeps_rows(
    mock_enc: MagicMock, tmp_path: Path
) -> None:
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")
    mock_enc.return_value = [0.0, 1.0, 0.0]
    hits = [
        SearchResult(
            "a",
            0.9,
            {"doc_type": "kg_entity", "episode_id": "e1", "text": "Acme Corp"},
        ),
        SearchResult(
            "b",
            0.85,
            {"doc_type": "kg_entity", "episode_id": "e2", "text": "Acme Corp"},
        ),
        SearchResult(
            "c",
            0.8,
            {"doc_type": "kg_entity", "episode_id": "e3", "text": "Acme Corp"},
        ),
    ]
    mock_store = MagicMock()
    mock_store.ntotal = 3
    mock_store.stats.return_value.embedding_model = "m1"
    mock_store.search.return_value = hits

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        out = run_corpus_search(tmp_path, "q", top_k=10, dedupe_kg_surfaces=False)
    assert out.error is None
    assert len(out.results) == 3


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
def test_run_corpus_search_top_k_clamped(mock_enc: MagicMock, tmp_path: Path) -> None:
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")
    mock_enc.return_value = [1.0, 0.0, 0.0]

    mock_store = MagicMock()
    mock_store.ntotal = 5
    mock_store.stats.return_value.embedding_model = "m1"
    mock_store.search.return_value = []

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        run_corpus_search(tmp_path, "q", top_k=0)
        run_corpus_search(tmp_path, "q", top_k=9999)

    assert mock_store.search.call_count == 2
    assert mock_store.search.call_args_list[0].kwargs["top_k"] >= 1
    assert mock_store.search.call_args_list[1].kwargs["top_k"] <= 2500


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
def test_run_corpus_search_embedding_model_override_logs_mismatch(
    mock_enc: MagicMock, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")
    mock_enc.return_value = [1.0, 0.0]

    mock_store = MagicMock()
    mock_store.ntotal = 1
    mock_store.stats.return_value.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    mock_store.search.return_value = []

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        with caplog.at_level("WARNING"):
            run_corpus_search(
                tmp_path,
                "q",
                embedding_model="sentence-transformers/all-mpnet-base-v2",
            )
    assert any("differs from index model" in r.message for r in caplog.records)


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
def test_run_corpus_search_single_doc_type_sets_faiss_filter(
    mock_enc: MagicMock, tmp_path: Path
) -> None:
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")
    mock_enc.return_value = [1.0, 0.0]

    mock_store = MagicMock()
    mock_store.ntotal = 1
    mock_store.stats.return_value.embedding_model = "m1"
    mock_store.search.return_value = []

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        run_corpus_search(tmp_path, "q", doc_types=["insight"])

    mock_store.search.assert_called_once()
    assert mock_store.search.call_args.kwargs["filters"] == {"doc_type": "insight"}


@pytest.mark.unit
@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
def test_run_corpus_search_blank_doc_types_normalized_away(
    mock_enc: MagicMock, tmp_path: Path
) -> None:
    idx = tmp_path / "search"
    idx.mkdir(parents=True)
    (idx / "vectors.faiss").write_bytes(b"x")
    mock_enc.return_value = [1.0, 0.0]

    mock_store = MagicMock()
    mock_store.ntotal = 1
    mock_store.stats.return_value.embedding_model = "m1"
    mock_store.search.return_value = []

    with patch(
        "podcast_scraper.search.corpus_search.FaissVectorStore.load", return_value=mock_store
    ):
        run_corpus_search(tmp_path, "q", doc_types=["  ", ""])

    assert mock_store.search.call_args.kwargs["filters"] is None
