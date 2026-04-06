"""Unit tests for ``run_corpus_search`` (shared HTTP + CLI search)."""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.search.corpus_search import run_corpus_search
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
