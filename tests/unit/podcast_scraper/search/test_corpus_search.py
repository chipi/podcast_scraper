"""Unit tests for ``corpus_search`` (dedupe, filters, mocked FAISS path)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.search.corpus_search import dedupe_kg_surface_rows, run_corpus_search
from podcast_scraper.search.faiss_store import VECTORS_FILE
from podcast_scraper.search.protocol import IndexStats, SearchResult

pytestmark = [pytest.mark.unit]


def test_run_corpus_search_empty_query(tmp_path: Path) -> None:
    out = run_corpus_search(tmp_path, "   ")
    assert out.error == "empty_query"


def test_run_corpus_search_no_index(tmp_path: Path) -> None:
    out = run_corpus_search(tmp_path, "climate")
    assert out.error == "no_index"
    assert "search" in (out.detail or "")


def test_dedupe_kg_surface_rows_merges_same_surface_text() -> None:
    rows = [
        {
            "doc_id": "a",
            "score": 0.9,
            "metadata": {"doc_type": "kg_entity", "episode_id": "ep1"},
            "text": "Acme Corp",
        },
        {
            "doc_id": "b",
            "score": 0.7,
            "metadata": {"doc_type": "kg_entity", "episode_id": "ep2"},
            "text": "acme  corp",
        },
        {"doc_id": "c", "score": 0.5, "metadata": {"doc_type": "insight"}, "text": "x"},
    ]
    out = dedupe_kg_surface_rows(rows)
    assert len(out) == 2
    assert out[0]["doc_id"] == "a"
    wmeta = out[0]["metadata"]
    assert wmeta.get("kg_surface_match_count") == 2
    assert "ep1" in wmeta.get("kg_surface_episode_ids", [])
    assert "ep2" in wmeta.get("kg_surface_episode_ids", [])


def test_dedupe_kg_surface_rows_non_dict_metadata_still_keeps_row() -> None:
    rows = [
        {"doc_id": "x", "score": 1.0, "metadata": "bad", "text": "t"},
    ]
    out = dedupe_kg_surface_rows(rows)
    assert len(out) == 1


@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
@patch("podcast_scraper.search.corpus_search.FaissVectorStore.load")
def test_run_corpus_search_success_mocked(
    mock_load: MagicMock,
    mock_encode: MagicMock,
    tmp_path: Path,
) -> None:
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / VECTORS_FILE).write_bytes(b"")

    mock_encode.return_value = [0.1] * 8

    store = MagicMock()
    store.ntotal = 2
    store.stats.return_value = IndexStats(
        total_vectors=2,
        doc_type_counts={"insight": 2},
        feeds_indexed=[],
        embedding_model="test-model",
        embedding_dim=8,
        last_updated="",
        index_size_bytes=0,
    )
    store.search.return_value = [
        SearchResult(
            doc_id="d1",
            score=0.99,
            metadata={"doc_type": "insight", "episode_id": "e1", "text": "hello"},
        ),
    ]
    mock_load.return_value = store

    out = run_corpus_search(tmp_path, "q", top_k=5, dedupe_kg_surfaces=False)
    assert out.error is None
    assert len(out.results) == 1
    assert out.results[0]["doc_id"] == "d1"
    assert out.lift_stats is not None
    assert "transcript_hits_returned" in out.lift_stats


@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
@patch("podcast_scraper.search.corpus_search.FaissVectorStore.load")
def test_run_corpus_search_load_failed(
    mock_load: MagicMock,
    mock_encode: MagicMock,
    tmp_path: Path,
) -> None:
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / VECTORS_FILE).write_bytes(b"")
    mock_load.side_effect = RuntimeError("bad index")
    out = run_corpus_search(tmp_path, "q")
    assert out.error == "load_failed"
    mock_encode.assert_not_called()


@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
@patch("podcast_scraper.search.corpus_search.FaissVectorStore.load")
def test_run_corpus_search_embed_failed(
    mock_load: MagicMock,
    mock_encode: MagicMock,
    tmp_path: Path,
) -> None:
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / VECTORS_FILE).write_bytes(b"")
    store = MagicMock()
    store.ntotal = 1
    store.stats.return_value = IndexStats(
        total_vectors=1,
        doc_type_counts={},
        feeds_indexed=[],
        embedding_model="m",
        embedding_dim=4,
        last_updated="",
        index_size_bytes=0,
    )
    mock_load.return_value = store
    mock_encode.side_effect = RuntimeError("no embedder")
    out = run_corpus_search(tmp_path, "q")
    assert out.error == "embed_failed"


@patch("podcast_scraper.search.corpus_search.embedding_loader.encode")
@patch("podcast_scraper.search.corpus_search.FaissVectorStore.load")
def test_run_corpus_search_multi_doc_type_filter(
    mock_load: MagicMock,
    mock_encode: MagicMock,
    tmp_path: Path,
) -> None:
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / VECTORS_FILE).write_bytes(b"")
    mock_encode.return_value = [0.0] * 4
    store = MagicMock()
    store.ntotal = 10
    store.stats.return_value = IndexStats(
        total_vectors=10,
        doc_type_counts={},
        feeds_indexed=[],
        embedding_model="m",
        embedding_dim=4,
        last_updated="",
        index_size_bytes=0,
    )
    store.search.return_value = [
        SearchResult("a", 0.9, {"doc_type": "insight", "episode_id": "e"}),
        SearchResult("b", 0.8, {"doc_type": "quote", "episode_id": "e"}),
        SearchResult("c", 0.7, {"doc_type": "summary", "episode_id": "e"}),
    ]
    mock_load.return_value = store
    out = run_corpus_search(
        tmp_path,
        "q",
        doc_types=["insight", "quote"],
        top_k=5,
        dedupe_kg_surfaces=False,
    )
    assert out.error is None
    dts = {r["metadata"]["doc_type"] for r in out.results}
    assert dts <= {"insight", "quote"}
    assert "summary" not in dts


def test_run_corpus_search_embed_returns_wrong_shape(
    tmp_path: Path,
) -> None:
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / VECTORS_FILE).write_bytes(b"")
    store = MagicMock()
    store.ntotal = 1
    store.stats.return_value = IndexStats(
        total_vectors=1,
        doc_type_counts={},
        feeds_indexed=[],
        embedding_model="m",
        embedding_dim=4,
        last_updated="",
        index_size_bytes=0,
    )
    with (
        patch(
            "podcast_scraper.search.corpus_search.FaissVectorStore.load",
            return_value=store,
        ),
        patch(
            "podcast_scraper.search.corpus_search.embedding_loader.encode",
            return_value=[[0.1, 0.2]],
        ),
    ):
        out = run_corpus_search(tmp_path, "q")
    assert out.error == "embed_failed"
