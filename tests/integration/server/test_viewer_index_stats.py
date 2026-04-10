"""Viewer API: GET /api/index/stats (RFC-062).

Requires ``fastapi`` (``pip install -e '.[server]'``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper import config_constants
from podcast_scraper.search.protocol import IndexStats
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def test_index_stats_no_corpus_when_no_path_and_no_state(tmp_path: Path) -> None:
    app = create_app(None, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/index/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["reason"] == "no_corpus_path"


def test_index_stats_no_index_dir(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/index/stats", params={"path": str(tmp_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["reason"] == "no_index"
    assert "search" in (body.get("index_path") or "")
    assert "reindex_recommended" in body
    assert body["reindex_recommended"] is False
    assert body["reindex_reasons"] == []
    assert body.get("artifact_newest_mtime") is None
    assert body["search_root_hints"] == []
    assert body.get("rebuild_in_progress") is False
    assert body.get("rebuild_last_error") is None


def test_index_stats_uses_app_state_output_dir(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/index/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["reason"] == "no_index"


def test_index_stats_rejects_bad_path(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/index/stats",
        params={"path": str(tmp_path / "nonexistent")},
    )
    assert response.status_code == 400


def test_index_stats_load_failed_when_faiss_store_load_raises(tmp_path: Path) -> None:
    """Route maps ``FaissVectorStore.load`` failures to ``reason=load_failed`` (no real FAISS)."""
    index_dir = tmp_path / "search"
    index_dir.mkdir(parents=True)
    (index_dir / "vectors.faiss").write_bytes(b"not-a-faiss-index")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    with patch(
        "podcast_scraper.search.faiss_store.FaissVectorStore.load",
        side_effect=ValueError("corrupt or unreadable index"),
    ):
        response = client.get("/api/index/stats", params={"path": str(tmp_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["reason"] == "load_failed"


def test_index_stats_no_index_recommends_when_metadata_exists(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    (meta / "a.metadata.json").write_text(
        '{"episode": {"episode_id": "e1"}, "feed": {}}',
        encoding="utf-8",
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/index/stats", params={"path": str(tmp_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["reason"] == "no_index"
    assert body["reindex_recommended"] is True
    assert "no_index_but_metadata" in body["reindex_reasons"]
    assert body.get("artifact_newest_mtime")


def test_index_stats_available_with_mocked_vector_store(tmp_path: Path) -> None:
    """Happy path uses ``FaissVectorStore.load`` + ``stats()``; mock store — no FAISS build."""
    index_dir = tmp_path / "search"
    index_dir.mkdir(parents=True)
    (index_dir / "vectors.faiss").touch()

    fake_stats = IndexStats(
        total_vectors=1,
        doc_type_counts={"insight": 1},
        feeds_indexed=["f1"],
        embedding_model=config_constants.DEFAULT_EMBEDDING_MODEL,
        embedding_dim=384,
        last_updated="2024-01-01T00:00:00Z",
        index_size_bytes=100,
    )
    fake_store = MagicMock()
    fake_store.stats.return_value = fake_stats

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    with patch(
        "podcast_scraper.search.faiss_store.FaissVectorStore.load",
        return_value=fake_store,
    ):
        response = client.get("/api/index/stats", params={"path": str(tmp_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    stats = body["stats"]
    assert stats["total_vectors"] == 1
    assert stats["doc_type_counts"]["insight"] == 1
    assert "f1" in stats["feeds_indexed"]
    assert body["reindex_recommended"] is False
    assert isinstance(body["reindex_reasons"], list)
    assert body["search_root_hints"] == []
    assert body.get("rebuild_in_progress") is False
    assert body.get("rebuild_last_error") is None


def test_index_stats_feeds_indexed_normalized_and_deduped(tmp_path: Path) -> None:
    """``feeds_indexed`` matches catalog-style ids: strip, dedupe, sorted."""
    index_dir = tmp_path / "search"
    index_dir.mkdir(parents=True)
    (index_dir / "vectors.faiss").touch()

    fake_stats = IndexStats(
        total_vectors=2,
        doc_type_counts={"insight": 2},
        feeds_indexed=["  f1  ", "f1", "f2"],
        embedding_model=config_constants.DEFAULT_EMBEDDING_MODEL,
        embedding_dim=384,
        last_updated="2024-01-01T00:00:00Z",
        index_size_bytes=100,
    )
    fake_store = MagicMock()
    fake_store.stats.return_value = fake_stats

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    with patch(
        "podcast_scraper.search.faiss_store.FaissVectorStore.load",
        return_value=fake_store,
    ):
        response = client.get("/api/index/stats", params={"path": str(tmp_path)})
    assert response.status_code == 200
    stats = response.json()["stats"]
    assert stats["feeds_indexed"] == ["f1", "f2"]
