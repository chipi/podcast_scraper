"""M3 viewer API: GET /api/index/stats (RFC-062). Skipped when ``fastapi`` is missing."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app


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


def test_index_stats_available_with_minimal_faiss_index(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("numpy")

    from podcast_scraper.search.faiss_store import (
        FaissVectorStore,
        METADATA_FILE,
        VECTORS_FILE,
    )

    index_dir = tmp_path / "search"
    index_dir.mkdir(parents=True)
    store = FaissVectorStore(384, embedding_model="test-model", index_dir=index_dir)
    store.batch_upsert(
        ["a"],
        [[0.1] * 384],
        [{"doc_type": "insight", "feed_id": "f1", "text": "x"}],
    )
    store.persist(index_dir)

    assert (index_dir / VECTORS_FILE).is_file()
    assert (index_dir / METADATA_FILE).is_file()

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/index/stats", params={"path": str(tmp_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    stats = body["stats"]
    assert stats["total_vectors"] == 1
    assert stats["doc_type_counts"]["insight"] == 1
    assert "f1" in stats["feeds_indexed"]
