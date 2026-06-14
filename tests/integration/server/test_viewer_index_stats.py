"""Viewer API: GET /api/index/stats (LanceDB index stats; #995).

Requires ``fastapi`` (``pip install -e '.[dev]'``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper import config_constants
from podcast_scraper.search.lance_index_stats import LanceIndexStats
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]

_READ_STATS = "podcast_scraper.search.lance_index_stats.read_lance_index_stats"


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


def test_index_stats_available_with_mocked_index(tmp_path: Path) -> None:
    """Happy path reads ``read_lance_index_stats``; mock it — no real LanceDB build."""
    fake = LanceIndexStats(
        total_vectors=1,
        doc_type_counts={"insight": 1},
        feeds_indexed=["f1"],
        embedding_model=config_constants.DEFAULT_EMBEDDING_MODEL,
        embedding_dim=384,
        last_updated="2024-01-01T00:00:00Z",
        index_size_bytes=100,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    with patch(_READ_STATS, return_value=fake):
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
    fake = LanceIndexStats(
        total_vectors=2,
        doc_type_counts={"insight": 2},
        feeds_indexed=["  f1  ", "f1", "f2"],
        embedding_model=config_constants.DEFAULT_EMBEDDING_MODEL,
        embedding_dim=384,
        last_updated="2024-01-01T00:00:00Z",
        index_size_bytes=100,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    with patch(_READ_STATS, return_value=fake):
        response = client.get("/api/index/stats", params={"path": str(tmp_path)})
    assert response.status_code == 200
    stats = response.json()["stats"]
    assert stats["feeds_indexed"] == ["f1", "f2"]
