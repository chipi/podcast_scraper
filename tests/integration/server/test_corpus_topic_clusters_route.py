"""Integration tests for GET /api/corpus/topic-clusters (RFC-075)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def test_topic_clusters_uses_default_output_dir(tmp_path: Path) -> None:
    search = tmp_path / "search"
    search.mkdir()
    payload = {
        "schema_version": "2",
        "threshold": 0.75,
        "clusters": [],
        "topic_count": 0,
        "cluster_count": 0,
    }
    (search / "topic_clusters.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    app = create_app(tmp_path, static_dir=False)
    with TestClient(app) as client:
        r = client.get("/api/corpus/topic-clusters")
    assert r.status_code == 200
    body = r.json()
    assert body.get("threshold") == 0.75
    assert body.get("schema_version") == "2"
