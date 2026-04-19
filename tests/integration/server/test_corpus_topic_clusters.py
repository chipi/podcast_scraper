"""Integration tests for GET /api/corpus/topic-clusters.

Requires ``fastapi`` (``pip install -e '.[server]'``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


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
    client = TestClient(app)
    r = client.get("/api/corpus/topic-clusters")
    assert r.status_code == 200
    body = r.json()
    assert body.get("threshold") == 0.75
    assert body.get("schema_version") == "2"


def test_topic_clusters_404_when_missing(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/topic-clusters", params={"path": str(tmp_path)})
    assert r.status_code == 404
    body = r.json()
    assert body.get("available") is False


def test_topic_clusters_200_returns_json(tmp_path: Path) -> None:
    search = tmp_path / "search"
    search.mkdir()
    payload = {"schema_version": "1", "clusters": [], "topic_count": 0}
    (search / "topic_clusters.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/topic-clusters", params={"path": str(tmp_path)})
    assert r.status_code == 200
    assert r.json() == payload


def test_topic_clusters_200_returns_schema_v2_payload(tmp_path: Path) -> None:
    search = tmp_path / "search"
    search.mkdir()
    payload = {
        "schema_version": "2",
        "clusters": [
            {
                "graph_compound_parent_id": "tc:x",
                "cil_alias_target_topic_id": "topic:y",
                "canonical_label": "Y",
                "member_count": 1,
                "members": [{"topic_id": "topic:y"}],
            }
        ],
        "topic_count": 1,
        "cluster_count": 1,
        "singletons": 0,
    }
    (search / "topic_clusters.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/topic-clusters", params={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "2"
    assert body["clusters"][0]["graph_compound_parent_id"] == "tc:x"
    assert body["clusters"][0]["cil_alias_target_topic_id"] == "topic:y"
