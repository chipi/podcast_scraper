"""Integration tests for GET /api/corpus/theme-clusters.

Theme clusters (co-occurrence lift) are served from ``enrichments/`` — the
sibling of the semantic ``/api/corpus/topic-clusters`` endpoint.
Requires ``fastapi`` (``pip install -e '.[dev]'``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def test_theme_clusters_uses_default_output_dir(tmp_path: Path) -> None:
    enr = tmp_path / "enrichments"
    enr.mkdir()
    payload = {
        "schema_version": "1",
        "method": "cooccurrence_lift",
        "merge_threshold": 2.0,
        "clusters": [],
        "topic_count": 0,
        "cluster_count": 0,
    }
    (enr / "topic_theme_clusters.json").write_text(json.dumps(payload), encoding="utf-8")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/theme-clusters")
    assert r.status_code == 200
    body = r.json()
    assert body.get("method") == "cooccurrence_lift"
    assert body.get("merge_threshold") == 2.0


def test_theme_clusters_404_when_missing(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/theme-clusters", params={"path": str(tmp_path)})
    assert r.status_code == 404
    body = r.json()
    assert body.get("available") is False


def test_theme_clusters_200_returns_theme_payload(tmp_path: Path) -> None:
    enr = tmp_path / "enrichments"
    enr.mkdir()
    payload = {
        "schema_version": "1",
        "method": "cooccurrence_lift",
        "clusters": [
            {
                "cluster_type": "theme",
                "graph_compound_parent_id": "thc:shadow-fleet",
                "canonical_label": "shadow fleet",
                "member_count": 2,
                "members": [
                    {"topic_id": "topic:shadow-fleet", "label": "shadow fleet"},
                    {"topic_id": "topic:oil-prices", "label": "oil prices"},
                ],
            }
        ],
        "topic_count": 2,
        "cluster_count": 1,
        "singletons": 0,
    }
    (enr / "topic_theme_clusters.json").write_text(json.dumps(payload), encoding="utf-8")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/theme-clusters", params={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["clusters"][0]["cluster_type"] == "theme"
    assert body["clusters"][0]["graph_compound_parent_id"] == "thc:shadow-fleet"
