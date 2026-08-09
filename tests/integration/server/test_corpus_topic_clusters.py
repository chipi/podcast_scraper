"""Integration tests for GET /api/corpus/topic-clusters.

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


# --- topic-clusters rebuild: fold into /api/index/rebuild + dedicated endpoint (task-#14) ---


def test_index_rebuild_folds_topic_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/index/rebuild now also (re)builds topic_clusters.json — no CLI/SSH step."""
    from typing import cast

    from podcast_scraper.server.index_rebuild import CorpusRebuildGate
    from podcast_scraper.server.routes import index_rebuild as ir

    calls: dict = {}
    monkeypatch.setattr(ir, "_minimal_vector_config", lambda *a, **k: object())
    monkeypatch.setattr(ir, "index_corpus", lambda *a, **k: None)
    monkeypatch.setattr(ir, "invalidate_newest_index_source_mtime_cache", lambda *a, **k: None)
    import podcast_scraper.search.topic_clusters as tc

    monkeypatch.setattr(
        tc,
        "build_topic_clusters_for_corpus",
        lambda output_dir, **k: calls.update(threshold=k.get("threshold")),
    )

    class _Gate:
        def end(self, err: object) -> None:
            calls["ended"] = err

    ir._spawn_rebuild_thread(
        str(tmp_path),
        str(tmp_path),
        rebuild=False,
        vector_index_path=None,
        vector_embedding_model=None,
        vector_index_types=None,
        topic_cluster_threshold=0.75,
        gate=cast(CorpusRebuildGate, _Gate()),
    )
    assert calls["threshold"] == 0.75  # clusters built after the index (the fold)
    assert calls["ended"] is None  # clean run


def test_topic_clusters_rebuild_requires_viewer(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.post("/api/corpus/topic-clusters/rebuild", params={"path": str(tmp_path)})
    assert r.status_code in (401, 403)


def test_topic_clusters_rebuild_202_with_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("lancedb")
    from podcast_scraper.server.routes.app_auth import require_viewer_access

    app = create_app(tmp_path, static_dir=False)
    app.dependency_overrides[require_viewer_access] = lambda: object()  # fake viewer session
    import podcast_scraper.search.topic_clusters as tc

    monkeypatch.setattr(tc, "build_topic_clusters_for_corpus", lambda *a, **k: {"clusters": []})
    client = TestClient(app)
    r = client.post("/api/corpus/topic-clusters/rebuild", params={"path": str(tmp_path)})
    assert r.status_code == 202
    assert r.json()["accepted"] is True
