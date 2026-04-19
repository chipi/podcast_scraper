"""Viewer API: health + artifacts.

Requires ``fastapi`` (``pip install -e '.[server]'``).
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.faiss_store import VECTORS_FILE
from podcast_scraper.server.app import create_app
from podcast_scraper.server.schemas import HealthResponse

pytestmark = [pytest.mark.integration]


def test_health_ok(tmp_path: Path) -> None:
    """Health JSON matches :class:`HealthResponse` defaults (full router wiring)."""
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = HealthResponse.model_validate(response.json())
    assert body == HealthResponse()


def test_list_artifacts_finds_gi_and_kg(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "ep1.gi.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    (meta / "ep1.kg.json").write_text(json.dumps({"b": 2}), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/artifacts", params={"path": str(tmp_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["path"] == str(tmp_path.resolve())
    names = {item["name"] for item in body["artifacts"]}
    assert names == {"ep1.gi.json", "ep1.kg.json"}
    kinds = {item["kind"] for item in body["artifacts"]}
    assert kinds == {"gi", "kg"}
    for item in body["artifacts"]:
        assert "mtime_utc" in item
        assert str(item["mtime_utc"]).endswith("Z")
    assert body.get("hints") == []


def test_list_artifacts_hints_when_index_at_corpus_parent(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "search").mkdir(parents=True)
    (corpus / "search" / VECTORS_FILE).write_bytes(b"")
    feed_meta = corpus / "feeds" / "rss_x" / "metadata"
    feed_meta.mkdir(parents=True)
    (feed_meta / "ep1.gi.json").write_text("{}", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/artifacts", params={"path": str(feed_meta)})
    assert response.status_code == 200
    body = response.json()
    assert body["artifacts"]
    hints = body.get("hints") or []
    assert len(hints) == 1
    assert str(corpus.resolve()) in hints[0]


def test_get_artifact_returns_json(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    payload = {"grounded_insights": {"version": "test"}}
    (meta / "ep1.gi.json").write_text(json.dumps(payload), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/artifacts/metadata/ep1.gi.json",
        params={"path": str(tmp_path)},
    )
    assert response.status_code == 200
    assert response.json() == payload


def test_get_artifact_rejects_traversal(tmp_path: Path) -> None:
    """Use percent-encoding so ``..`` is not stripped during URL normalization."""
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    malicious = "metadata/../../../outside.gi.json"
    encoded = quote(malicious, safe="")
    response = client.get(
        f"/api/artifacts/{encoded}",
        params={"path": str(tmp_path)},
    )
    assert response.status_code == 400


def test_get_artifact_404(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/artifacts/metadata/missing.gi.json",
        params={"path": str(tmp_path)},
    )
    assert response.status_code == 404


def test_get_artifact_rejects_invalid_json(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "bad.gi.json").write_text("{not json", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/artifacts/metadata/bad.gi.json",
        params={"path": str(tmp_path)},
    )
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]


def test_get_artifact_rejects_non_gi_kg_suffix(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "readme.txt").write_text("x", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/artifacts/metadata/readme.txt",
        params={"path": str(tmp_path)},
    )
    assert response.status_code == 400
