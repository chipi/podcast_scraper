"""M1 viewer API: health + artifacts (RFC-062). Skipped when ``fastapi`` is not installed."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app


def test_health_ok(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
