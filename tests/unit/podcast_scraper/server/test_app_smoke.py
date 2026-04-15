"""Lightweight FastAPI app factory and route smoke tests (unit layer)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app, create_app_for_uvicorn

pytestmark = pytest.mark.unit


def test_create_app_for_uvicorn_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODCAST_SERVE_OUTPUT_DIR", raising=False)
    with pytest.raises(RuntimeError, match="PODCAST_SERVE_OUTPUT_DIR"):
        create_app_for_uvicorn()


def test_create_app_for_uvicorn_uses_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_SERVE_OUTPUT_DIR", str(tmp_path))
    app = create_app_for_uvicorn()
    assert app.state.output_dir == tmp_path.resolve()


def test_api_health_and_search_no_corpus(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    with TestClient(app) as client:
        h = client.get("/api/health")
        assert h.status_code == 200
        assert h.json().get("status") == "ok"
        s = client.get("/api/search?q=hello")
        assert s.status_code == 200
        body = s.json()
        assert body.get("query") == "hello"


def test_api_artifacts_lists_empty_corpus(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    with TestClient(app) as client:
        r = client.get("/api/artifacts", params={"path": str(tmp_path)})
        assert r.status_code == 200
        data = r.json()
        assert data.get("artifacts") == []
