"""GET /api/ops/summary — wraps the podcast_obs control-plane summary."""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from podcast_scraper.server.routes import ops


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(ops.router, prefix="/api")
    return TestClient(app)


def _clear_obs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("PODCAST_OBS_"):
            monkeypatch.delenv(key, raising=False)


def test_ops_summary_returns_control_plane_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_obs_env(monkeypatch)
    fake = {
        "ok": True,
        "source": "summary",
        "data": {
            "target": "default",
            "live": ["health"],
            "unconfigured": ["cost"],
            "failed": [],
            "sources": {
                "health": {"ok": True, "source": "prod_api.health", "data": {"status": "ok"}}
            },
        },
    }
    monkeypatch.setattr("podcast_obs.aggregate.summary", lambda target: fake)
    resp = _client().get("/api/ops/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert {"target", "live", "unconfigured", "failed", "sources"} <= body.keys()
    assert body["live"] == ["health"]


def test_ops_summary_defaults_target_to_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_obs_env(monkeypatch)
    captured: dict = {}

    def fake_summary(target):
        captured["api_base"] = target.api_base
        captured["timeout"] = target.timeout
        return {
            "data": {
                "target": target.name,
                "live": [],
                "unconfigured": [],
                "failed": [],
                "sources": {},
            }
        }

    monkeypatch.setattr("podcast_obs.aggregate.summary", fake_summary)
    resp = _client().get("/api/ops/summary")
    assert resp.status_code == 200
    assert captured["api_base"] == "http://127.0.0.1:8000"  # defaulted to this server
    assert captured["timeout"] == 6.0  # responsive web timeout override
