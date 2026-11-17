"""GET /api/ops/summary — wraps the podcast_obs control-plane summary."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.server.routes import ops  # noqa: E402

# critical_path so this viewer API route runs in the PR fast subset
# (test-integration-fast = "integration and critical_path"), matching the
# other test_viewer_*/test_server_* route tests — gives /api/ops/summary
# PR-time regression + patch coverage.
pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def _client(base_url: str = "http://127.0.0.1:8000") -> TestClient:
    app = FastAPI()
    app.include_router(ops.router, prefix="/api")
    return TestClient(app, base_url=base_url)


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
    assert captured["timeout"] == 4.0  # responsive web timeout override


def test_ops_summary_probes_the_port_this_server_listens_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server started on another port (``make serve-api SERVE_PORT=8731``) probed 8000 and
    reported health, version and runs as "Connection refused" (public-only review, 2026-11-17)."""
    _clear_obs_env(monkeypatch)
    captured: dict = {}

    def fake_summary(target):
        captured["api_base"] = target.api_base
        return {"data": {"target": "", "live": [], "unconfigured": [], "failed": [], "sources": {}}}

    monkeypatch.setattr("podcast_obs.aggregate.summary", fake_summary)
    assert _client("http://127.0.0.1:8731").get("/api/ops/summary").status_code == 200
    assert captured["api_base"] == "http://127.0.0.1:8731"


def test_ops_summary_probes_this_server_as_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """``/api/jobs`` is behind the operator gate (an admin session or the operator key). Without an
    operator key the self-probe went anonymous and Ops showed ``runs`` as failed with a 403 to the
    admin looking at it (public-only walk, 2026-11-17). Probing itself, it now goes as the caller.
    """
    _clear_obs_env(monkeypatch)
    captured: dict = {}

    def fake_summary(target):
        captured["cookie"] = target.cookie
        return {"data": {"target": "", "live": [], "unconfigured": [], "failed": [], "sources": {}}}

    monkeypatch.setattr("podcast_obs.aggregate.summary", fake_summary)
    client = _client()
    client.cookies.set("app_session", "s3ss10n")
    assert client.get("/api/ops/summary").status_code == 200
    assert captured["cookie"] == "app_session=s3ss10n"


def test_ops_summary_never_sends_the_caller_session_to_a_configured_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured ``api_base`` may be another deploy: the caller's session stays on this one."""
    _clear_obs_env(monkeypatch)
    monkeypatch.setenv("PODCAST_OBS_API_BASE", "https://elsewhere.example")
    captured: dict = {}

    def fake_summary(target):
        captured["cookie"] = target.cookie
        captured["api_base"] = target.api_base
        return {"data": {"target": "", "live": [], "unconfigured": [], "failed": [], "sources": {}}}

    monkeypatch.setattr("podcast_obs.aggregate.summary", fake_summary)
    client = _client()
    client.cookies.set("app_session", "s3ss10n")
    assert client.get("/api/ops/summary").status_code == 200
    assert captured["api_base"] == "https://elsewhere.example"
    assert captured["cookie"] is None


def test_operator_probes_send_the_caller_session_and_the_operator_key() -> None:
    """The probes send the session cookie the route hands them, beside the operator key."""
    from podcast_obs.config import TargetConfig
    from podcast_obs.sources import prod_api

    assert prod_api._auth(TargetConfig(name="t")) == {}
    assert prod_api._auth(TargetConfig(name="t", cookie="app_session=x")) == {
        "Cookie": "app_session=x"
    }
    assert prod_api._auth(TargetConfig(name="t", operator_key="k", cookie="a=b")) == {
        "X-Operator-Key": "k",
        "Cookie": "a=b",
    }


def test_ops_summary_real_fanout_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    # No mock of summary: the real route -> podcast_obs fan-out runs. api_base defaults to the
    # local server (nothing listening) -> prod_api probes fail; external sources unconfigured.
    # Validates the route never 500s/hangs and returns the partitioned buckets.
    _clear_obs_env(monkeypatch)
    resp = _client().get("/api/ops/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert {"target", "live", "unconfigured", "failed", "sources"} <= body.keys()
    assert "deploys" in body["unconfigured"]  # no github token -> not configured
    assert "health" not in body["unconfigured"]  # api_base defaulted -> configured (live or failed)
