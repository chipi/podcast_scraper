"""prod_api probes: not-configured, success, transport error, parsing."""

from __future__ import annotations

import pytest

from podcast_obs.config import TargetConfig
from podcast_obs.sources import prod_api

_HEALTH = {
    "status": "ok",
    "code_version": "2.6.0",
    "corpus_code_version": "2.6.0",
    "corpus_produced_by": {"git_sha": "abc1234", "produced_at": "2026-06-01T00:00:00Z"},
    "corpus_version_warning": None,
}


def _target(**kw) -> TargetConfig:
    return TargetConfig(name="t", **kw)


def test_health_not_configured() -> None:
    result = prod_api.health(_target())
    assert result["ok"] is False
    assert result["configured"] is False


def test_health_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prod_api, "get_json", lambda url, **_: _HEALTH)
    result = prod_api.health(_target(api_base="http://x"))
    assert result["ok"] is True
    assert result["data"]["status"] == "ok"


def test_health_transport_error_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, **_):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(prod_api, "get_json", boom)
    result = prod_api.health(_target(api_base="http://x"))
    assert result["ok"] is False
    assert result["configured"] is True  # configured, just unreachable
    assert "connection refused" in result["error"]


def test_deployed_version_derives_from_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prod_api, "get_json", lambda url, **_: _HEALTH)
    data = prod_api.deployed_version(_target(api_base="http://x"))["data"]
    assert data["code_version"] == "2.6.0"
    assert data["corpus_git_sha"] == "abc1234"
    assert data["corpus_produced_at"] == "2026-06-01T00:00:00Z"


def test_deployed_version_propagates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prod_api, "get_json", lambda url, **_: (_ for _ in ()).throw(RuntimeError("x"))
    )
    result = prod_api.deployed_version(_target(api_base="http://x"))
    assert result["ok"] is False
    assert result["source"] == "prod_api.version"


def test_recent_runs_sorted_newest_first_and_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "path": "/corpus",
        "jobs": [
            {"job_id": "a", "status": "completed", "created_at": "2026-01-01T00:00:00Z"},
            {"job_id": "b", "status": "running", "created_at": "2026-03-01T00:00:00Z"},
            {"job_id": "c", "status": "completed", "created_at": "2026-02-01T00:00:00Z"},
        ],
    }
    monkeypatch.setattr(prod_api, "get_json", lambda url, **_: payload)
    result = prod_api.recent_pipeline_runs(_target(api_base="http://x"), limit=2)
    ids = [run["job_id"] for run in result["data"]["runs"]]
    assert ids == ["b", "c"]  # newest first, limited to 2
    assert result["data"]["count"] == 2
    assert result["data"]["path"] == "/corpus"


def test_recent_runs_not_configured() -> None:
    result = prod_api.recent_pipeline_runs(_target())
    assert result["ok"] is False
    assert result["configured"] is False
