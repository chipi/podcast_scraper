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


def test_cache_stats_not_configured() -> None:
    result = prod_api.cache_stats(_target())
    assert result["ok"] is False and result["configured"] is False


def test_cache_stats_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"namespaces": {"app_catalog_rows": {"hits": 40, "misses": 1, "hit_rate_pct": 97.6}}}
    seen: dict = {}

    def _get(url, **_):
        seen["url"] = url
        return payload

    monkeypatch.setattr(prod_api, "get_json", _get)
    result = prod_api.cache_stats(_target(api_base="http://x"))
    assert result["ok"] is True
    assert seen["url"].endswith("/api/ops/cache-stats")
    assert result["data"]["namespaces"]["app_catalog_rows"]["hit_rate_pct"] == 97.6


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


# --- operator-gated probes must carry X-Operator-Key -------------------------------
# ``/api/jobs`` and ``/api/ops/*`` moved under ``app_operator_guard`` (#1071/#1128) while these
# probes still called them bare, so ``ops summary`` reported runs/cache_stats as *failed* against
# a healthy prod. Regression guard: the header is present when a key is configured, absent when
# not, and never a literal ``None``.


def _captured_headers(monkeypatch: pytest.MonkeyPatch, fn, target) -> dict:
    seen: dict = {}

    def spy(url, **kw):
        seen.update(kw.get("headers") or {})
        return {"jobs": [], "path": "/app/output"}

    monkeypatch.setattr(prod_api, "get_json", spy)
    fn(target)
    return seen


def test_runs_sends_operator_key(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = _captured_headers(
        monkeypatch,
        prod_api.recent_pipeline_runs,
        _target(api_base="http://x", operator_key="secret-key"),
    )
    assert headers["X-Operator-Key"] == "secret-key"


def test_runs_omits_header_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = _captured_headers(
        monkeypatch, prod_api.recent_pipeline_runs, _target(api_base="http://x")
    )
    assert "X-Operator-Key" not in headers


def test_cache_stats_sends_operator_key(monkeypatch: pytest.MonkeyPatch) -> None:
    headers = _captured_headers(
        monkeypatch,
        prod_api.cache_stats,
        _target(api_base="http://x", operator_key="secret-key"),
    )
    assert headers["X-Operator-Key"] == "secret-key"
