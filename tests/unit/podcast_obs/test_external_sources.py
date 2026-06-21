"""github / loki / grafana / sentry sources: not-configured + success parsing."""

from __future__ import annotations

import pytest

from podcast_obs.config import TargetConfig
from podcast_obs.sources import github, grafana, loki, sentry


def _t(**kw) -> TargetConfig:
    return TargetConfig(name="t", **kw)


# --- github.recent_deploys ---------------------------------------------------------


def test_deploys_not_configured() -> None:
    assert github.recent_deploys(_t())["configured"] is False


def test_deploys_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "workflow_runs": [
            {
                "run_number": 5,
                "status": "completed",
                "conclusion": "success",
                "head_sha": "abcdef1234567",
                "actor": {"login": "chipi"},
                "event": "workflow_dispatch",
                "created_at": "2026-06-01T00:00:00Z",
                "run_started_at": "2026-06-01T00:00:00Z",
                "updated_at": "2026-06-01T00:02:00Z",
                "html_url": "https://gh/run/5",
            },
            {
                "run_number": 4,
                "status": "completed",
                "conclusion": "failure",
                "head_sha": "0000000",
            },
        ]
    }
    monkeypatch.setattr(github, "get_json", lambda url, **_: payload)
    data = github.recent_deploys(_t(github_token="x"), limit=10)["data"]
    assert data["count"] == 2
    assert data["deploys"][0]["sha"] == "abcdef1"  # truncated to 7
    assert data["deploys"][0]["duration_s"] == 120.0
    assert data["failure_rate"] == 0.5  # 1 of 2 concluded failed


# --- loki.cost_today / recent_logs -------------------------------------------------


def _loki_target() -> TargetConfig:
    return _t(
        loki_url="https://logs.example.net/loki/api/v1/push",
        loki_user="12345",
        grafana_token="tok",
    )


def test_cost_not_configured() -> None:
    assert loki.cost_today(_t())["configured"] is False


def test_loki_prefers_loki_token_over_grafana(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        loki,
        "get_json",
        lambda url, **kw: captured.update(auth=kw.get("auth")) or {"data": {"result": []}},
    )
    target = _t(
        loki_url="https://x.net/loki/api/v1/push",
        loki_user="u",
        loki_token="READ",
        grafana_token="WRITE",
    )
    loki.cost_today(target)
    assert captured["auth"] == (
        "u",
        "READ",
    )  # access-policy token wins over the service-account one


def test_loki_falls_back_to_grafana_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        loki,
        "get_json",
        lambda url, **kw: captured.update(auth=kw.get("auth")) or {"data": {"result": []}},
    )
    target = _t(loki_url="https://x.net/loki/api/v1/push", loki_user="u", grafana_token="ONLY")
    loki.cost_today(target)
    assert captured["auth"] == ("u", "ONLY")


def test_cost_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        loki, "get_json", lambda url, **_: {"data": {"result": [{"value": [0, "0.5142"]}]}}
    )
    data = loki.cost_today(_loki_target())["data"]
    assert data["estimated_cost_usd"] == 0.5142


def test_cost_empty_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loki, "get_json", lambda url, **_: {"data": {"result": []}})
    assert loki.cost_today(_loki_target())["data"]["estimated_cost_usd"] == 0.0


def test_logs_parsing_newest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": {
            "result": [
                {
                    "stream": {"service": "pipeline"},
                    "values": [
                        ["1717200000000000000", "ERROR older"],
                        ["1717200002000000000", "ERROR newer"],
                    ],
                }
            ]
        }
    }
    monkeypatch.setattr(loki, "get_json", lambda url, **_: payload)
    data = loki.recent_logs(_loki_target(), window="1h", limit=10)["data"]
    assert data["count"] == 2
    assert data["lines"][0]["line"] == "ERROR newer"  # newest first
    assert data["lines"][0]["service"] == "pipeline"


def test_loki_query_base_strips_push_suffix() -> None:
    assert loki._query_base("https://x.net/loki/api/v1/push") == "https://x.net"


# --- grafana.recent_alerts ---------------------------------------------------------


def test_alerts_not_configured() -> None:
    assert grafana.recent_alerts(_t())["configured"] is False


def test_alerts_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {
            "labels": {"alertname": "HighCost", "severity": "warning"},
            "status": {"state": "active"},
            "startsAt": "2026-06-01T00:00:00Z",
            "annotations": {"summary": "spend spiking"},
        }
    ]
    monkeypatch.setattr(grafana, "get_json", lambda url, **_: payload)
    data = grafana.recent_alerts(_t(grafana_url="https://g.net", grafana_token="tok"))["data"]
    assert data["count"] == 1
    assert data["firing"] == 1
    assert data["alerts"][0]["alertname"] == "HighCost"


# --- sentry.recent_errors ----------------------------------------------------------


def test_errors_not_configured_without_token() -> None:
    assert sentry.recent_errors(_t())["configured"] is False


def test_errors_not_configured_without_org() -> None:
    assert sentry.recent_errors(_t(sentry_token="x"))["configured"] is False


def test_errors_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    issues = [
        {
            "title": "KeyError: x",
            "culprit": "mod.fn",
            "level": "error",
            "count": "3",
            "lastSeen": "2026-06-01T00:00:00Z",
            "permalink": "https://sentry/x",
        }
    ]
    monkeypatch.setattr(sentry, "get_json", lambda url, **_: issues)
    target = _t(sentry_token="x", sentry_org="acme", sentry_projects=("api",))
    data = sentry.recent_errors(target)["data"]
    assert data["total_issues"] == 1
    assert data["projects"][0]["issues"][0]["title"] == "KeyError: x"


def test_errors_all_projects_failing_is_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, **_):
        raise RuntimeError("404 not found")

    monkeypatch.setattr(sentry, "get_json", boom)
    target = _t(sentry_token="x", sentry_org="acme", sentry_projects=("nope", "alsonope"))
    result = sentry.recent_errors(target)
    assert result["ok"] is False  # not a healthy zero — all projects failed
    assert result["configured"] is True
    assert "all configured" in result["error"]
