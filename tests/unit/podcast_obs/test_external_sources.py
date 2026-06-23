"""github / loki / grafana / sentry sources: not-configured + success parsing."""

from __future__ import annotations

import pytest

from podcast_obs.config import TargetConfig
from podcast_obs.sources import github, grafana, langfuse, loki, sentry


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


# --- langfuse.recent_traces --------------------------------------------------------


def test_traces_not_configured_without_keys() -> None:
    assert langfuse.recent_traces(_t())["configured"] is False
    # public key alone isn't enough — the Basic-auth pair needs both.
    assert langfuse.recent_traces(_t(langfuse_public_key="pk"))["configured"] is False


def test_traces_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": [
            {
                "id": "t1",
                "name": "summarization:claude",
                "timestamp": "2026-06-22T00:00:00Z",
                "latency": 1.2,
                "totalCost": 0.0151,
            },
            {"id": "t2", "name": "gi:gpt", "timestamp": "2026-06-22T00:01:00Z"},
        ]
    }
    monkeypatch.setattr(langfuse, "get_json", lambda url, **_: payload)
    target = _t(langfuse_public_key="pk", langfuse_secret_key="sk")
    result = langfuse.recent_traces(target)
    assert result["ok"] is True
    data = result["data"]
    assert data["base_url"] == "https://cloud.langfuse.com"  # default when base unset
    assert data["count"] == 2
    assert data["traces"][0]["name"] == "summarization:claude"
    assert data["traces"][0]["totalCost"] == 0.0151


def test_traces_self_hosted_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(langfuse, "get_json", lambda url, **_: {"data": []})
    target = _t(
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        langfuse_base_url="http://langfuse.tailnet:3000/",
    )
    data = langfuse.recent_traces(target)["data"]
    assert data["base_url"] == "http://langfuse.tailnet:3000"  # trailing slash stripped
    assert data["count"] == 0


def test_traces_error_is_failed_not_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, **_):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(langfuse, "get_json", boom)
    result = langfuse.recent_traces(_t(langfuse_public_key="pk", langfuse_secret_key="sk"))
    assert result["ok"] is False
    assert result.get("configured") is not False  # keys present → not an "unconfigured"


# --- loki helpers: window parsing, URL base, malformed payloads (regression guards) -----


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        ("30m", 1800),
        ("2h", 7200),
        ("1d", 86400),
        ("45s", 45),
        ("", 3600),
        ("abc", 3600),
        ("10x", 3600),
    ],
)
def test_parse_window_seconds(window: str, expected: int) -> None:
    assert loki._parse_window_seconds(window) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://x.net/loki/api/v1/push",
        "https://x.net/loki/api/v1/query",
        "https://x.net/loki/api/v1/query_range",
        "https://x.net/",
        "https://x.net",
    ],
)
def test_query_base_normalises_to_host(url: str) -> None:
    assert loki._query_base(url) == "https://x.net"


def test_query_base_none() -> None:
    assert loki._query_base(None) is None


def test_cost_malformed_payload_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loki, "get_json", lambda url, **_: {"unexpected": True})
    assert loki.cost_today(_loki_target())["data"]["estimated_cost_usd"] is None


def test_logs_malformed_payload_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loki, "get_json", lambda url, **_: {})
    assert loki.recent_logs(_loki_target())["data"]["lines"] == []


def test_logs_query_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        loki,
        "get_json",
        lambda url, **kw: captured.update(params=kw.get("params")) or {"data": {"result": []}},
    )
    loki.recent_logs(_loki_target(), level="error", service="api", contains='a"b\\c')
    query = captured["params"]["query"]
    assert 'service="api"' in query  # service label injected
    assert "(error|critical|exception|traceback|fatal)" in query  # error filter present
    assert r'|= "a\"b\\c"' in query  # quote AND backslash escaped (C2 regression guard)


def test_logs_level_all_skips_error_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        loki,
        "get_json",
        lambda url, **kw: captured.update(params=kw.get("params")) or {"data": {"result": []}},
    )
    loki.recent_logs(_loki_target(), level="all")
    assert "exception" not in captured["params"]["query"]  # no error filter when level != error


def test_deploys_duration_none_for_in_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "workflow_runs": [
            {
                "run_number": 9,
                "status": "in_progress",
                "conclusion": None,  # not concluded -> duration_s must be None
                "head_sha": "abc1234",
                "run_started_at": "2026-06-01T00:00:00Z",
                "updated_at": "2026-06-01T00:05:00Z",
            }
        ]
    }
    monkeypatch.setattr(github, "get_json", lambda url, **_: payload)
    data = github.recent_deploys(_t(github_token="x"))["data"]
    assert data["deploys"][0]["duration_s"] is None  # C3 regression guard
    assert data["failure_rate"] is None  # nothing concluded


def test_deploys_duration_bad_iso_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "workflow_runs": [
            {
                "run_number": 8,
                "conclusion": "success",
                "head_sha": "abc1234",
                "run_started_at": "garbage",
                "updated_at": "2026-06-01T00:05:00Z",
            }
        ]
    }
    monkeypatch.setattr(github, "get_json", lambda url, **_: payload)
    assert github.recent_deploys(_t(github_token="x"))["data"]["deploys"][0]["duration_s"] is None
