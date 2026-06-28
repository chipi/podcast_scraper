"""summary aggregation buckets + CLI exit codes."""

from __future__ import annotations

import pytest

from podcast_obs import aggregate, cli, mcp_server
from podcast_obs.config import TargetConfig
from podcast_obs.sources import loki, prod_api


def _clear_obs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("PODCAST_OBS_"):
            monkeypatch.delenv(key, raising=False)


def test_summary_all_unconfigured_when_no_api_base() -> None:
    result = aggregate.summary(TargetConfig(name="t"))
    data = result["data"]
    assert result["ok"] is True
    assert data["live"] == []
    assert set(data["unconfigured"]) == {
        "health",
        "version",
        "runs",
        "deploys",
        "cost",
        "logs",
        "errors",
        "alerts",
        "traces",
        # RFC-088 surface (also needs api_base)
        "enrichment_status",
        "enrichment_health",
        "enrichment_events",
    }
    assert data["failed"] == []


def test_summary_prod_api_live_externals_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    # api_base set (prod_api + enrichment live) but no external tokens -> externals
    # report not-configured. The fake patch covers both prod_api and enrichment since
    # they're the only get_json call sites that hit the deploy's api_base.
    from podcast_obs.sources import enrichment

    payload = {"status": "ok", "code_version": "1", "jobs": []}
    monkeypatch.setattr(prod_api, "get_json", lambda url, **_: payload)
    monkeypatch.setattr(enrichment, "get_json", lambda url, **_: payload)
    data = aggregate.summary(TargetConfig(name="t", api_base="http://x"))["data"]
    assert set(data["live"]) == {
        "health",
        "version",
        "runs",
        "enrichment_status",
        "enrichment_health",
        "enrichment_events",
    }
    assert set(data["unconfigured"]) == {"deploys", "cost", "logs", "errors", "alerts", "traces"}
    assert data["failed"] == []


def test_cli_health_unreachable_returns_1(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _clear_obs_env(monkeypatch)
    monkeypatch.setenv("PODCAST_OBS_API_BASE", "http://localhost:9")
    monkeypatch.setattr(
        prod_api, "get_json", lambda url, **_: (_ for _ in ()).throw(RuntimeError("refused"))
    )
    rc = cli.main(["health"])
    assert rc == 1
    assert '"ok": false' in capsys.readouterr().out


def test_cli_summary_reachable_returns_0(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    _clear_obs_env(monkeypatch)
    monkeypatch.setenv("PODCAST_OBS_API_BASE", "http://x")
    monkeypatch.setattr(
        prod_api, "get_json", lambda url, **_: {"status": "ok", "code_version": "1", "jobs": []}
    )
    rc = cli.main(["summary"])
    assert rc == 0
    assert '"summary"' in capsys.readouterr().out


def test_cli_config_error_returns_2(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_obs_env(monkeypatch)
    monkeypatch.setenv("PODCAST_OBS_TARGET", "only")
    # ask for a target that doesn't exist in the single-target env config
    rc = cli.main(["--target", "nope", "health"])
    assert rc == 2


def test_summary_failed_bucket_from_raising_and_err(monkeypatch: pytest.MonkeyPatch) -> None:
    def raises(_t):
        raise RuntimeError("kaboom")  # a probe that RAISES (not returns err) -> safety-net except

    def errs(_t):
        return {"ok": False, "source": "x", "configured": True, "error": "down"}

    def lives(_t):
        return {"ok": True, "source": "y", "data": {}}

    monkeypatch.setattr(aggregate, "_PROBES", [("boom", raises), ("down", errs), ("up", lives)])
    data = aggregate.summary(TargetConfig(name="t"))["data"]
    assert set(data["failed"]) == {"boom", "down"}  # raising + configured-error both fail
    assert data["live"] == ["up"]
    assert data["unconfigured"] == []


def test_cli_runs_passes_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_obs_env(monkeypatch)
    monkeypatch.setenv("PODCAST_OBS_API_BASE", "http://x")
    captured: dict = {}
    monkeypatch.setattr(
        prod_api,
        "recent_pipeline_runs",
        lambda t, limit=10: captured.update(limit=limit) or {"ok": True, "source": "r", "data": {}},
    )
    cli.main(["runs", "--limit", "3"])
    assert captured["limit"] == 3


def test_cli_logs_passes_args(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_obs_env(monkeypatch)
    monkeypatch.setenv("PODCAST_OBS_API_BASE", "http://x")
    captured: dict = {}
    monkeypatch.setattr(
        loki,
        "recent_logs",
        lambda t, **kw: captured.update(kw) or {"ok": True, "source": "logs", "data": {}},
    )
    cli.main(["logs", "--service", "api", "--contains", "OOM", "--window", "6h"])
    assert captured["service"] == "api"
    assert captured["contains"] == "OOM"
    assert captured["window"] == "6h"


def test_cli_serve_maps_http_to_streamable(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_obs_env(monkeypatch)
    monkeypatch.setenv("PODCAST_OBS_API_BASE", "http://x")
    captured: dict = {}
    monkeypatch.setattr(mcp_server, "run_server", lambda config, **kw: captured.update(kw))
    rc = cli.main(["serve", "--transport", "http", "--port", "9000"])
    assert rc == 0
    assert captured["transport"] == "streamable-http"  # http -> streamable-http mapping
    assert captured["port"] == 9000
