"""summary aggregation buckets + CLI exit codes."""

from __future__ import annotations

import pytest

from podcast_obs import aggregate, cli
from podcast_obs.config import TargetConfig
from podcast_obs.sources import prod_api


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
    }
    assert data["failed"] == []


def test_summary_prod_api_live_externals_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    # api_base set (prod_api live) but no external tokens -> externals report not-configured.
    monkeypatch.setattr(
        prod_api, "get_json", lambda url, **_: {"status": "ok", "code_version": "1", "jobs": []}
    )
    data = aggregate.summary(TargetConfig(name="t", api_base="http://x"))["data"]
    assert set(data["live"]) == {"health", "version", "runs"}
    assert set(data["unconfigured"]) == {"deploys", "cost", "logs", "errors", "alerts"}
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
