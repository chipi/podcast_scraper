"""github / grafana / sentry sources: not-configured + success parsing."""

from __future__ import annotations

import pytest

from podcast_obs.config import TargetConfig
from podcast_obs.sources import github, grafana, langfuse, sentry


def _t(**kw) -> TargetConfig:
    return TargetConfig(name="t", **kw)


# --- github.recent_deploys ---------------------------------------------------------


def _only(payload: dict, workflow: str = "deploy-prod.yml"):
    """A get_json fake: *payload* for *workflow*'s runs, no runs for every other workflow."""
    return lambda url, **_: payload if f"/workflows/{workflow}/" in url else {"workflow_runs": []}


def test_deploys_not_configured() -> None:
    assert github.recent_deploys(_t())["configured"] is False


def test_deploys_cover_every_prod_deploy_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2026-10-15: the source read only deploy-prod.yml (control plane), so the one-sha
    # deploy-all-prod and deploy-config runs that day were invisible ("last deploy 09-29").
    runs = {
        "deploy-prod.yml": [{"run_number": 60, "created_at": "2026-09-29T19:12:48Z"}],
        "deploy-all-prod.yml": [
            {"run_number": 7, "conclusion": "failure", "created_at": "2026-10-15T05:42:52Z"}
        ],
        "deploy-config.yml": [
            {"run_number": 30, "conclusion": "success", "created_at": "2026-10-15T05:56:17Z"}
        ],
    }

    def fake(url: str, **_):
        wf = url.split("/workflows/")[1].split("/")[0]
        return {"workflow_runs": runs.get(wf, [])}

    monkeypatch.setattr(github, "get_json", fake)
    data = github.recent_deploys(_t(github_token="x"), limit=10)["data"]
    assert [(d["workflow"], d["run_number"]) for d in data["deploys"]] == [
        ("deploy-config.yml", 30),
        ("deploy-all-prod.yml", 7),
        ("deploy-prod.yml", 60),
    ]  # newest first, across workflows


def test_deploys_limit_applies_after_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(url: str, **_):
        wf = url.split("/workflows/")[1].split("/")[0]
        return {
            "workflow_runs": [{"run_number": 1, "created_at": f"2026-10-0{len(wf) % 9}T00:00:00Z"}]
        }

    monkeypatch.setattr(github, "get_json", fake)
    data = github.recent_deploys(_t(github_token="x"), limit=2)["data"]
    assert data["count"] == 2


def test_deploys_one_workflow_failing_does_not_hide_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(url: str, **_):
        if "/workflows/deploy-player.yml/" in url:
            raise RuntimeError("boom")
        if "/workflows/deploy-prod.yml/" in url:
            return {"workflow_runs": [{"run_number": 1, "created_at": "2026-10-01T00:00:00Z"}]}
        return {"workflow_runs": []}

    monkeypatch.setattr(github, "get_json", fake)
    result = github.recent_deploys(_t(github_token="x"))
    assert result["ok"] is True
    assert result["data"]["count"] == 1
    assert "deploy-player.yml" in result["data"]["errors"][0]


def test_deploys_all_workflows_failing_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, **_):
        raise RuntimeError("down")

    monkeypatch.setattr(github, "get_json", boom)
    assert github.recent_deploys(_t(github_token="x"))["ok"] is False


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
    monkeypatch.setattr(github, "get_json", _only(payload))
    data = github.recent_deploys(_t(github_token="x"), limit=10)["data"]
    assert data["count"] == 2
    assert data["deploys"][0]["sha"] == "abcdef1"  # truncated to 7
    assert data["deploys"][0]["duration_s"] == 120.0
    assert data["failure_rate"] == 0.5  # 1 of 2 concluded failed


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
    issue = data["projects"][0]["issues"][0]
    assert issue["title"] == "KeyError: x"
    # the pivot handle: each issue carries its GlitchTip UI permalink so an agent can click through.
    assert issue["permalink"] == "https://sentry/x"


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
    monkeypatch.setattr(github, "get_json", _only(payload))
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
    monkeypatch.setattr(github, "get_json", _only(payload))
    assert github.recent_deploys(_t(github_token="x"))["data"]["deploys"][0]["duration_s"] is None
