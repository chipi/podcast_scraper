"""Tests for the dev-only in-app observability push (logs + metrics).

The load-bearing property is **inert unless explicitly enabled** — the packaged image
never sets the push-URL env vars, so every entry point must be a true no-op there
(prod ships via Alloy). The rest verifies labels + that a POST fires when enabled.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest

pytestmark = pytest.mark.unit

_ENV_VARS = (
    "PODCAST_LOGS_PUSH_URL",
    "PODCAST_METRICS_PUSH_URL",
    "PODCAST_OBS_INSTANCE",
    "PODCAST_OBS_PORT",
    "PORT",
    "OTEL_SERVICE_NAME",
    "PODCAST_ENV",
    "PODCAST_WORKTREE",
)


@pytest.fixture
def dev_push(monkeypatch):
    """Fresh dev_push module with a clean env (no push vars) each test."""
    for v in _ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    import podcast_scraper.obs.dev_push as mod

    return importlib.reload(mod)


# --- the prod guarantee: inert unless enabled ---------------------------------------


def test_inert_by_default(dev_push):
    assert dev_push.logs_push_enabled() is False
    assert dev_push.metrics_push_enabled() is False


def test_push_event_is_noop_without_url(dev_push):
    with patch.object(dev_push, "_post") as post:
        dev_push.push_event({"event_type": "x"})
        dev_push._flush_logs()
    post.assert_not_called()
    # no background worker should have been spawned
    assert dev_push._log_worker is None


def test_metrics_noop_without_url(dev_push):
    with patch.object(dev_push, "_post") as post:
        dev_push.push_metrics_once()
        assert dev_push.start_metrics_pusher() is False
    post.assert_not_called()


# --- labels: environment + instance (multi-worktree identity) -----------------------


def test_instance_defaults_to_worktree_and_port(dev_push, monkeypatch):
    monkeypatch.setenv("PODCAST_OBS_PORT", "8000")
    with patch.object(dev_push.os.path, "basename", return_value="wtA"):
        assert dev_push._instance() == "wtA-8000"


def test_instance_explicit_override_wins(dev_push, monkeypatch):
    monkeypatch.setenv("PODCAST_OBS_INSTANCE", "custom-1")
    monkeypatch.setenv("PODCAST_OBS_PORT", "9999")
    assert dev_push._instance() == "custom-1"


def test_labels_carry_env_and_service(dev_push, monkeypatch):
    monkeypatch.setenv("PODCAST_ENV", "dev")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "api")
    monkeypatch.setenv("PODCAST_OBS_INSTANCE", "wtB-8001")
    monkeypatch.setenv("PODCAST_WORKTREE", "feat/x")
    labels = dev_push.obs_labels()
    assert labels == {
        "service": "api",
        "environment": "dev",
        "instance": "wtB-8001",
        "worktree": "feat/x",
    }


# --- a POST fires when enabled ------------------------------------------------------


def test_logs_push_posts_when_enabled(dev_push, monkeypatch):
    monkeypatch.setenv("PODCAST_LOGS_PUSH_URL", "http://backend:9428/insert/jsonline")
    monkeypatch.setenv("PODCAST_OBS_INSTANCE", "wtC-8002")
    with patch.object(dev_push, "_post") as post:
        dev_push.push_event({"event_type": "llm_cost", "cost_usd": 0.01})
        dev_push._flush_logs()
    assert post.called
    url, body = post.call_args.args[0], post.call_args.args[1]
    assert url.startswith("http://backend:9428/insert/jsonline?")
    assert "_stream_fields=service" in url
    sent = body.decode()
    assert '"event_type": "llm_cost"' in sent
    assert '"instance": "wtC-8002"' in sent  # label merged onto the record


def test_metrics_push_posts_with_extra_labels(dev_push, monkeypatch):
    monkeypatch.setenv("PODCAST_METRICS_PUSH_URL", "http://backend:8428/api/v1/import/prometheus")
    monkeypatch.setenv("PODCAST_OBS_INSTANCE", "wtD-8003")
    monkeypatch.setenv("PODCAST_ENV", "dev")
    with patch.object(dev_push, "_post") as post:
        dev_push.push_metrics_once()
    assert post.called
    url = post.call_args.args[0]
    labels = parse_qs(urlsplit(url).query)["extra_label"]
    assert "environment=dev" in labels
    assert "instance=wtD-8003" in labels


def test_emit_event_triggers_push_when_enabled(dev_push, monkeypatch):
    """The emit_event hook drives the push (no-op default; fires when the URL is set)."""
    monkeypatch.setenv("PODCAST_LOGS_PUSH_URL", "http://backend:9428/insert/jsonline")
    monkeypatch.setenv("PODCAST_OBS_INSTANCE", "wtE-8004")
    from podcast_scraper.obs import events

    with patch.object(dev_push, "_post") as post:
        events.emit_event("search_query", query_type="semantic")
        dev_push._flush_logs()
    assert post.called


def test_service_falls_back_to_the_prod_pipeline_name(dev_push, monkeypatch):
    # Same value prod uses (compose OTEL_SERVICE_NAME), so dev and prod join on one name.
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    assert dev_push.obs_labels()["service"] == "podcast-pipeline"


# --- the dev telemetry contract: dev signals name their git branch (2026-11-16) -------


def test_worktree_explicit_override_wins(dev_push, monkeypatch):
    monkeypatch.setenv("PODCAST_WORKTREE", "feat/x")
    assert dev_push.worktree() == "feat/x"


def test_worktree_is_the_checkouts_branch(dev_push):
    done = dev_push.subprocess.CompletedProcess([], 0, stdout="feat/prompt-and-cleaning-seams\n")
    with patch.object(dev_push.subprocess, "run", return_value=done):
        assert dev_push.worktree() == "feat/prompt-and-cleaning-seams"


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(("HEAD\n", 0), id="detached-head"),
        pytest.param(("", 128), id="not-a-git-checkout"),
    ],
)
def test_no_branch_is_no_worktree(dev_push, result):
    out, code = result
    done = dev_push.subprocess.CompletedProcess([], code, stdout=out)
    with patch.object(dev_push.subprocess, "run", return_value=done):
        assert dev_push.worktree() is None


def test_dev_labels_carry_the_branch_and_prod_labels_never_do(dev_push, monkeypatch):
    monkeypatch.setenv("PODCAST_WORKTREE", "feat/x")
    assert dev_push.obs_labels()["worktree"] == "feat/x"
    monkeypatch.setenv("PODCAST_ENV", "prod")
    assert "worktree" not in dev_push.obs_labels()


def test_logs_stream_and_metric_labels_carry_the_branch(dev_push, monkeypatch):
    monkeypatch.setenv("PODCAST_WORKTREE", "feat/a&b#c")
    monkeypatch.setenv("PODCAST_LOGS_PUSH_URL", "http://backend:9428/insert/jsonline")
    monkeypatch.setenv("PODCAST_METRICS_PUSH_URL", "http://backend:8428/api/v1/import/prometheus")
    assert "worktree" in parse_qs(urlsplit(dev_push._logs_endpoint()).query)["_stream_fields"][0]
    with patch.object(dev_push, "_post") as post:
        dev_push.push_metrics_once()
    labels = parse_qs(urlsplit(post.call_args.args[0]).query)["extra_label"]
    assert "worktree=feat/a&b#c" in labels, "a branch name survives the query string"
