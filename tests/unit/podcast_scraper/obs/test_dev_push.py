"""Tests for the dev-only in-app observability push (logs + metrics).

The load-bearing property is **inert unless explicitly enabled** — the packaged image
never sets the push-URL env vars, so every entry point must be a true no-op there
(prod ships via Alloy). The rest verifies labels + that a POST fires when enabled.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

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
    labels = dev_push.obs_labels()
    assert labels == {"service": "api", "environment": "dev", "instance": "wtB-8001"}


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
    assert "extra_label=environment=dev" in url
    assert "extra_label=instance=wtD-8003" in url


def test_emit_event_triggers_push_when_enabled(dev_push, monkeypatch):
    """The emit_event hook drives the push (no-op default; fires when the URL is set)."""
    monkeypatch.setenv("PODCAST_LOGS_PUSH_URL", "http://backend:9428/insert/jsonline")
    monkeypatch.setenv("PODCAST_OBS_INSTANCE", "wtE-8004")
    from podcast_scraper.obs import events

    with patch.object(dev_push, "_post") as post:
        events.emit_event("search_query", query_type="semantic")
        dev_push._flush_logs()
    assert post.called
