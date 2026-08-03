"""Live E2E for the observability control plane (#803) — closes the loop against real APIs.

Why a sibling dir (not ``tests/e2e/``): that suite's conftest blocks sockets and injects a
pipeline ``e2e_server``; these tests need *real* network to GitHub/Sentry/Grafana/VictoriaLogs
and the target deploy. Run them explicitly (real sockets, no ``--disable-socket``)::

    .venv/bin/python -m pytest tests/e2e_observability/ -q --no-cov -p no:cacheprovider

**Non-determinism is handled by asserting shape + invariants, never values** (a deploy/cost/log
list changes between runs). Each test **self-skips** when its source isn't configured
(``configured=False``) or the target is unreachable — so the same file is safe to run anywhere:
locally it uses whatever creds you have (GitHub via ``gh``, prod_api against a reachable deploy),
skipping the rest until you provide tokens.

Configuration (any of):
- ``PODCAST_OBS_CONFIG`` (YAML) or ``PODCAST_OBS_*`` env vars — the normal control-plane config.
- Otherwise a default target at ``http://localhost:8080`` (override with ``PODCAST_OBS_API_BASE``).
- GitHub: ``PODCAST_OBS_GITHUB_TOKEN`` if set, else the authenticated ``gh`` CLI token.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from typing import Any

import pytest

from podcast_obs import aggregate
from podcast_obs.config import ObservabilityConfig, TargetConfig
from podcast_obs.sources import github, grafana, langfuse, prod_api, sentry, victoria

pytestmark = pytest.mark.e2e

# Derived from aggregate's canonical probe list so this never drifts when a source is
# added or removed (e.g. #1355's Loki→VictoriaLogs swap and the RFC-088 enrichment probes).
_ALL_LABELS = {label for label, _ in aggregate._PROBES}


def _gh_cli_token() -> str | None:
    """A read-only GitHub token from the authenticated ``gh`` CLI, if available."""
    if not shutil.which("gh"):
        return None
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _live_target() -> TargetConfig:
    if os.environ.get("PODCAST_OBS_CONFIG") or os.environ.get("PODCAST_OBS_API_BASE"):
        target = ObservabilityConfig.load().target(os.environ.get("PODCAST_OBS_TARGET"))
    else:
        target = TargetConfig(name="e2e", api_base="http://localhost:8080")
    if not target.github_token:
        token = _gh_cli_token()
        if token:
            target = replace(target, github_token=token)
    return target


def _live_or_skip(result: dict, label: str) -> Any:
    """Skip cleanly when not configured or unreachable; return the data when live."""
    if not result["ok"] and result.get("configured") is False:
        pytest.skip(f"{label}: not configured ({result['error']})")
    if not result["ok"]:
        pytest.skip(f"{label}: unreachable ({result['error']})")
    return result["data"]


def test_prod_api_health_live() -> None:
    data = _live_or_skip(prod_api.health(_live_target()), "prod_api.health")
    assert data.get("status") == "ok"
    assert isinstance(data.get("code_version"), str)


def test_prod_api_runs_live() -> None:
    data = _live_or_skip(prod_api.recent_pipeline_runs(_live_target(), limit=5), "prod_api.runs")
    assert isinstance(data["runs"], list)
    for run in data["runs"]:
        assert "job_id" in run and "status" in run


def test_github_deploys_live() -> None:
    data = _live_or_skip(github.recent_deploys(_live_target(), limit=5), "github.deploys")
    assert isinstance(data["deploys"], list)
    assert data["failure_rate"] is None or 0.0 <= data["failure_rate"] <= 1.0
    for deploy in data["deploys"]:
        assert {"run_number", "status", "conclusion", "sha"} <= set(deploy)
        assert deploy["conclusion"] is None or isinstance(deploy["conclusion"], str)


def test_victoria_cost_events_live() -> None:
    # #1355 migrated the control-plane cost signal off Loki onto VictoriaLogs: cost is now the
    # raw ``llm_cost`` emit_event stream (``aggregate`` maps the "cost" label to this), not a
    # server-side ``estimated_cost_usd`` sum. Assert the wrapper shape, not values (live data).
    data = _live_or_skip(
        victoria.events(_live_target(), "llm_cost", window="24h", limit=5), "victoria.cost"
    )
    assert data["event_type"] == "llm_cost"
    assert isinstance(data["events"], list)
    assert data["count"] == len(data["events"])


def test_victoria_recent_logs_live() -> None:
    data = _live_or_skip(
        victoria.recent_logs(_live_target(), window="1h", limit=10), "victoria.logs"
    )
    assert isinstance(data["lines"], list)
    assert data["count"] == len(data["lines"])


def test_grafana_alerts_live() -> None:
    data = _live_or_skip(grafana.recent_alerts(_live_target(), limit=10), "grafana.alerts")
    assert isinstance(data["alerts"], list)


def test_sentry_errors_live() -> None:
    data = _live_or_skip(
        sentry.recent_errors(_live_target(), window="24h", limit=5), "sentry.errors"
    )
    assert isinstance(data["projects"], list)
    assert data["total_issues"] >= 0


def test_langfuse_traces_live() -> None:
    data = _live_or_skip(langfuse.recent_traces(_live_target(), limit=5), "langfuse.traces")
    assert isinstance(data["traces"], list)
    assert isinstance(data["count"], int)
    assert data["base_url"].startswith("http")


def test_summary_structure_live() -> None:
    # summary always returns ok=True; assert the label partition is complete and disjoint.
    data = aggregate.summary(_live_target())["data"]
    labels = set(data["live"]) | set(data["unconfigured"]) | set(data["failed"])
    assert labels == _ALL_LABELS
    assert len(labels) == len(data["live"]) + len(data["unconfigured"]) + len(data["failed"])
