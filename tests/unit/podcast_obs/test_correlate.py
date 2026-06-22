"""Agent-correlatable o11y: run-scoped sources + the correlate join (#1053)."""

from __future__ import annotations

import hashlib
import json

import pytest

from podcast_obs import aggregate
from podcast_obs.config import TargetConfig
from podcast_obs.sources import langfuse, loki, sentry


def _t(**kw) -> TargetConfig:
    return TargetConfig(name="t", **kw)


def _lf(**kw) -> TargetConfig:
    return _t(langfuse_public_key="pk", langfuse_secret_key="sk", **kw)


# ── langfuse.trace_id_for_run / trace_by_run ────────────────────────────────


def test_trace_id_matches_sdk_seed_algorithm() -> None:
    # must equal create_trace_id(seed=run_id) = sha256(seed)[:16].hex()
    assert langfuse.trace_id_for_run("run-x") == hashlib.sha256(b"run-x").digest()[:16].hex()


def test_trace_by_run_not_configured() -> None:
    assert langfuse.trace_by_run(_t(), "run-1")["configured"] is False


def test_trace_by_run_404_is_found_false_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, **_):
        raise RuntimeError("404 not found")

    monkeypatch.setattr(langfuse, "get_json", boom)
    res = langfuse.trace_by_run(_lf(), "run-empty")
    assert res["ok"] is True  # a run with no LLM calls is valid, not an error
    assert res["data"]["found"] is False
    assert res["data"]["trace_id"] == langfuse.trace_id_for_run("run-empty")


def test_trace_by_run_surfaces_episode_and_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "name": "run trace",
        "totalCost": 0.025,
        "observations": [
            {
                "name": "summarization:claude",
                "model": "claude",
                "totalCost": 0.023,
                "usageDetails": {"input": 1800, "output": 420},
                "metadata": {"stage": "summarization", "episode_id": "ep:e01", "provider": "c"},
            },
            {
                "name": "gi:gpt",
                "model": "gpt",
                "totalCost": 0.001,
                "metadata": {"stage": "gi", "episode_id": "ep:e02"},
            },
        ],
    }
    monkeypatch.setattr(langfuse, "get_json", lambda url, **_: payload)
    data = langfuse.trace_by_run(_lf(), "run-1")["data"]
    assert data["found"] is True
    assert data["observation_count"] == 2
    # the agent can attribute each call to a stage AND an episode
    assert data["observations"][0]["episode_id"] == "ep:e01"
    assert data["observations"][0]["stage"] == "summarization"
    assert data["observations"][1]["episode_id"] == "ep:e02"


# ── loki.cost_for_run ───────────────────────────────────────────────────────


def test_cost_for_run_not_configured() -> None:
    assert loki.cost_for_run(_t(), "run-1")["configured"] is False


def _loki_target() -> TargetConfig:
    return _t(loki_url="https://logs.example/loki/api/v1/push", loki_user="42", loki_token="glc_x")


def test_cost_for_run_parses_events_and_total(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        json.dumps(
            {
                "event_type": "llm_cost",
                "provider": "claude",
                "stage": "summarization",
                "model": "claude",
                "estimated_cost_usd": 0.023,
                "run_id": "run-1",
            }
        ),
        json.dumps(
            {
                "event_type": "llm_cost",
                "provider": "gpt",
                "stage": "gi",
                "model": "gpt",
                "estimated_cost_usd": 0.001,
                "run_id": "run-1",
            }
        ),
    ]
    streams = {"data": {"result": [{"stream": {}, "values": [["2", lines[1]], ["1", lines[0]]]}]}}
    captured = {}

    def fake(url, **kw):
        captured["query"] = kw["params"]["query"]
        return streams

    monkeypatch.setattr(loki, "get_json", fake)
    res = loki.cost_for_run(_loki_target(), "run-1")
    assert res["ok"] is True
    assert 'run_id="run-1"' in captured["query"]  # filtered by the join key
    assert res["data"]["count"] == 2
    assert res["data"]["total_cost_usd"] == pytest.approx(0.024)
    assert {e["stage"] for e in res["data"]["events"]} == {"summarization", "gi"}


# ── sentry.recent_errors(run_id=...) ────────────────────────────────────────


def test_recent_errors_run_id_filters_query(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake(url, **kw):
        captured["query"] = kw["params"]["query"]
        return []

    monkeypatch.setattr(sentry, "get_json", fake)
    target = _t(sentry_token="x", sentry_org="acme", sentry_projects=("api",))
    sentry.recent_errors(target, run_id="run-1")
    assert 'run_id:"run-1"' in captured["query"]
    assert "is:unresolved" not in captured["query"]  # full picture for the run


# ── aggregate.correlate ─────────────────────────────────────────────────────


def test_correlate_joins_and_degrades_per_source(monkeypatch: pytest.MonkeyPatch) -> None:
    # langfuse configured (mock the trace), loki + sentry NOT configured.
    monkeypatch.setattr(langfuse, "get_json", lambda url, **_: {"name": "t", "observations": []})
    res = aggregate.correlate(_lf(), "run-1")
    assert res["ok"] is True
    d = res["data"]
    assert d["run_id"] == "run-1"
    assert set(d["signals"].keys()) == {"trace", "cost", "errors"}
    assert "trace" in d["live"]  # langfuse answered
    assert {"cost", "errors"} <= set(d["unconfigured"])  # degraded independently


def test_correlate_one_bad_source_does_not_break_the_join(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(target, run_id):
        raise RuntimeError("kaboom")

    # a correlator that raises (not just returns err) must be caught by the join loop.
    monkeypatch.setattr(aggregate, "_CORRELATORS", [("trace", boom)])
    res = aggregate.correlate(_lf(), "run-1")
    assert res["ok"] is True  # the join still returns
    assert res["data"]["signals"]["trace"]["ok"] is False
