"""MCP wrapper: tool table wiring, target resolution, and build smoke."""

from __future__ import annotations

import pytest

from podcast_obs import mcp_server
from podcast_obs.config import ObservabilityConfig, TargetConfig
from podcast_obs.sources import enrichment, prod_api


def _config(**target_kw) -> ObservabilityConfig:
    target = TargetConfig(name="local", **target_kw)
    return ObservabilityConfig(targets={"local": target}, default_target="local")


def test_tool_table_names_and_count() -> None:
    tools = mcp_server._build_tools(_config())
    names = [fn.__name__ for fn in tools]
    assert names == [
        "prod_health",
        "prod_resilience",
        "prod_version",
        "prod_recent_runs",
        "prod_recent_deploys",
        "prod_cost_today",
        "prod_usage",
        "prod_recent_logs",
        "prod_recent_errors",
        "prod_recent_alerts",
        "prod_recent_traces",
        "prod_summary",
        "prod_correlate",
        "enrichment_run_status",
        "enrichment_recent_runs",
        "enrichment_health",
        "enrichment_metrics",
        "enrichment_recent_events",
        "enrichment_eval_history",
        "enrichment_re_enable",
        "enrichment_cancel",
    ]
    # every tool carries a docstring (FastMCP uses it as the tool description)
    assert all(fn.__doc__ for fn in tools)


def test_tool_dispatches_to_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prod_api, "get_json", lambda url, **_: {"status": "ok", "code_version": "1"}
    )
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["prod_health"]()
    assert result["ok"] is True
    assert result["data"]["status"] == "ok"


def test_tool_unknown_target_returns_config_error() -> None:
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["prod_health"](target="does-not-exist")
    assert result["ok"] is False
    assert result["source"] == "config"


def test_tool_default_target_used_when_omitted() -> None:
    # No api_base -> prod_api.health reports not-configured, proving the default target resolved.
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config())}
    result = tools["prod_health"]()
    assert result["configured"] is False


def test_build_server_registers_tools() -> None:
    server = mcp_server.build_server(_config(api_base="http://x"))
    assert server is not None
    assert server.name == "podcast-obs"


# --- RFC-088 enrichment-tool wiring ------------------------------------------------


def test_enrichment_run_status_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        enrichment, "get_json", lambda url, **_: {"available": True, "status": "ok"}
    )
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["enrichment_run_status"]()
    assert result["ok"] is True
    assert result["source"] == "enrichment.status"


def test_enrichment_recent_runs_filters_to_corpus_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "jobs": [
            {"job_id": "p1", "command_type": "full_incremental_pipeline", "created_at": "1"},
            {"job_id": "e1", "command_type": "corpus_enrichment", "created_at": "2"},
        ]
    }
    monkeypatch.setattr(enrichment, "get_json", lambda url, **_: payload)
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["enrichment_recent_runs"](limit=10)
    assert [r["job_id"] for r in result["data"]["runs"]] == ["e1"]


def test_enrichment_re_enable_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_post(url, *, json=None, **_):
        seen["url"] = url
        seen["json"] = json
        return {"enricher_id": "topic_similarity", "auto_disabled": False}

    monkeypatch.setattr(enrichment, "post_json", fake_post)
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["enrichment_re_enable"]("topic_similarity", reason="transient HF outage")
    assert result["ok"] is True
    assert seen["url"].endswith("/api/enrichment/health/topic_similarity/re-enable")
    assert seen["json"] == {"reason": "transient HF outage"}


def test_enrichment_cancel_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        enrichment, "post_json", lambda url, **_: {"job_id": "j7", "status": "cancelled"}
    )
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["enrichment_cancel"]("j7")
    assert result["ok"] is True
    assert result["data"]["status"] == "cancelled"


def test_enrichment_eval_history_passes_local_root(tmp_path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "enrichment-2026-06-25").mkdir()
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["enrichment_eval_history"](eval_root=str(runs))
    assert result["ok"] is True
    assert result["data"]["count"] == 1


def test_enrichment_tool_unknown_target_is_config_error() -> None:
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["enrichment_run_status"](target="missing")
    assert result["ok"] is False
    assert result["source"] == "config"


def test_enrichment_health_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_get_json(url, *, params=None, **_):
        seen["url"] = url
        seen["params"] = params
        return {"enrichers": {"x": {"auto_disabled": False}}}

    monkeypatch.setattr(enrichment, "get_json", fake_get_json)
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["enrichment_health"](enricher_id="x")
    assert result["ok"] is True
    assert result["source"] == "enrichment.health"
    assert seen["params"] == {"enricher_id": "x"}


def test_enrichment_metrics_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_get_json(url, *, params=None, **_):
        seen["params"] = params
        return {"window": "1h", "per_enricher": {}}

    monkeypatch.setattr(enrichment, "get_json", fake_get_json)
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["enrichment_metrics"](window="1h")
    assert result["ok"] is True
    assert seen["params"] == {"window": "1h"}


def test_enrichment_recent_events_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_get_json(url, *, params=None, **_):
        seen["params"] = params
        return {"events": [], "count": 0}

    monkeypatch.setattr(enrichment, "get_json", fake_get_json)
    tools = {fn.__name__: fn for fn in mcp_server._build_tools(_config(api_base="http://x"))}
    result = tools["enrichment_recent_events"](
        enricher_id="topic_similarity", event_type="enrichment.enricher.completed", limit=20
    )
    assert result["ok"] is True
    assert seen["params"] == {
        "limit": 20,
        "enricher_id": "topic_similarity",
        "event_type": "enrichment.enricher.completed",
    }


def test_enrichment_source_command_type_constant_matches_server_jobs() -> None:
    """Local COMMAND_ENRICHMENT mirror must stay in lockstep with
    server.jobs.COMMAND_ENRICHMENT (cross-package without import)."""
    from podcast_obs.sources.enrichment import COMMAND_ENRICHMENT as obs_const
    from podcast_scraper.server.jobs import COMMAND_ENRICHMENT as server_const

    assert obs_const == server_const
