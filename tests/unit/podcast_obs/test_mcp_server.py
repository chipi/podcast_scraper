"""MCP wrapper: tool table wiring, target resolution, and build smoke."""

from __future__ import annotations

import pytest

from podcast_obs import mcp_server
from podcast_obs.config import ObservabilityConfig, TargetConfig
from podcast_obs.sources import prod_api


def _config(**target_kw) -> ObservabilityConfig:
    target = TargetConfig(name="local", **target_kw)
    return ObservabilityConfig(targets={"local": target}, default_target="local")


def test_tool_table_names_and_count() -> None:
    tools = mcp_server._build_tools(_config())
    names = [fn.__name__ for fn in tools]
    assert names == [
        "prod_health",
        "prod_version",
        "prod_recent_runs",
        "prod_recent_deploys",
        "prod_cost_today",
        "prod_recent_logs",
        "prod_recent_errors",
        "prod_recent_alerts",
        "prod_recent_traces",
        "prod_summary",
        "prod_correlate",
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
