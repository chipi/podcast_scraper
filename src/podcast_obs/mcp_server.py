"""MCP wrapper over the control-plane core (#803) — Layer B.

Exposes the same probes as MCP tools for agent clients (Claude Code, etc.). The core
(:mod:`podcast_obs.sources`) is the single source of truth; this is a thin adapter.

Transports:
- ``stdio`` — local dev / a co-located agent.
- ``sse`` / ``streamable-http`` — the containerised control plane, reachable over the tailnet,
  so an agent on another box can query it.

FastMCP is imported lazily inside :func:`build_server` so the core package (and its tests)
import without the MCP SDK installed (it rides in the ``[observability]`` extra).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .aggregate import summary as _summary
from .config import ObservabilityConfig, ObservabilityConfigError, TargetConfig
from .result import err
from .sources import github, grafana, loki, prod_api, sentry

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8848

_INSTRUCTIONS = (
    "Read-only prod observability control plane. Tools answer 'what is a deploy doing right "
    "now?' — health, version, recent pipeline runs and deploys, today's LLM cost, recent error "
    "logs (Loki) and Sentry issues, and current Grafana alerts. Each tool takes an optional "
    "`target` (a configured deploy name; omit for the default). Results are uniform envelopes: "
    "{ok, source, data|error, configured}; configured=false means that source isn't wired for "
    "the target. Compose with other MCP servers (e.g. Grafana) for deeper drill-down."
)


def _run(
    config: ObservabilityConfig,
    target: Optional[str],
    probe: Callable[..., dict],
    **kwargs: Any,
) -> dict:
    """Resolve *target* (or the default) and invoke *probe*; return a config-error envelope
    instead of raising when the target is unknown."""
    try:
        resolved: TargetConfig = config.target(target)
    except ObservabilityConfigError as exc:
        return err("config", str(exc))
    return probe(resolved, **kwargs)


def _build_tools(config: ObservabilityConfig) -> list[Callable[..., dict]]:
    """The MCP tool callables (closures over *config*). Returned for direct testing."""

    def prod_health(target: Optional[str] = None) -> dict:
        """Full /api/health for a deploy (status, code/corpus versions, feature flags)."""
        return _run(config, target, prod_api.health)

    def prod_version(target: Optional[str] = None) -> dict:
        """The running code version and the corpus stamp (git sha) a deploy is serving."""
        return _run(config, target, prod_api.deployed_version)

    def prod_recent_runs(target: Optional[str] = None, limit: int = 10) -> dict:
        """Recent pipeline runs (/api/jobs) for a deploy, newest first."""
        return _run(config, target, prod_api.recent_pipeline_runs, limit=limit)

    def prod_recent_deploys(target: Optional[str] = None, limit: int = 10) -> dict:
        """Recent deploy-prod.yml runs (GitHub Actions) with conclusions + failure rate."""
        return _run(config, target, github.recent_deploys, limit=limit)

    def prod_cost_today(target: Optional[str] = None) -> dict:
        """Estimated LLM spend over the last 24h for a deploy (from Loki cost events)."""
        return _run(config, target, loki.cost_today)

    def prod_recent_logs(
        target: Optional[str] = None,
        level: str = "error",
        service: Optional[str] = None,
        window: str = "1h",
        limit: int = 50,
        contains: Optional[str] = None,
    ) -> dict:
        """Recent container logs from Loki (error-ish by default) — what Sentry didn't capture."""
        return _run(
            config,
            target,
            loki.recent_logs,
            level=level,
            service=service,
            window=window,
            limit=limit,
            contains=contains,
        )

    def prod_recent_errors(
        target: Optional[str] = None, window: str = "24h", limit: int = 10
    ) -> dict:
        """Recent unresolved Sentry issues for a deploy's environment."""
        return _run(config, target, sentry.recent_errors, window=window, limit=limit)

    def prod_recent_alerts(target: Optional[str] = None, limit: int = 20) -> dict:
        """Current Grafana alerts (alertname, severity, state, summary)."""
        return _run(config, target, grafana.recent_alerts, limit=limit)

    def prod_summary(target: Optional[str] = None) -> dict:
        """One-call control-plane glance: every source for a deploy (live/unconfigured/failed)."""
        return _run(config, target, _summary)

    return [
        prod_health,
        prod_version,
        prod_recent_runs,
        prod_recent_deploys,
        prod_cost_today,
        prod_recent_logs,
        prod_recent_errors,
        prod_recent_alerts,
        prod_summary,
    ]


def build_server(
    config: ObservabilityConfig, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> Any:
    """Build a FastMCP server exposing the control-plane probes as tools."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("podcast-obs", instructions=_INSTRUCTIONS, host=host, port=port)
    for tool in _build_tools(config):
        server.tool()(tool)
    return server


def run_server(
    config: ObservabilityConfig,
    *,
    transport: str = "stdio",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Build and run the MCP server over *transport* (stdio / sse / streamable-http)."""
    build_server(config, host=host, port=port).run(transport=transport)
