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

from .aggregate import correlate as _correlate, summary as _summary
from .config import ObservabilityConfig, ObservabilityConfigError, TargetConfig
from .result import err
from .sources import enrichment, github, grafana, langfuse, loki, prod_api, sentry

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8848

_INSTRUCTIONS = (
    "Read-only prod observability control plane. Tools answer 'what is a deploy doing right "
    "now?' — health, version, recent pipeline runs and deploys, today's LLM cost, recent error "
    "logs (Loki) and Sentry issues, current Grafana alerts, and recent Langfuse LLM traces. "
    "Each tool takes an optional "
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

    def prod_recent_traces(target: Optional[str] = None, limit: int = 10) -> dict:
        """Recent Langfuse LLM traces for a deploy (id/name/timestamp/latency/cost)."""
        return _run(config, target, langfuse.recent_traces, limit=limit)

    def prod_summary(target: Optional[str] = None) -> dict:
        """One-call control-plane glance: every source for a deploy (live/unconfigured/failed)."""
        return _run(config, target, _summary)

    def prod_correlate(run_id: str, target: Optional[str] = None) -> dict:
        """Every signal for ONE run_id, joined: Langfuse trace (per-call model/cost/tokens) +
        Loki llm_cost events + Sentry errors + enrichment health snapshot. The cross-layer
        view for a single run (#1053 + RFC-088)."""
        return _run(config, target, lambda t: _correlate(t, run_id))

    # --- RFC-088 enrichment-layer tools --------------------------------------------

    def enrichment_run_status(target: Optional[str] = None) -> dict:
        """Last enrichment-layer status snapshot for the deploy's corpus."""
        return _run(config, target, enrichment.run_status)

    def enrichment_recent_runs(target: Optional[str] = None, limit: int = 10) -> dict:
        """Recent enrichment jobs (`command_type=corpus_enrichment`), newest first."""
        return _run(config, target, enrichment.recent_runs, limit=limit)

    def enrichment_health(target: Optional[str] = None, enricher_id: Optional[str] = None) -> dict:
        """Per-enricher health: consecutive_failures, auto_disabled, last_error.
        Pass `enricher_id` to drill into a single enricher's record."""
        return _run(config, target, enrichment.health, enricher_id=enricher_id)

    def enrichment_metrics(target: Optional[str] = None, window: str = "24h") -> dict:
        """Rollup metrics over a window (default 24h): per-enricher success/duration/cost."""
        return _run(config, target, enrichment.metrics, window=window)

    def enrichment_recent_events(
        target: Optional[str] = None,
        enricher_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> dict:
        """JSONL event tail (filter by enricher_id / event_type, default last 50)."""
        return _run(
            config,
            target,
            enrichment.recent_events,
            enricher_id=enricher_id,
            event_type=event_type,
            limit=limit,
        )

    def enrichment_eval_history(
        target: Optional[str] = None,
        eval_root: Optional[str] = None,
        limit: int = 10,
    ) -> dict:
        """Recent enrichment-tagged eval runs from `data/eval/runs/` on disk
        (operator-side; eval artefacts are frozen-once-written)."""
        return _run(
            config,
            target,
            enrichment.eval_history,
            eval_root=eval_root,
            limit=limit,
        )

    def enrichment_re_enable(
        enricher_id: str,
        target: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> dict:
        """Clear `auto_disabled` for an enricher and zero `consecutive_failures` after a
        transient outage. `reason` is appended to the health audit trail."""
        return _run(
            config,
            target,
            enrichment.re_enable,
            enricher_id=enricher_id,
            reason=reason,
        )

    def enrichment_cancel(job_id: str, target: Optional[str] = None) -> dict:
        """Cancel a running or queued enrichment job by id (command_type-agnostic
        cancel — works because the jobs registry doesn't distinguish kinds)."""
        return _run(config, target, enrichment.cancel, job_id=job_id)

    return [
        prod_health,
        prod_version,
        prod_recent_runs,
        prod_recent_deploys,
        prod_cost_today,
        prod_recent_logs,
        prod_recent_errors,
        prod_recent_alerts,
        prod_recent_traces,
        prod_summary,
        prod_correlate,
        enrichment_run_status,
        enrichment_recent_runs,
        enrichment_health,
        enrichment_metrics,
        enrichment_recent_events,
        enrichment_eval_history,
        enrichment_re_enable,
        enrichment_cancel,
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
