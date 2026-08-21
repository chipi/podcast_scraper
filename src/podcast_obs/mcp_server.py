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

import os
from typing import Any, Callable, Optional

from . import __version__
from .aggregate import (
    analytics as _analytics,
    correlate as _correlate,
    investigate as _investigate,
    summary as _summary,
    surface as _surface,
)
from .config import ObservabilityConfig, ObservabilityConfigError, TargetConfig
from .result import err
from .sources import enrichment, github, grafana, langfuse, prod_api, sentry, victoria

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8848

# Bump the tool-schema suffix when the tool surface changes shape (v3 added prod_cache_stats — the
# read-cache health probe). Lets an agent negotiate compatibility.
_VERSION_TAG = f"podcast_obs {__version__} (tools v3)"


def _writes_allowed() -> bool:
    """Mutating tools are OFF unless PODCAST_OBS_ALLOW_WRITES is explicitly set — the control plane
    is observe-first, so a read-only agent can't accidentally re-enable/cancel a deploy job."""
    return os.environ.get("PODCAST_OBS_ALLOW_WRITES", "").strip().lower() in {"1", "true", "yes"}


_INSTRUCTIONS = (
    "Observability control plane for the podcast deploys. Mostly read-only; two enrichment tools "
    "(enrichment_re_enable, enrichment_cancel) MUTATE deploy state and are gated behind "
    "PODCAST_OBS_ALLOW_WRITES=1 (default off → they refuse). "
    "Observe a whole surface with obs_surface(surface=api|pipeline|player|operator); drill on a "
    "join key with obs_investigate(trace_id|run_id|episode_id). Reach raw signals with obs_events "
    "(the emit_event stream: pipeline_stage / llm_cost / search_query in VictoriaLogs), "
    "obs_metrics (PromQL / VictoriaMetrics), obs_traces (VictoriaTraces spans). Plus health/runs/"
    "deploys, cost, error logs + errors, alerts, and Langfuse LLM traces. Each tool takes optional "
    "`target` (a configured deploy name; omit for the default). Results are uniform envelopes: "
    "{ok, source, data|error, configured}; configured=false means that source isn't wired for the "
    f"target. Schema/version: {_VERSION_TAG}. Compose with a Grafana MCP for deep dashboards."
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


def _build_tools(config: ObservabilityConfig) -> list[Callable[..., dict]]:  # noqa: C901
    """The MCP tool callables (closures over *config*). Returned for direct testing.

    Deliberately one function of many tiny closures (a registration table), not complex logic — the
    C901 count just reflects the number of tools.
    """

    def prod_health(target: Optional[str] = None) -> dict:
        """Full /api/health for a deploy (status, code/corpus versions, feature flags)."""
        return _run(config, target, prod_api.health)

    def prod_version(target: Optional[str] = None) -> dict:
        """The running code version and the corpus stamp (git sha) a deploy is serving."""
        return _run(config, target, prod_api.deployed_version)

    def prod_resilience(target: Optional[str] = None) -> dict:
        """Is the deploy backing off or out of money? Open LLM/RSS circuit breakers (per provider),
        their cooldowns, and the configured LLM call-fuse budgets. Non-empty ``llm_breakers_open``
        means we are actively backing off a provider (e.g. gemini-flash-lite under load)."""
        return _run(config, target, prod_api.resilience)

    def prod_cache_stats(target: Optional[str] = None) -> dict:
        """Are the read caches earning their keep, or slowing us down? Per-namespace hit/miss/size
        for the in-process perf caches (catalog, slug index, KG index, corpus artifacts), plus
        ``avg_build_ms`` (what a miss costs) and ``est_saved_seconds`` (wall-clock saved on hits).
        Read it: high hit rate + high avg_build_ms = a win; low + low = near-free overhead to drop;
        a hit rate that craters after a deploy = warming regressed.
        """
        return _run(config, target, prod_api.cache_stats)

    def prod_recent_runs(target: Optional[str] = None, limit: int = 10) -> dict:
        """Recent pipeline runs (/api/jobs) for a deploy, newest first."""
        return _run(config, target, prod_api.recent_pipeline_runs, limit=limit)

    def prod_recent_deploys(target: Optional[str] = None, limit: int = 10) -> dict:
        """Recent deploy-prod.yml runs (GitHub Actions) with conclusions + failure rate."""
        return _run(config, target, github.recent_deploys, limit=limit)

    def prod_cost_today(target: Optional[str] = None) -> dict:
        """LLM cost events over the last 24h for a deploy (VictoriaLogs llm_cost stream)."""
        return _run(
            config, target, lambda t: victoria.events(t, "llm_cost", window="24h", limit=500)
        )

    def prod_usage(
        target: Optional[str] = None,
        group_by: str = "provider,model",
        run_id: Optional[str] = None,
    ) -> dict:
        """LLM token/cost rollup for a deploy, sliced by ``group_by`` — attribute tokens+cost to a
        model, operation (gi/evidence/cleaning/summarization), episode, or run. Carries the
        input/output/cached/cache-write breakdown and is de-duplicated by request_id (no double
        counting). Self-contained (no Loki). ``group_by`` is a comma list of: provider, model,
        served_model, operation, stage, episode_id, run_id, feed_id."""
        return _run(config, target, prod_api.usage, group_by=group_by, run_id=run_id or "")

    def prod_recent_logs(
        target: Optional[str] = None,
        level: str = "error",
        service: Optional[str] = None,
        window: str = "1h",
        limit: int = 50,
        contains: Optional[str] = None,
    ) -> dict:
        """Recent logs from VictoriaLogs (error-ish by default) — what GlitchTip didn't capture."""
        return _run(
            config,
            target,
            victoria.recent_logs,
            level=level,
            surface=service,
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
        """Every signal for ONE run_id, joined: VictoriaTraces spans + VictoriaLogs llm_cost
        events + errors + logs + enrichment (Langfuse llm_trace as an optional supplement). The
        cross-layer view for a single run (#1053 + RFC-088)."""
        return _run(config, target, lambda t: _correlate(t, run_id))

    def obs_surface(surface: str, target: Optional[str] = None, window: str = "1h") -> dict:
        """Observe ONE surface — the "observe the API / the pipeline" verb. `surface` is
        api / pipeline / player / operator. Returns its five-signal snapshot: RED metrics
        (VictoriaMetrics), recent errors (GlitchTip), error logs (VictoriaLogs), traces
        (VictoriaTraces), and — for the pipeline — the per-stage pipeline_stage rollup + LLM cost.
        Each signal degrades independently (configured=false when its backend isn't wired)."""
        return _run(config, target, lambda t: _surface(t, surface, window=window))

    def obs_analytics(target: Optional[str] = None, window: str = "24h") -> dict:
        """User-action analytics from Umami (ADR-126) — what people actually DID on the operator
        viewer / player: the typed custom events (user_actions), page/visitor totals
        (page_analytics), and who's live now (active_users). The website_id is per-environment
        (operator-dev/-prod). Degrades to configured=false when Umami read creds aren't wired."""
        return _run(config, target, lambda t: _analytics(t, window=window))

    def obs_investigate(
        target: Optional[str] = None,
        trace_id: Optional[str] = None,
        run_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        window: str = "24h",
    ) -> dict:
        """Drill on ONE join key — give exactly one of `trace_id` (a request → span tree + logs),
        `run_id` (a pipeline run → trace/cost/errors/logs/pipeline_stage), or `episode_id` (one
        episode → its pipeline_stage + cost + logs). Fans every backend and returns the correlated
        bundle. The cross-backend investigate verb (built on run_id/episode_id/trace_id)."""
        return _run(
            config,
            target,
            lambda t: _investigate(
                t, trace_id=trace_id, run_id=run_id, episode_id=episode_id, window=window
            ),
        )

    def obs_events(
        event_type: str,
        target: Optional[str] = None,
        surface: Optional[str] = None,
        run_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        window: str = "1h",
        limit: int = 50,
    ) -> dict:
        """Query the canonical emit_event stream in VictoriaLogs by `event_type`
        (`pipeline_stage` per-stage cost/quality/versions, `llm_cost`, `search_query`). Optional
        `surface` (api/pipeline), `run_id`, `episode_id` scope it. This is how an agent reaches the
        per-episode processing signal (RFC-109) and correlates it."""
        return _run(
            config,
            target,
            victoria.events,
            event_type=event_type,
            surface=surface,
            run_id=run_id,
            episode_id=episode_id,
            window=window,
            limit=limit,
        )

    def obs_metrics(query: str, target: Optional[str] = None) -> dict:
        """Run one PromQL instant query against VictoriaMetrics (raw RED/resource metrics)."""
        return _run(config, target, victoria.metrics_instant, query=query)

    def obs_traces(
        service: str, target: Optional[str] = None, window: str = "1h", limit: int = 10
    ) -> dict:
        """Recent VictoriaTraces spans for a Jaeger service (podcast-api / podcast-pipeline) — the
        general request/span traces Langfuse (LLM-only) doesn't cover."""
        return _run(
            config, target, victoria.traces_recent, service=service, window=window, limit=limit
        )

    def prod_run_summary(target: Optional[str] = None) -> dict:
        """Last completed enrichment run summary (`/api/enrichment/run-summary`)."""
        return _run(config, target, enrichment.run_summary)

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
        """MUTATES deploy state. Clear `auto_disabled` for an enricher and zero
        `consecutive_failures` after a transient outage. `reason` is appended to the health audit
        trail. Gated behind PODCAST_OBS_ALLOW_WRITES=1."""
        if not _writes_allowed():
            return err("writes-disabled", "mutating tool off; set PODCAST_OBS_ALLOW_WRITES=1")
        return _run(
            config,
            target,
            enrichment.re_enable,
            enricher_id=enricher_id,
            reason=reason,
        )

    def enrichment_cancel(job_id: str, target: Optional[str] = None) -> dict:
        """MUTATES deploy state. Cancel a running or queued enrichment job by id (command_type-
        agnostic — works because the jobs registry doesn't distinguish kinds). Gated behind
        PODCAST_OBS_ALLOW_WRITES=1."""
        if not _writes_allowed():
            return err("writes-disabled", "mutating tool off; set PODCAST_OBS_ALLOW_WRITES=1")
        return _run(config, target, enrichment.cancel, job_id=job_id)

    return [
        prod_health,
        prod_resilience,
        prod_cache_stats,
        prod_version,
        prod_recent_runs,
        prod_recent_deploys,
        prod_cost_today,
        prod_usage,
        prod_recent_logs,
        prod_recent_errors,
        prod_recent_alerts,
        prod_recent_traces,
        prod_summary,
        prod_correlate,
        obs_surface,
        obs_analytics,
        obs_investigate,
        obs_events,
        obs_metrics,
        obs_traces,
        prod_run_summary,
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
    """Build and run the MCP server over *transport* (stdio / sse / streamable-http).

    stdio is local-trust (no auth). The networked transports are PUBLIC-facing (reached through
    the ``ops.<domain>`` edge), so they are wrapped in :class:`~podcast_obs.auth.ObsAuthMiddleware`
    — every request presents a bearer to the app's verify seam and must resolve to an ADMIN user,
    else 401. The RFC 9728 discovery doc stays un-authenticated so a cold client can bootstrap.
    """
    server = build_server(config, host=host, port=port)
    if transport == "stdio":
        server.run(transport="stdio")
        return

    from .auth import ObsAuthMiddleware

    app = server.sse_app() if transport == "sse" else server.streamable_http_app()
    import uvicorn

    uvicorn.run(ObsAuthMiddleware(app), host=host, port=port)
