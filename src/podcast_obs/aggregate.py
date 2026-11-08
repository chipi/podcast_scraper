"""``summary`` — the control-plane glance: fan out across every implemented source for a
target, tolerating per-source failure, so a half-configured deploy still returns useful state.

As later slices add sources (github/grafana/sentry), append them to ``_PROBES`` and they
automatically join the summary and the CLI.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Callable, Optional

from .config import TargetConfig
from .result import err, ok
from .sources import enrichment, github, grafana, langfuse, prod_api, sentry, umami, victoria

# A surface name → its metrics ``job`` label and its Jaeger trace ``service`` name. These differ
# (metrics carry job="api"). The pipeline traces as "podcast-pipeline" in prod (compose
# OTEL_SERVICE_NAME; 33.9k spans in 7 days, 2026-10-14) and in dev (init_otel's default) —
# "pipeline" was the old dev-only name and found no prod run. A surface with no HTTP metrics
# (the pipeline subprocess) has job=None.
# ``logs`` is the surface's LogsQL selector. The log label ``surface`` is a sub-part of an app
# (api/mcp/app/obs/pipeline/web, LOG-STANDARD), so the MCP surface name is not a log filter:
# ``surface:player`` matched nothing and ``surface:api`` mixed the podcast and player APIs.
# ``job``/``service`` select the RED metrics: one scrape job ``api`` covers the three APIs, told
# apart by ``service`` (live-verified 2026-11-08; there is no ``job="player"`` or ``"operator"``).
_SURFACE: dict[str, dict[str, str | None]] = {
    "api": {
        "job": "api",
        "service": "podcast-api",
        "trace": "podcast-api",
        "logs": "app:podcast AND surface:api",
    },
    "pipeline": {
        "job": None,
        "service": None,
        "trace": "podcast-pipeline",
        "logs": "surface:pipeline",
    },
    "player": {"job": "api", "service": "player-api", "trace": "player-api", "logs": "app:player"},
    "operator": {
        "job": "api",
        "service": "operator-public-api",
        "trace": "operator-public-api",
        "logs": "app:operator",
    },
}

#: Every pipeline run event: ``pipeline_run.started`` / ``.completed`` (status succeeded, partial
#: or failed) from the run, ``.skipped`` from the scheduler (which runs in the api process, so no
#: surface filter).
RUN_EVENTS = "pipeline_run.*"

# (label, probe) — each probe takes a TargetConfig and returns a result envelope.
# Sources whose credentials aren't set for a target return ``configured=False`` and land in the
# "unconfigured" bucket, so a local-only target still gives a useful glance.
_PROBES: list[tuple[str, Callable[[TargetConfig], dict]]] = [
    ("health", prod_api.health),
    ("version", prod_api.deployed_version),
    ("runs", prod_api.recent_pipeline_runs),
    # Read-cache health (catalog / slug / KG index / artifacts) — hit rate + time-saved, so the
    # glance shows whether the caches are paying off or drifting toward cold-and-useless.
    ("cache_stats", prod_api.cache_stats),
    ("deploys", github.recent_deploys),
    # Observability probes come from the self-hosted Victoria stack (the box the app ships to), not
    # the legacy Grafana-Cloud Loki / Langfuse — otherwise the glance is dead against this deploy.
    ("cost", lambda target: victoria.events(target, "llm_cost", window="24h", limit=5)),
    ("logs", lambda target: victoria.recent_logs(target, limit=5)),  # compact for the glance
    ("errors", sentry.recent_errors),  # GlitchTip issues (optional; error LOGS come via `logs`)
    ("alerts", grafana.recent_alerts),
    # compact for the glance — recent pipeline spans from VictoriaTraces
    ("traces", lambda target: victoria.traces_recent(target, "podcast-pipeline", limit=5)),
    # RFC-088 enrichment-layer surface — the deploy's last status, health, and a compact
    # tail of events round out the control-plane glance.
    ("enrichment_status", enrichment.run_status),
    ("enrichment_health", enrichment.health),
    ("enrichment_events", lambda target: enrichment.recent_events(target, limit=5)),
]


def summary(target: TargetConfig) -> dict:
    """Run every implemented probe against *target* and collect the envelopes by label."""
    sources: dict[str, dict] = {}
    for label, probe in _PROBES:
        try:
            sources[label] = probe(target)
        except Exception as exc:  # noqa: BLE001 — a probe must never break the summary
            sources[label] = err(f"summary.{label}", f"probe raised: {exc}")
    live = sorted(label for label, res in sources.items() if res.get("ok"))
    unconfigured = sorted(
        label
        for label, res in sources.items()
        if not res.get("ok") and res.get("configured") is False
    )
    failed = sorted(
        label
        for label, res in sources.items()
        if not res.get("ok") and res.get("configured") is not False
    )
    return ok(
        "summary",
        {
            "target": target.name,
            "live": live,
            "unconfigured": unconfigured,
            "failed": failed,
            "sources": sources,
        },
    )


def _collect(probes: dict) -> dict:
    """Run ``{label: thunk}`` probes, tolerating failure; bucket live/unconfigured/failed."""
    signals: dict[str, dict] = {}
    for label, thunk in probes.items():
        try:
            signals[label] = thunk()
        except Exception as exc:  # noqa: BLE001 — one signal must never break the join
            signals[label] = err(label, f"probe raised: {exc}")
    return {
        "live": sorted(k for k, r in signals.items() if r.get("ok")),
        "unconfigured": sorted(
            k for k, r in signals.items() if not r.get("ok") and r.get("configured") is False
        ),
        "failed": sorted(
            k for k, r in signals.items() if not r.get("ok") and r.get("configured") is not False
        ),
        "signals": signals,
    }


def _umami_site(target: TargetConfig, frontend: Optional[str]) -> TargetConfig:
    """The target pointed at *frontend*'s own Umami site (player and operator are separate)."""
    site = target.umami_website_ids.get(frontend or "")
    return dataclasses.replace(target, umami_website_id=site) if site else target


#: The Postgres exporter's scrape job.
_PG = 'job="postgres"'
#: The exporter's own statements, left out of the top lists: they are its scrape, not the app.
_NOT_EXPORTER = 'user!="monitor"'


def _database_queries(window: str) -> dict[str, str]:
    """``{signal: PromQL}`` for the database surface, from the Postgres exporter's series.

    Statements carry ``queryid``, ``user`` and ``datname``, not the SQL text (the exporter does not
    export it). A ``queryid`` maps to its text with
    ``SELECT query FROM pg_stat_statements WHERE queryid = <id>`` on the server.
    """
    st = f"{_PG},{_NOT_EXPORTER}"
    by_stmt = "queryid, user, datname"
    time_rate = f"sum by ({by_stmt}) (rate(pg_stat_statements_seconds_total{{{st}}}[{window}]))"
    calls_rate = f"sum by ({by_stmt}) (rate(pg_stat_statements_calls_total{{{st}}}[{window}]))"
    return {
        "up": f"pg_up{{{_PG}}}",
        "exporter_error": f"pg_exporter_last_scrape_error{{{_PG}}}",
        "max_connections": f"pg_settings_max_connections{{{_PG}}}",
        "connections": f"sum by (datname, state) (pg_stat_activity_count{{{_PG}}}) > 0",
        "longest_transaction_seconds": (
            f"max by (datname, state) (pg_stat_activity_max_tx_duration{{{_PG}}}) > 0"
        ),
        "statements_by_time": f"topk(10, {time_rate} > 0)",
        "statements_by_calls": f"topk(10, {calls_rate} > 0)",
        "statements_by_mean_seconds": f"topk(10, ({time_rate}) / ({calls_rate}) > 0)",
        "tables_by_size_bytes": (
            f"topk(15, pg_stat_user_tables_table_size_bytes{{{_PG}}}"
            f" + pg_stat_user_tables_index_size_bytes{{{_PG}}})"
        ),
        "tables_by_dead_rows": f"topk(10, pg_stat_user_tables_n_dead_tup{{{_PG}}} > 0)",
        "tables_by_seq_rows_read": (
            f"topk(10, rate(pg_stat_user_tables_seq_tup_read{{{_PG}}}[{window}]) > 0)"
        ),
        "locks": f"sum by (datname, mode) (pg_locks_count{{{_PG}}}) > 0",
        "database_size_bytes": f"pg_database_size_bytes{{{_PG}}}",
    }


def database_surface(target: TargetConfig, *, window: str = "1h") -> dict:
    """The database as a surface: is it up, who is connected, which statements cost the most,
    which tables are large, bloated or scanned whole, what is locked. Each signal is one PromQL
    query over the exporter's series and degrades on its own."""
    probes: dict[str, Callable[[], dict]] = {
        name: partial(victoria.metrics_instant, target, query)
        for name, query in _database_queries(window).items()
    }
    return ok(
        "surface",
        {"target": target.name, "surface": "database", "window": window, **_collect(probes)},
    )


def surface(target: TargetConfig, name: str, *, window: str = "1h") -> dict:
    """Observe ONE surface (api / pipeline / player / operator / database): its snapshot.

    The literal "observe the API" / "observe the pipeline" verb — RED metrics (VictoriaMetrics),
    recent errors (GlitchTip), error-ish logs (VictoriaLogs), recent traces (VictoriaTraces), and
    for the pipeline the per-stage `pipeline_stage` rollup + LLM cost. Each degrades independently.
    ``database`` is the Postgres server instead (:func:`database_surface`).
    """
    if name == "database":
        return database_surface(target, window=window)
    mapping = _SURFACE.get(name, {"job": name, "trace": name})
    job, trace_service = mapping["job"], (mapping["trace"] or name)
    probes: dict[str, Callable[[], dict]] = {
        "errors": lambda: sentry.recent_errors(target, window=window, limit=10),
        "logs": lambda: victoria.recent_logs(
            target,
            surface=name,
            selector=mapping.get("logs"),
            level="error",
            window=window,
            limit=20,
        ),
        "traces": lambda: victoria.traces_recent(target, trace_service, window=window, limit=10),
    }
    if job:  # a surface with no HTTP metrics (the pipeline subprocess) skips RED
        service = mapping.get("service")
        probes["metrics"] = lambda: victoria.red_metrics(target, job, window="5m", service=service)
    if name == "pipeline":
        probes["runs"] = lambda: victoria.events(target, RUN_EVENTS, window=window, limit=50)
        probes["pipeline_stage"] = lambda: victoria.events(
            target, "pipeline_stage", surface="pipeline", window=window, limit=50
        )
        probes["cost"] = lambda: victoria.events(
            target, "llm_cost", surface="pipeline", window=window, limit=200
        )
    if name in ("operator", "player"):
        # The user-action lens (ADR-126 (not in this repository) Umami): what people actually DID on
        # this frontend — the typed custom events, page/visitor totals, and who's live now — from
        # THIS frontend's site.
        site = _umami_site(target, name)
        probes["user_actions"] = lambda: umami.events(site, window=window)
        probes["page_analytics"] = lambda: umami.stats(site, window=window)
        probes["active_users"] = lambda: umami.active(site)
    collected = _collect(probes)
    return ok(
        "surface",
        {
            "target": target.name,
            "surface": name,
            "job": job,
            "trace_service": trace_service,
            "window": window,
            **collected,
        },
    )


def analytics(target: TargetConfig, *, window: str = "24h", frontend: Optional[str] = None) -> dict:
    """Umami user-action analytics — what people DID on a frontend
    (ADR-126, not in this repository), joined in one call.

    The direct verb behind ``surface operator``'s Umami signals: the typed custom events (user
    actions), page/visitor totals, and who's live now. The website_id is per-environment
    (operator-dev/-prod), so this reads whichever site this target's env is pointed at. Degrades
    to ``configured=False`` when Umami read creds aren't wired.
    """

    def probes(site: TargetConfig) -> dict[str, Callable[[], dict]]:
        return {
            "user_actions": lambda: umami.events(site, window=window),
            "page_analytics": lambda: umami.stats(site, window=window),
            "active_users": lambda: umami.active(site),
        }

    # No frontend on a target with one site per frontend (prod: player + operator): there is no
    # default site, so this read nothing and reported all three signals "unconfigured" while both
    # sites were live (2026-10-27). Answer for every frontend instead, each labelled.
    if frontend is None and target.umami_website_ids and not target.umami_website_id:
        per = {
            fe: _collect(probes(_umami_site(target, fe))) for fe in sorted(target.umami_website_ids)
        }
        merged = {
            key: sorted(f"{fe}.{label}" for fe, got in per.items() for label in got[key])
            for key in ("live", "unconfigured", "failed")
        }
        return ok(
            "analytics",
            {"target": target.name, "frontend": None, "window": window, **merged, "frontends": per},
        )
    return ok(
        "analytics",
        {
            "target": target.name,
            "frontend": frontend,
            "window": window,
            **_collect(probes(_umami_site(target, frontend))),
        },
    )


def investigate(
    target: TargetConfig,
    *,
    trace_id: Optional[str] = None,
    run_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    window: str = "24h",
) -> dict:
    """Drill on ONE join key — fan every backend and return the correlated bundle.

    Give exactly one of ``trace_id`` (a request → its span tree + logs), ``run_id`` (a pipeline run
    → trace/cost/errors/logs/pipeline_stage), or ``episode_id`` (one episode → its pipeline_stage +
    cost + logs). The keys our recent work made real (run_id / episode_id / trace_id) are what make
    this cross-backend.
    """
    if not (trace_id or run_id or episode_id):
        return err("investigate", "provide one of trace_id / run_id / episode_id")
    probes: dict[str, Callable[[], dict]] = {}
    if trace_id:
        probes["trace"] = lambda: victoria.trace_by_id(target, trace_id)
        probes["trace_logs"] = lambda: victoria.recent_logs(
            target, level="", contains=trace_id, window=window, limit=100
        )
    if run_id:
        for _label, _probe in _CORRELATORS:
            probes[_label] = partial(_probe, target, run_id)
        probes["pipeline_stage"] = lambda: victoria.events(
            target, "pipeline_stage", run_id=run_id, window=window, limit=200
        )
        probes["run_events"] = lambda: victoria.events(
            target, RUN_EVENTS, run_id=run_id, window=window, limit=20
        )
    if episode_id:
        probes["ep_pipeline_stage"] = lambda: victoria.events(
            target, "pipeline_stage", episode_id=episode_id, window=window, limit=200
        )
        probes["ep_cost"] = lambda: victoria.events(
            target, "llm_cost", episode_id=episode_id, window=window, limit=200
        )
        probes["ep_logs"] = lambda: victoria.recent_logs(
            target, level="", contains=episode_id, window=window, limit=100
        )
    collected = _collect(probes)
    return ok(
        "investigate",
        {
            "target": target.name,
            "trace_id": trace_id,
            "run_id": run_id,
            "episode_id": episode_id,
            "window": window,
            **collected,
        },
    )


# (label, run-scoped probe) — every signal we can pull for ONE run_id (#1053).
#
# Current self-hosted stack: cost/errors/logs come from VictoriaLogs (the box the app ships to),
# NOT the legacy Grafana-Cloud Loki / Sentry that aren't part of this deployment. This is what a run
# investigation actually joins — the earlier re-alignment fixed surface() but left these on the dead
# legacy sources, so `investigate --run-id` reported cost/logs/errors as "unconfigured" while the
# data sat in VictoriaLogs (live-verified: llm_cost events queryable by run_id). Langfuse trace +
# enrichment events stay as OPTIONAL supplements — they degrade cleanly when unconfigured.
_CORRELATORS: list[tuple[str, Callable[[TargetConfig, str], dict]]] = [
    # VictoriaLogs: the run's llm_cost events (per-call provider/model/cost) — token-free.
    (
        "cost",
        lambda target, run_id: victoria.events(
            target, "llm_cost", run_id=run_id, window="24h", limit=200
        ),
    ),
    # VictoriaLogs: error-level lines for the run — visible without a Sentry token. In prod Alloy
    # tails container stderr; the CorrelationFormatter stamps ``[run=<id>]`` so `contains` scopes.
    (
        "errors",
        lambda target, run_id: victoria.recent_logs(
            target, level="error", contains=f"run={run_id}", window="24h", limit=100
        ),
    ),
    # GlitchTip issues tagged with this run (SDK-captured exceptions) — each carries a `permalink`
    # to the issue in the GlitchTip UI, so an agent can PIVOT from a run into the error's page. The
    # token-free `errors` above gives the text; this gives the clickable link (needs a read token).
    ("error_issues", lambda target, run_id: sentry.recent_errors(target, run_id=run_id)),
    # VictoriaLogs: every log line for the run.
    (
        "logs",
        lambda target, run_id: victoria.recent_logs(
            target, level="", contains=f"run={run_id}", window="24h", limit=100
        ),
    ),
    # VictoriaTraces: the run's episode.process spans (the root span stamps run_id) — run→trace
    # pivot on the current stack, no Langfuse needed.
    ("trace", lambda target, run_id: victoria.traces_by_run(target, run_id)),
    # Langfuse: per-call model/cost/tokens for the run (supplementary; degrades if keys unset).
    ("llm_trace", langfuse.trace_by_run),
    # RFC-088: enrichment events filtered to this run (supplementary; degrades if API unreachable).
    (
        "enrichment_events",
        lambda target, run_id: enrichment.recent_events(target, run_id=run_id, limit=100),
    ),
]


def correlate(target: TargetConfig, run_id: str) -> dict:
    """Every signal for one ``run_id``, joined — the agent's cross-layer view (#1053).

    Fans out the run-scoped probes (VictoriaTraces spans, VictoriaLogs cost events + logs +
    errors, enrichment; Langfuse llm_trace as an optional supplement) and returns them under one
    envelope, each degrading independently (``configured=False`` when its backend isn't wired).
    This is what lets an agent take a run and see what it did, what it cost, and whether it
    errored — in one call.
    """
    # An empty id drops every run filter below and returns the whole window's signals as if they
    # were this run's.
    if not (run_id or "").strip():
        return err("correlate", "run_id is required")
    signals: dict[str, dict] = {}
    for label, probe in _CORRELATORS:
        try:
            signals[label] = probe(target, run_id)
        except Exception as exc:  # noqa: BLE001 — one signal must never break the join
            signals[label] = err(f"correlate.{label}", f"probe raised: {exc}")
    return ok(
        "correlate",
        {
            "target": target.name,
            "run_id": run_id,
            "live": sorted(label for label, res in signals.items() if res.get("ok")),
            "unconfigured": sorted(
                label
                for label, res in signals.items()
                if not res.get("ok") and res.get("configured") is False
            ),
            "signals": signals,
        },
    )
