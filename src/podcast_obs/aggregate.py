"""``summary`` — the control-plane glance: fan out across every implemented source for a
target, tolerating per-source failure, so a half-configured deploy still returns useful state.

As later slices add sources (github/grafana/sentry), append them to ``_PROBES`` and they
automatically join the summary and the CLI.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

from .config import TargetConfig
from .result import err, ok
from .sources import enrichment, github, grafana, langfuse, loki, prod_api, sentry, victoria

# A surface name → its metrics ``job`` label and its Jaeger trace ``service`` name. These differ
# and are LIVE-VERIFIED against homelab (metrics carry job="api"; Jaeger service is "pipeline",
# not "podcast-pipeline"). A surface with no HTTP metrics (the pipeline subprocess) has job=None.
_SURFACE: dict[str, dict[str, str | None]] = {
    "api": {"job": "api", "trace": "podcast-api"},
    "pipeline": {"job": None, "trace": "pipeline"},
    "player": {"job": "player", "trace": "player-api"},
    "operator": {"job": "operator", "trace": "operator-public-api"},
}

# (label, probe) — each probe takes a TargetConfig and returns a result envelope.
# Sources whose credentials aren't set for a target return ``configured=False`` and land in the
# "unconfigured" bucket, so a local-only target still gives a useful glance.
_PROBES: list[tuple[str, Callable[[TargetConfig], dict]]] = [
    ("health", prod_api.health),
    ("version", prod_api.deployed_version),
    ("runs", prod_api.recent_pipeline_runs),
    ("deploys", github.recent_deploys),
    ("cost", loki.cost_today),
    ("logs", lambda target: loki.recent_logs(target, limit=5)),  # compact for the glance
    ("errors", sentry.recent_errors),
    ("alerts", grafana.recent_alerts),
    ("traces", lambda target: langfuse.recent_traces(target, limit=5)),  # compact for the glance
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


def surface(target: TargetConfig, name: str, *, window: str = "1h") -> dict:
    """Observe ONE surface (api / pipeline / player / operator): its five-signal snapshot.

    The literal "observe the API" / "observe the pipeline" verb — RED metrics (VictoriaMetrics),
    recent errors (GlitchTip), error-ish logs (VictoriaLogs), recent traces (VictoriaTraces), and
    for the pipeline the per-stage `pipeline_stage` rollup + LLM cost. Each degrades independently.
    """
    mapping = _SURFACE.get(name, {"job": name, "trace": name})
    job, trace_service = mapping["job"], (mapping["trace"] or name)
    probes: dict[str, Callable[[], dict]] = {
        "errors": lambda: sentry.recent_errors(target, window=window, limit=10),
        "logs": lambda: victoria.recent_logs(
            target, surface=name, level="error", window=window, limit=20
        ),
        "traces": lambda: victoria.traces_recent(target, trace_service, window=window, limit=10),
    }
    if job:  # a surface with no HTTP metrics (the pipeline subprocess) skips RED
        probes["metrics"] = lambda: victoria.red_metrics(target, job, window="5m")
    if name == "pipeline":
        probes["pipeline_stage"] = lambda: victoria.events(
            target, "pipeline_stage", surface="pipeline", window=window, limit=50
        )
        probes["cost"] = lambda: victoria.events(
            target, "llm_cost", surface="pipeline", window=window, limit=200
        )
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
_CORRELATORS: list[tuple[str, Callable[[TargetConfig, str], dict]]] = [
    ("trace", langfuse.trace_by_run),  # Langfuse: per-call model/cost/tokens for the run
    ("cost", loki.cost_for_run),  # Loki: the run's llm_cost events + total
    ("errors", lambda target, run_id: sentry.recent_errors(target, run_id=run_id)),  # Sentry
    # Loki: the run's log lines (CorrelationFormatter stamps ``[run=<id>]`` onto each).
    (
        "logs",
        lambda target, run_id: loki.recent_logs(
            target, level="", contains=f"run={run_id}", window="24h", limit=100
        ),
    ),
    # RFC-088: enrichment events filtered to this run (enrichment.*.run_id matches).
    (
        "enrichment_events",
        lambda target, run_id: enrichment.recent_events(target, run_id=run_id, limit=100),
    ),
]


def correlate(target: TargetConfig, run_id: str) -> dict:
    """Every signal for one ``run_id``, joined — the agent's cross-layer view (#1053).

    Fans out the run-scoped probes (Langfuse trace, Loki cost events, Sentry errors) and
    returns them under one envelope, each degrading independently (``configured=False``
    when its backend isn't wired). This is what lets an agent take a run and see what it
    did, what it cost, and whether it errored — in one call.
    """
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
