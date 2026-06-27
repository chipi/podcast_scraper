"""``summary`` — the control-plane glance: fan out across every implemented source for a
target, tolerating per-source failure, so a half-configured deploy still returns useful state.

As later slices add sources (github/grafana/sentry), append them to ``_PROBES`` and they
automatically join the summary and the CLI.
"""

from __future__ import annotations

from typing import Callable

from .config import TargetConfig
from .result import err, ok
from .sources import enrichment, github, grafana, langfuse, loki, prod_api, sentry

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
        lambda target, run_id: enrichment.recent_events(target, limit=100),
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
