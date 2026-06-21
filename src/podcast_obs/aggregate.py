"""``summary`` — the control-plane glance: fan out across every implemented source for a
target, tolerating per-source failure, so a half-configured deploy still returns useful state.

As later slices add sources (github/grafana/sentry), append them to ``_PROBES`` and they
automatically join the summary and the CLI.
"""

from __future__ import annotations

from typing import Callable

from .config import TargetConfig
from .result import err, ok
from .sources import github, grafana, loki, prod_api, sentry

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
