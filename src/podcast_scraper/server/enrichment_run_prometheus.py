"""Prometheus observations from ``enrichments/run_summary.json`` (post-job).

The metrics half of #2071. Enrichment reached the central stack through no channel at all:
measured on prod 2026-09-14, ``{__name__=~"enrichment.*"}`` and ``{__name__=~"enricher.*"}``
were both EMPTY in VictoriaMetrics while the pipeline had ``podcast_pipeline_run_*`` histograms.
So enrichment could be inspected on demand via ``/api/enrichment/*``, but nothing could ALERT on
it — not "this enricher has failed its last N runs", not "enrichment stopped happening at all".

Deliberately mirrors :mod:`podcast_scraper.server.pipeline_run_prometheus`: same lazy
``prometheus_client`` import, same ``PODCAST_METRICS_ENABLED`` gate, same post-terminal hook in
``jobs._finalize_job``. A second, differently-shaped telemetry path is how enrichment ended up
invisible in the first place, so this one joins the existing road rather than laying a new one.

Source is ``run_summary.json`` rather than ``run.jsonl``: the summary is the already-aggregated
per-enricher rollup (runs_ok / runs_failed / duration_ms / records_written / retries / cost), so
there is no second aggregation to drift from the one ``/api/enrichment/run-summary`` serves.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Same ladder as the pipeline histograms — an enrichment run is seconds-to-minutes, and a WEB
#: tier enricher walking a rate-limited upstream can sit at the top end (5 orgs took 45s against
#: Wikidata on 2026-09-14), so the buckets must not saturate there.
_SECONDS_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    900.0,
)

#: run_summary per-enricher counter keys -> the ``outcome`` label value.
_OUTCOME_KEYS = {
    "runs_ok": "ok",
    "runs_failed": "failed",
    "runs_timeout": "timeout",
    "runs_quarantined": "quarantined",
    "runs_skipped": "skipped",
    "runs_cancelled": "cancelled",
}

_PROM_STATE: dict[str, Any] = {"done": False}


def _env_metrics_enabled() -> bool:
    v = os.environ.get("PODCAST_METRICS_ENABLED", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _ensure_prom() -> None:
    if _PROM_STATE["done"]:
        return
    try:
        from prometheus_client import Counter, Histogram
    except ImportError:
        return

    _PROM_STATE["runs"] = Counter(
        "podcast_enrichment_runs_total",
        "Enrichment runs that reached a terminal status.",
        ["status"],
    )
    _PROM_STATE["run_duration"] = Histogram(
        "podcast_enrichment_run_duration_seconds",
        "run_summary.json: wall time of the whole enrichment run.",
        buckets=_SECONDS_BUCKETS,
    )
    # The series that answers "is this enricher healthy": rate() by outcome per enricher.
    _PROM_STATE["enricher_runs"] = Counter(
        "podcast_enrichment_enricher_runs_total",
        "run_summary.json: per-enricher terminal outcomes.",
        ["enricher_id", "outcome"],
    )
    _PROM_STATE["enricher_duration"] = Histogram(
        "podcast_enrichment_enricher_duration_seconds",
        "run_summary.json: per-enricher wall time.",
        ["enricher_id"],
        buckets=_SECONDS_BUCKETS,
    )
    # Throughput. A run that is "ok" while writing zero rows is the org_web failure mode
    # (#2071): green status, no output. Only a records series makes that visible.
    _PROM_STATE["records"] = Counter(
        "podcast_enrichment_records_written_total",
        "run_summary.json: rows written per enricher.",
        ["enricher_id"],
    )
    _PROM_STATE["retries"] = Counter(
        "podcast_enrichment_retries_total",
        "run_summary.json: retries per enricher.",
        ["enricher_id"],
    )
    _PROM_STATE["cost"] = Counter(
        "podcast_enrichment_cost_usd_total",
        "run_summary.json: LLM cost (USD) per enricher.",
        ["enricher_id"],
    )
    _PROM_STATE["done"] = True


def _read_run_summary(corpus_root: Path) -> dict[str, Any] | None:
    path = Path(corpus_root) / "enrichments" / "run_summary.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("enrichment run_summary unreadable at %s: %s", path, exc)
        return None
    return doc if isinstance(doc, dict) else None


def observe_enrichment_terminal_metrics(corpus_root: Path, job: Mapping[str, Any]) -> None:
    """Record Prometheus samples after ``jobs._finalize_job`` updates an enrichment *job*.

    Best-effort and silent when metrics are disabled or ``prometheus_client`` is absent —
    observability is additive to a run, never load-bearing for it.
    """
    if not _env_metrics_enabled():
        return
    _ensure_prom()
    runs_ctr = _PROM_STATE.get("runs")
    if runs_ctr is None:  # prometheus_client not installed
        return

    status = str(job.get("status") or "").strip().lower()
    if status not in {"succeeded", "failed", "cancelled", "stale"}:
        return
    runs_ctr.labels(status=status).inc()

    summary = _read_run_summary(Path(corpus_root))
    if summary is None:
        return

    duration_ms = summary.get("duration_ms")
    if isinstance(duration_ms, (int, float)) and duration_ms >= 0:
        _PROM_STATE["run_duration"].observe(float(duration_ms) / 1000.0)

    per_enricher = summary.get("per_enricher")
    if not isinstance(per_enricher, dict):
        return

    for enricher_id, row in per_enricher.items():
        if not isinstance(row, dict):
            continue
        eid = str(enricher_id)
        for key, outcome in _OUTCOME_KEYS.items():
            n = row.get(key)
            if isinstance(n, (int, float)) and n > 0:
                _PROM_STATE["enricher_runs"].labels(enricher_id=eid, outcome=outcome).inc(float(n))
        d_ms = row.get("duration_ms")
        if isinstance(d_ms, (int, float)) and d_ms >= 0:
            _PROM_STATE["enricher_duration"].labels(enricher_id=eid).observe(float(d_ms) / 1000.0)
        for field, metric in (
            ("records_written", "records"),
            ("retries", "retries"),
            ("cost_usd", "cost"),
        ):
            v = row.get(field)
            if isinstance(v, (int, float)) and v > 0:
                _PROM_STATE[metric].labels(enricher_id=eid).inc(float(v))
