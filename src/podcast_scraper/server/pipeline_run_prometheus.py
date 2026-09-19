"""Prometheus observations from pipeline ``run.json`` summaries (post-job).

When ``POST /api/jobs`` subprocess exits, successful runs leave one or more
``run.json`` files under the corpus tree (per-feed workspace). This module
finds those files whose mtimes fall inside a padded wall-clock window derived
from the job registry's ``started_at`` / ``ended_at`` stamps and observes
histogram samples so Grafana Agent can scrape ``/metrics`` (requires
``PODCAST_METRICS_ENABLED``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

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


def parse_iso_utc_z(value: str | None) -> datetime | None:
    """Parse API/registry ISO timestamps (``…Z`` or offset-aware)."""
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def discover_run_json_paths_in_mtime_window(
    root: Path,
    t_lo: float,
    t_hi: float,
) -> list[Path]:
    """Return ``run.json`` paths under *root* whose mtimes lie in ``[t_lo, t_hi]``.

    ``t_lo`` / ``t_hi`` are :func:`os.path.getmtime` epoch seconds (float).
    """
    root_safe = str(root.resolve())
    safe_prefix = root_safe + os.sep
    out: list[Path] = []
    try:
        for dirpath, _, filenames in os.walk(root_safe):
            if "run.json" not in filenames:
                continue
            candidate = os.path.normpath(os.path.join(dirpath, "run.json"))
            if candidate != root_safe and not candidate.startswith(safe_prefix):
                continue
            try:
                mt = os.path.getmtime(candidate)
            except OSError:
                continue
            if t_lo <= mt <= t_hi:
                out.append(Path(candidate))
    except OSError as exc:
        logger.warning("run.json discovery failed under %s: %s", root_safe, exc)
        return []
    return sorted(out)


def _env_metrics_enabled() -> bool:
    v = os.environ.get("PODCAST_METRICS_ENABLED", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


_PROM_STATE: dict[str, Any] = {"done": False}


def _ensure_prom_hist() -> None:
    if _PROM_STATE["done"]:
        return
    try:
        from prometheus_client import Counter, Histogram
    except ImportError:
        return

    _PROM_STATE["Histogram"] = Histogram
    _PROM_STATE["Counter"] = Counter
    _PROM_STATE["avg_transcribe"] = Histogram(
        "podcast_pipeline_run_avg_transcribe_seconds",
        "Per-feed run.json: average transcribe wall time per episode (seconds).",
        buckets=_SECONDS_BUCKETS,
    )
    _PROM_STATE["avg_summarize"] = Histogram(
        "podcast_pipeline_run_avg_summarize_seconds",
        "Per-feed run.json: average summarize wall time per episode (seconds).",
        buckets=_SECONDS_BUCKETS,
    )
    _PROM_STATE["avg_gi"] = Histogram(
        "podcast_pipeline_run_avg_gi_seconds",
        "Per-feed run.json: average GI artifact generation time per episode.",
        buckets=_SECONDS_BUCKETS,
    )
    _PROM_STATE["avg_kg"] = Histogram(
        "podcast_pipeline_run_avg_kg_seconds",
        "Per-feed run.json: average KG artifact generation time per episode.",
        buckets=_SECONDS_BUCKETS,
    )
    _PROM_STATE["run_duration"] = Histogram(
        "podcast_pipeline_run_duration_seconds",
        "Per-feed run.json: pipeline run_duration_seconds aggregate.",
        buckets=_SECONDS_BUCKETS,
    )
    _PROM_STATE["wait_transcription"] = Histogram(
        "podcast_pipeline_run_time_transcription_wait_seconds",
        "Per-feed run.json: time_transcription_wait_seconds (thread-accounted).",
        buckets=_SECONDS_BUCKETS,
    )
    _PROM_STATE["wait_summarization"] = Histogram(
        "podcast_pipeline_run_time_summarization_wait_seconds",
        "Per-feed run.json: time_summarization_wait_seconds (thread-accounted).",
        buckets=_SECONDS_BUCKETS,
    )
    _PROM_STATE["jobs_finished"] = Counter(
        "podcast_pipeline_jobs_finished_total",
        "Pipeline subprocess jobs that reached a terminal status.",
        ["status", "command_type"],
    )
    _PROM_STATE["run_json_hits"] = Counter(
        "podcast_pipeline_run_json_observed_total",
        "run.json files whose metrics were observed for Prometheus.",
    )
    # Cost / volume series (P2.9) — Counters so PromQL can rate()/sum() spend + throughput over
    # time. Incremented by each observed run.json's totals.
    _PROM_STATE["run_cost_usd"] = Counter(
        "podcast_pipeline_run_cost_usd_total",
        "Per-feed run.json: total LLM + transcription cost (USD) summed across stages.",
    )
    _PROM_STATE["run_episodes"] = Counter(
        "podcast_pipeline_run_episodes_total",
        "Per-feed run.json: episodes scraped/processed.",
    )
    _PROM_STATE["run_gi_artifacts"] = Counter(
        "podcast_pipeline_run_gi_artifacts_total",
        "Per-feed run.json: GI artifacts generated.",
    )
    _PROM_STATE["run_kg_artifacts"] = Counter(
        "podcast_pipeline_run_kg_artifacts_total",
        "Per-feed run.json: KG artifacts generated.",
    )
    _PROM_STATE["done"] = True


def _float_metric_block(metrics: Mapping[str, Any], key: str) -> float | None:
    v = metrics.get(key)
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        if x >= 0.0:
            return x
        return None
    return None


def _observe_metrics_mapping(metrics: Mapping[str, Any]) -> None:
    _ensure_prom_hist()
    if not _PROM_STATE.get("avg_transcribe"):
        return
    pairs = [
        ("avg_transcribe", "avg_transcribe_seconds"),
        ("avg_summarize", "avg_summarize_seconds"),
        ("avg_gi", "avg_gi_seconds"),
        ("avg_kg", "avg_kg_seconds"),
        ("run_duration", "run_duration_seconds"),
        ("wait_transcription", "time_transcription_wait_seconds"),
        ("wait_summarization", "time_summarization_wait_seconds"),
    ]
    for prom_key, json_key in pairs:
        val = _float_metric_block(metrics, json_key)
        if val is None:
            continue
        hist = _PROM_STATE[prom_key]
        hist.observe(val)
    ctr = _PROM_STATE.get("run_json_hits")
    if ctr is not None:
        ctr.inc()

    # Cost / volume Counters (P2.9). Cost = sum of the per-stage cost fields present in run.json.
    from podcast_scraper.workflow.corpus_cost_aggregation import _STAGE_COST_FIELDS

    cost = sum((_float_metric_block(metrics, k) or 0.0) for k in _STAGE_COST_FIELDS)
    if cost > 0.0:
        _PROM_STATE["run_cost_usd"].inc(cost)
    for prom_key, json_key in (
        ("run_episodes", "episodes_scraped_total"),
        ("run_gi_artifacts", "gi_artifacts_generated"),
        ("run_kg_artifacts", "kg_artifacts_generated"),
    ):
        val = _float_metric_block(metrics, json_key)
        if val and val > 0.0:
            _PROM_STATE[prom_key].inc(val)


#: ``command_type`` values allowed onto the metric as themselves. Anything else collapses to
#: ``other``. The set mirrors ``jobs.COMMAND_*``; it is duplicated rather than imported because
#: importing ``server.jobs`` here would pull the whole job-runner in at metrics-init time.
#: ``test_command_type_label_allows_every_declared_command`` asserts the two stay in step.
_KNOWN_COMMAND_TYPES = frozenset(
    {"full_incremental_pipeline", "corpus_enrichment", "corpus_reindex"}
)


def _command_type_label(job: Mapping[str, Any]) -> str:
    """Bound the ``command_type`` label to a known set.

    A label fed straight from a record field is where cardinality explodes: the values are
    fixed constants today, but nothing stops a future caller passing a per-run string, and a
    counter with unbounded labels degrades the whole TSDB rather than just this metric.
    """
    value = str(job.get("command_type") or "").strip()
    return value if value in _KNOWN_COMMAND_TYPES else "other"


def observe_pipeline_terminal_metrics(corpus_root: Path, job: Mapping[str, Any]) -> None:
    """Record Prometheus samples after ``jobs._finalize_job`` updates *job*.

    The ``command_type`` label on ``podcast_pipeline_jobs_finished_total`` is what lets an
    alert say "the NIGHTLY has not completed" rather than "no job of any kind has completed".
    Without it, an operator-triggered enrichment succeeding would mask a dead nightly — and
    the stalled-ingestion alert has to distinguish those two (#2119 follow-up).
    """
    if not _env_metrics_enabled():
        return
    _ensure_prom_hist()
    jobs_ctr = _PROM_STATE.get("jobs_finished")
    if jobs_ctr is None:
        return

    status = str(job.get("status") or "").strip().lower()
    if status not in {"succeeded", "failed", "cancelled", "stale"}:
        return

    jobs_ctr.labels(status=status, command_type=_command_type_label(job)).inc()

    if status != "succeeded":
        return

    _st = job.get("started_at")
    started = parse_iso_utc_z(_st if isinstance(_st, str) else None)
    _en = job.get("ended_at")
    ended = parse_iso_utc_z(_en if isinstance(_en, str) else None)
    if ended is None:
        ended = datetime.now(timezone.utc)
    pad = 180.0
    if started is None:
        t_hi = ended.timestamp() + 60.0
        t_lo = t_hi - 86400.0 * 2
    else:
        t_lo = started.timestamp() - pad
        t_hi = ended.timestamp() + pad

    paths = discover_run_json_paths_in_mtime_window(corpus_root.resolve(), t_lo, t_hi)
    if not paths:
        logger.debug(
            "prometheus: no run.json in mtime window for job=%s corpus=%s",
            job.get("job_id"),
            corpus_root,
        )
        return

    for p in paths:
        try:
            raw_any = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw_any, dict):
            continue
        metrics_any = raw_any.get("metrics")
        if not isinstance(metrics_any, dict):
            continue
        _observe_metrics_mapping(metrics_any)


def last_success_age_seconds(corpus_root: Path, now: float | None = None) -> dict[str, float]:
    """Seconds since the most recent SUCCESSFUL job of each ``command_type``.

    Read from the job registry rather than accumulated in memory, which is the property that
    matters: the registry is durable, so this survives an API restart or a deploy. A counter
    cannot do that — ``increase(...[36h])`` over a series that was reset minutes ago reports
    nothing-has-succeeded, which is indistinguishable from a genuine stall.

    Command types with no successful job are ABSENT from the result, never zero. Zero would
    read as "succeeded just now" and invert any alert built on it.
    """
    from podcast_scraper.server.pipeline_job_registry import read_jobs

    now = time.time() if now is None else now
    newest: dict[str, float] = {}
    for job in read_jobs(corpus_root):
        if str(job.get("status") or "").strip().lower() != "succeeded":
            continue
        raw_ended = job.get("ended_at")
        ended = parse_iso_utc_z(raw_ended if isinstance(raw_ended, str) else None)
        if ended is None:
            continue
        label = _command_type_label(job)
        ts = ended.timestamp()
        if ts > newest.get(label, float("-inf")):
            newest[label] = ts
    # Clamp at 0: a job record timestamped slightly in the future (clock skew between the
    # pipeline container and the API) must not produce a negative age that reads as healthy.
    return {label: max(0.0, now - ts) for label, ts in newest.items()}


def install_job_age_metrics(corpus_root: Path) -> bool:
    """Publish ``podcast_pipeline_last_success_age_seconds`` for the job registry.

    THE SIGNAL THIS REPLACES
    ------------------------
    The ``podcast-ingest-stalled`` alert used to query the VictoriaLogs tail of the job
    registry file. That could never work: the registry is a MUTABLE document rewritten whole
    on every status change (``pipeline_job_registry.write_jobs_atomic`` does a temp-file
    rename), while Alloy tails it with ``loki.source.file``, which assumes append-only. The
    tailer resumes at a stale byte offset inside a rewritten file and ships JSON fragments.

    Measured over 30 days, the alert's exact query matched on exactly ONE day — the day the
    whole file happened to be re-read from offset zero. Every other day: nothing. The alert
    was not broken recently; it never worked, and a bulk re-read was the only thing that ever
    made it look green.

    Collected at scrape time so the value reflects the registry as of the scrape.
    Never raises: telemetry must not break the app (ADR-120).
    """
    try:
        from prometheus_client import REGISTRY
        from prometheus_client.core import GaugeMetricFamily
    except Exception:  # noqa: BLE001 — metrics extras absent; run without this gauge
        return False

    class _JobAgeCollector:
        def collect(self):  # noqa: ANN202 — prometheus_client's duck-typed collector protocol
            age = GaugeMetricFamily(
                "podcast_pipeline_last_success_age_seconds",
                "Seconds since the last successful job of this command_type (absent = never).",
                labels=["command_type"],
            )
            try:
                ages = last_success_age_seconds(corpus_root)
            except Exception:  # noqa: BLE001 — an unreadable registry exports nothing
                ages = {}
            for label, seconds in ages.items():
                age.add_metric([label], seconds)
            yield age

    try:
        REGISTRY.register(_JobAgeCollector())  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 — duplicate registration on app re-create is harmless
        return False
    return True


__all__ = [
    "discover_run_json_paths_in_mtime_window",
    "install_job_age_metrics",
    "last_success_age_seconds",
    "observe_pipeline_terminal_metrics",
    "parse_iso_utc_z",
]
