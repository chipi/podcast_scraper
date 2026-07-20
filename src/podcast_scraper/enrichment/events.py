"""JSONL event vocabulary for the enrichment layer.

Every event line carries the correlation envelope (run_id +
enricher_id + tier + attempt) so a single ``run_id`` lookup in the
JSONL file returns the full enrichment chain for that run. The MCP
``enrichment_recent_events`` tool consumes these directly.

This module defines:

* Event type constants (str) — the stable vocabulary.
* Builder functions that compose ``(event_type, payload_dict)`` from
  the correlation context + per-event fields.
* A thin JSONL appender (``append_event``) the executor uses; the
  builders are equally usable with the existing workflow JSONLEmitter
  if the executor is invoked from inside a pipeline run.

See ``docs/wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md``
§"JSONL event vocabulary".
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.correlation import jsonl_event_extras
from podcast_scraper.enrichment.envelope import utc_iso_now
from podcast_scraper.enrichment.paths import ensure_directory
from podcast_scraper.enrichment.protocol import RunContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event type vocabulary (the stable contract; do not rename casually).
# ---------------------------------------------------------------------------

EVENT_RUN_STARTED = "enrichment.run.started"
EVENT_RUN_COMPLETED = "enrichment.run.completed"
EVENT_RUN_SKIPPED = "enrichment.run.skipped"

EVENT_ENRICHER_STARTED = "enrichment.enricher.started"
EVENT_ENRICHER_RETRY = "enrichment.enricher.retry"
EVENT_ENRICHER_COMPLETED = "enrichment.enricher.completed"
EVENT_ENRICHER_CIRCUIT_OPENED = "enrichment.enricher.circuit_opened"
EVENT_ENRICHER_AUTO_DISABLED = "enrichment.enricher.auto_disabled"
EVENT_ENRICHER_CANCELLED = "enrichment.enricher.cancelled"
EVENT_ENRICHER_STALL_WARNING = "enrichment.enricher.stall_warning"

EVENT_HEALTH_RE_ENABLED = "enrichment.health.re_enabled"

ALL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EVENT_RUN_STARTED,
        EVENT_RUN_COMPLETED,
        EVENT_RUN_SKIPPED,
        EVENT_ENRICHER_STARTED,
        EVENT_ENRICHER_RETRY,
        EVENT_ENRICHER_COMPLETED,
        EVENT_ENRICHER_CIRCUIT_OPENED,
        EVENT_ENRICHER_AUTO_DISABLED,
        EVENT_ENRICHER_CANCELLED,
        EVENT_ENRICHER_STALL_WARNING,
        EVENT_HEALTH_RE_ENABLED,
    }
)


# ---------------------------------------------------------------------------
# Builders — pair (event_type, payload). Always include event_type +
# timestamp ("ts") in the payload so the JSONL line is self-describing.
# ---------------------------------------------------------------------------


def _base_payload(event_type: str, **extras: Any) -> dict[str, Any]:
    """Build the common shape: event_type + ts + extras."""
    payload: dict[str, Any] = {
        "event_type": event_type,
        "ts": utc_iso_now(),
    }
    for key, value in extras.items():
        if value is not None:
            payload[key] = value
    return payload


def build_run_started(
    *,
    run_id: str,
    parent_run_id: str | None,
    profile: str | None,
    enricher_set: list[str],
) -> dict[str, Any]:
    """``enrichment.run.started`` — fired once at executor start."""
    return _base_payload(
        EVENT_RUN_STARTED,
        run_id=run_id,
        parent_run_id=parent_run_id,
        profile=profile,
        enricher_set=list(enricher_set),
    )


def build_run_completed(
    *,
    run_id: str,
    parent_run_id: str | None,
    duration_ms: int,
    per_enricher_totals: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """``enrichment.run.completed`` — fired once at executor end.

    ``per_enricher_totals`` maps enricher_id → ``{"ok": n, "failed": n, ...}``
    counter snapshot for the run.
    """
    return _base_payload(
        EVENT_RUN_COMPLETED,
        run_id=run_id,
        parent_run_id=parent_run_id,
        duration_ms=int(duration_ms),
        per_enricher_totals=dict(per_enricher_totals),
    )


def build_run_skipped(*, run_id: str, reason: str) -> dict[str, Any]:
    """``enrichment.run.skipped`` — pipeline-attached path bails when
    the core pipeline failed (per chunk-1 lock audit §B4).
    """
    return _base_payload(
        EVENT_RUN_SKIPPED,
        run_id=run_id,
        reason=reason,
    )


def build_enricher_started(ctx: RunContext, *, scope: str) -> dict[str, Any]:
    """``enrichment.enricher.started`` — fired before each enrich() call."""
    return _base_payload(
        EVENT_ENRICHER_STARTED,
        scope=scope,
        **jsonl_event_extras(ctx),
    )


def build_enricher_retry(
    ctx: RunContext, *, backoff_s: float, reason: str, error_class: str | None
) -> dict[str, Any]:
    """``enrichment.enricher.retry`` — fired before each retry attempt."""
    return _base_payload(
        EVENT_ENRICHER_RETRY,
        backoff_s=float(backoff_s),
        reason=reason,
        error_class=error_class,
        **jsonl_event_extras(ctx),
    )


def build_enricher_completed(
    ctx: RunContext,
    *,
    status: str,
    duration_ms: int,
    records_written: int,
    retries: int,
    error: str | None = None,
    error_class: str | None = None,
) -> dict[str, Any]:
    """``enrichment.enricher.completed`` — fired after each terminal outcome.

    ``error`` and ``error_class`` MUST be forwarded from ``EnricherResult``
    on non-ok terminal outcomes so post-mortem tooling (harden audits,
    dashboards, oncall) can diagnose without re-running. Both were
    dropped on the floor pre-2026-07-17 → the temporal_velocity v1.2.0
    prod-v2 regression surfaced as ``status:failed`` with zero
    diagnostic payload.
    """
    payload = _base_payload(
        EVENT_ENRICHER_COMPLETED,
        status=status,
        duration_ms=int(duration_ms),
        records_written=int(records_written),
        retries=int(retries),
        **jsonl_event_extras(ctx),
    )
    # Only surface diagnostics on non-ok terminal outcomes to keep the
    # happy-path event payload lean. Nulls elided so the JSONL schema
    # stays additive.
    if status != "ok":
        if error is not None:
            payload["error"] = str(error)
        if error_class is not None:
            payload["error_class"] = str(error_class)
    return payload


def build_circuit_opened(
    ctx: RunContext, *, consecutive_failures: int, cooldown_until: str | None
) -> dict[str, Any]:
    """``enrichment.enricher.circuit_opened`` — fired when the breaker trips."""
    return _base_payload(
        EVENT_ENRICHER_CIRCUIT_OPENED,
        consecutive_failures=int(consecutive_failures),
        cooldown_until=cooldown_until,
        opened_at=utc_iso_now(),
        **jsonl_event_extras(ctx),
    )


def build_auto_disabled(
    ctx: RunContext, *, consecutive_failed_runs: int, reason: str
) -> dict[str, Any]:
    """``enrichment.enricher.auto_disabled`` — fired when cross-run threshold tripped."""
    return _base_payload(
        EVENT_ENRICHER_AUTO_DISABLED,
        consecutive_failed_runs=int(consecutive_failed_runs),
        reason=reason,
        disabled_at=utc_iso_now(),
        **jsonl_event_extras(ctx),
    )


def build_cancelled(
    ctx: RunContext, *, reason: str, partial_records_written: int
) -> dict[str, Any]:
    """``enrichment.enricher.cancelled`` — fired on cooperative cancel."""
    return _base_payload(
        EVENT_ENRICHER_CANCELLED,
        reason=reason,
        partial_records_written=int(partial_records_written),
        **jsonl_event_extras(ctx),
    )


def build_stall_warning(
    ctx: RunContext,
    *,
    last_heartbeat_at: str,
    expected_interval_s: float,
) -> dict[str, Any]:
    """``enrichment.enricher.stall_warning`` — fired by the heartbeat watchdog."""
    return _base_payload(
        EVENT_ENRICHER_STALL_WARNING,
        last_heartbeat_at=last_heartbeat_at,
        expected_interval_s=float(expected_interval_s),
        **jsonl_event_extras(ctx),
    )


def build_health_re_enabled(
    *,
    enricher_id: str,
    operator_id: str | None,
    reset_counter: bool,
    cleared_cooldown: bool,
    reason: str,
) -> dict[str, Any]:
    """``enrichment.health.re_enabled`` — fired on operator manual recovery."""
    return _base_payload(
        EVENT_HEALTH_RE_ENABLED,
        enricher_id=enricher_id,
        operator_id=operator_id,
        reset_counter=bool(reset_counter),
        cleared_cooldown=bool(cleared_cooldown),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Thin appender — concurrent-write-safe via per-path thread lock.
# ---------------------------------------------------------------------------


_write_lock = threading.Lock()


def append_event(jsonl_path: Path, payload: dict[str, Any]) -> None:
    """Append one event as a JSONL line, atomically.

    Creates the parent directory if missing. Lock-serialized to keep
    interleaving sane when the executor's async pipeline emits from
    multiple coroutines (asyncio is single-threaded but the lock makes
    the file-append safe across thread-pool workers used by
    ``@sync_enricher``).

    Failures (disk full, IO error) log a WARNING but never raise into
    the executor — the o11y surface is best-effort.
    """
    try:
        ensure_directory(jsonl_path.parent)
        line = json.dumps(payload, ensure_ascii=False)
        with _write_lock:
            with open(jsonl_path, "a", encoding="utf-8") as fp:
                fp.write(line)
                fp.write("\n")
    except OSError as exc:
        logger.warning(
            "enrichment event append failed (path=%s, event=%s): %s",
            jsonl_path,
            payload.get("event_type"),
            exc,
        )
