"""``enrichments/run_summary.json`` — per-run outcome record.

Written once at end of run. The MCP ``enrichment_run_status(run_id)``
tool reads this for past runs; the viewer Operator-tab Enrichment
panel drill-down consumes it.

Schema:

::

    {
      "schema_version": "1",
      "run_id": "<job-id>",
      "parent_run_id": "<pipeline-run-id or null>",
      "profile": "<profile name or null>",
      "started_at": "<iso>",
      "finished_at": "<iso>",
      "duration_ms": <int>,
      "status": "ok" | "failed" | "cancelled",
      "per_enricher": {
        "<enricher_id>": {
          "status": "<status>",
          "duration_ms": <int>,
          "records_written": <int>,
          "retries": <int>,
          "circuit_state": "<state>",
          "model_id": "<id>",
          "model_version": "<version>",
          "tokens_in": <int>,
          "tokens_out": <int>,
          "cost_usd": <float>,
          "error_samples": [ ... up to 5 ]
        }, ...
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.metrics import EnrichmentMetrics
from podcast_scraper.enrichment.paths import (
    enrichment_run_summary_path,
    ensure_directory,
)

logger = logging.getLogger(__name__)

RUN_SUMMARY_SCHEMA_VERSION = "1"


def build_run_summary(
    *,
    run_id: str,
    parent_run_id: str | None,
    profile: str | None,
    started_at: str,
    finished_at: str,
    duration_ms: int,
    status: str,
    per_enricher: dict[str, EnrichmentMetrics],
) -> dict[str, Any]:
    """Build the run-summary dict from per-enricher metrics."""
    summary_per_enricher: dict[str, dict[str, Any]] = {}
    for enricher_id, m in per_enricher.items():
        summary_per_enricher[enricher_id] = {
            "status": m.last_run_status,
            "duration_ms": int(m.duration_seconds * 1000),
            "records_written": m.output_records_total,
            "retries": m.retries_total,
            "runs_total": m.runs_total,
            "runs_ok": m.runs_ok,
            "runs_failed": m.runs_failed,
            "runs_timeout": m.runs_timeout,
            "runs_quarantined": m.runs_quarantined,
            "runs_cancelled": m.runs_cancelled,
            "runs_skipped": m.runs_skipped,
            "model_id": m.model_id,
            "model_version": m.model_version,
            "tokens_in": m.tokens_in,
            "tokens_out": m.tokens_out,
            "cost_usd": m.cost_usd,
            "error_samples": list(m.error_samples),
        }
    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "profile": profile,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": int(duration_ms),
        "status": status,
        "per_enricher": summary_per_enricher,
    }


def write_run_summary(corpus_root: Path, payload: dict[str, Any]) -> Path:
    """Write ``enrichments/run_summary.json`` atomically; return its path."""
    path = enrichment_run_summary_path(corpus_root)
    _atomic_write_json(path, payload)
    return path


def read_run_summary(corpus_root: Path) -> dict[str, Any] | None:
    """Return the parsed run-summary, or ``None`` if absent / corrupt."""
    path = enrichment_run_summary_path(corpus_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically via tempfile + os.replace."""
    ensure_directory(path.parent)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
