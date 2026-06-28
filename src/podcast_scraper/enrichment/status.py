"""``.viewer/enrichment_status.json`` live-status writer.

Mirrors the shape of ``monitor/status.py`` (pipeline_status.json) so
the viewer Operator-tab Enrichment panel polls it the same way and
the CLI Rich-progress display can subscribe via the same idiom.

Atomic-write semantics (temp + ``os.replace``); ``.viewer/`` created
if missing.

The status writer is **stateless** — every update reads the current
state, mutates it, and writes back. No long-lived in-memory state
needed; the executor calls ``update_status(...)`` between batches.

See ``docs/wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md``
§"Live status".
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.envelope import utc_iso_now
from podcast_scraper.enrichment.paths import (
    enrichment_status_path,
    ensure_directory,
    viewer_dir,
)

logger = logging.getLogger(__name__)

STATUS_SCHEMA_VERSION = "1"


def write_status(
    corpus_root: Path,
    *,
    run_id: str,
    started_at: str,
    profile: str | None,
    current_enricher: dict[str, Any] | None,
    queue: list[str],
    completed: list[dict[str, Any]],
) -> Path:
    """Atomically write the live-status file.

    Returns the path written. Creates ``.viewer/`` if missing (chunk-1
    lock audit §B6).

    Schema:

    ::

        {
          "schema_version": "1",
          "run_id": "<job-id>",
          "started_at": "<iso>",
          "profile": "<profile name or null>",
          "current_enricher": {
            "enricher_id": "<id>",
            "scope": "episode" | "corpus",
            "tier": "<tier>",
            "attempt": <int>,
            "progress": {
              "items_done": <int>,
              "items_total": <int>,
              "eta_seconds": <int or null>
            },
            "last_heartbeat_at": "<iso>"
          } | null,
          "queue": ["<enricher_id>", ...],
          "completed": [
            {"enricher_id": "<id>", "status": "<status>", "duration_ms": <int>}, ...
          ]
        }
    """
    ensure_directory(viewer_dir(corpus_root))
    payload: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "profile": profile,
        "current_enricher": current_enricher,
        "queue": list(queue),
        "completed": list(completed),
    }
    path = enrichment_status_path(corpus_root)
    _atomic_write_json(path, payload)
    return path


def write_idle(corpus_root: Path) -> Path:
    """Write the idle (between-runs) status.

    Operator-tab Enrichment panel renders ``"idle"`` when
    ``current_enricher == null`` and ``queue + completed`` are empty.
    """
    return write_status(
        corpus_root,
        run_id="",
        started_at=utc_iso_now(),
        profile=None,
        current_enricher=None,
        queue=[],
        completed=[],
    )


def read_status(corpus_root: Path) -> dict[str, Any] | None:
    """Return the parsed status JSON, or ``None`` if absent / corrupt.

    Mirrors ``monitor/status.read_pipeline_status`` defensive-read shape.
    """
    path = enrichment_status_path(corpus_root)
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None


def build_current_enricher_block(
    *,
    enricher_id: str,
    scope: str,
    tier: str,
    attempt: int,
    items_done: int,
    items_total: int | None,
    eta_seconds: int | None = None,
    last_heartbeat_at: str | None = None,
) -> dict[str, Any]:
    """Construct the ``current_enricher`` sub-dict.

    Lifts the awkward nested-dict construction out of the executor so
    callers compose a flat keyword call.
    """
    return {
        "enricher_id": enricher_id,
        "scope": scope,
        "tier": tier,
        "attempt": int(attempt),
        "progress": {
            "items_done": int(items_done),
            "items_total": int(items_total) if items_total is not None else None,
            "eta_seconds": int(eta_seconds) if eta_seconds is not None else None,
        },
        "last_heartbeat_at": last_heartbeat_at or utc_iso_now(),
    }


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
