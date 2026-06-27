"""Enrichment-layer HTTP routes (RFC-088 / Epic #1101 chunk 1 sub-6).

Six routes total:

* ``POST /api/jobs/enrichment`` — enqueue a corpus_enrichment job
  (mirrors ``POST /api/jobs`` for pipeline jobs).
* ``GET  /api/enrichment/status`` — current live status (reads
  ``.viewer/enrichment_status.json``).
* ``GET  /api/enrichment/health`` — per-enricher health snapshot
  (reads ``.viewer/enrichment_health.json``); ``?enricher_id=…``
  filter narrows to one.
* ``POST /api/enrichment/health/{enricher_id}/re-enable`` — operator
  manual recovery; mirrors ``podcast enrich --re-enable``.
* ``GET  /api/enrichment/metrics`` — windowed metric snapshot
  (chunk-1 ships the route + empty payload; chunks 3–4 populate via
  ``data/eval/enrichment/<id>/history.jsonl``; chunk 6 viewer panel
  consumes).
* ``GET  /api/enrichment/run-summary`` — most recent
  ``enrichments/run_summary.json`` (or ``{}`` when none yet).
* ``GET  /api/enrichment/events`` — JSONL event slice from
  ``enrichments/run.jsonl`` (filter by ``enricher_id`` / ``event_type``
  / ``limit``).

The ``podcast_obs/sources/enrichment.py`` MCP source (sub-7) reads
these routes via httpx; the viewer Operator-tab Enrichment panel
(chunk 6) consumes the same surface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from podcast_scraper.enrichment.events import append_event, build_health_re_enabled
from podcast_scraper.enrichment.health import HealthRegistry
from podcast_scraper.enrichment.paths import enrichment_run_summary_path
from podcast_scraper.enrichment.status import read_status
from podcast_scraper.server.jobs import (
    COMMAND_ENRICHMENT,
    enqueue_enrichment_job,
    list_jobs_snapshot,
    schedule_post_submit,
)
from podcast_scraper.server.operator_paths import viewer_operator_yaml_path
from podcast_scraper.server.routes.index_rebuild import _resolve_corpus_root

logger = logging.getLogger(__name__)


router = APIRouter(tags=["enrichment"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EnrichmentJobRequest(BaseModel):
    """Body for ``POST /api/jobs/enrichment``."""

    only: list[str] | None = Field(default=None, description="Enricher ids to include.")
    skip: list[str] | None = Field(default=None, description="Enricher ids to skip.")
    corpus_only: bool = Field(default=False, description="Skip the episode-scope phase.")


class EnrichmentJobAccepted(BaseModel):
    job_id: str
    status: str
    corpus_path: str
    queue_position: int | None = None


class HealthReEnableRequest(BaseModel):
    reason: str = Field(default="manual re_enable via API", min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _corpus_and_operator(request: Request, path: str | None) -> tuple[Path, Path | None]:
    """Resolve corpus root + operator YAML path the same way as pipeline routes."""
    anchor = getattr(request.app.state, "output_dir", None)
    corpus = _resolve_corpus_root(path, anchor)
    if corpus is None:
        raise HTTPException(
            status_code=400,
            detail="Corpus path is required (query or server default).",
        )
    if not bool(getattr(request.app.state, "jobs_api_enabled", False)):
        raise HTTPException(status_code=500, detail="jobs_api is not enabled.")
    try:
        operator_yaml = viewer_operator_yaml_path(request.app, corpus)
    except Exception:  # pragma: no cover - same defensive pattern as pipeline route
        operator_yaml = None
    return corpus, operator_yaml


async def _kickoff_job(request: Request, corpus: Path, rec: dict[str, Any]) -> None:
    await schedule_post_submit(request.app, corpus, rec)


# ---------------------------------------------------------------------------
# POST /api/jobs/enrichment
# ---------------------------------------------------------------------------


@router.post(
    "/jobs/enrichment",
    response_model=EnrichmentJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_enrichment_job(
    request: Request,
    body: EnrichmentJobRequest | None = None,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> EnrichmentJobAccepted:
    """Enqueue a ``corpus_enrichment`` job (202 + optional queue position)."""
    corpus, operator_yaml = _corpus_and_operator(request, path)
    body = body or EnrichmentJobRequest()
    rec = await asyncio.to_thread(
        enqueue_enrichment_job,
        corpus,
        only=body.only,
        skip=body.skip,
        corpus_only=body.corpus_only,
        operator_yaml=operator_yaml,
    )
    # Kickoff in background (same idiom as pipeline route).
    asyncio.create_task(_kickoff_job(request, corpus, rec))
    qp = None
    if rec.get("status") == "queued":
        snap = await asyncio.to_thread(list_jobs_snapshot, corpus)
        for row in snap:
            if str(row.get("job_id")) == str(rec.get("job_id")):
                qp = row.get("queue_position")
                if isinstance(qp, int):
                    break
                qp = None
    return EnrichmentJobAccepted(
        job_id=str(rec["job_id"]),
        status=str(rec["status"]),
        corpus_path=os.path.normpath(str(corpus.resolve())),
        queue_position=qp,
    )


# ---------------------------------------------------------------------------
# GET /api/enrichment/status
# ---------------------------------------------------------------------------


@router.get("/enrichment/status")
async def get_enrichment_status(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> dict[str, Any]:
    """Return the current live status (``{}`` when the file is absent / corrupt)."""
    corpus, _op = _corpus_and_operator(request, path)
    payload = await asyncio.to_thread(read_status, corpus)
    return payload or {"available": False, "reason": "no status yet"}


# ---------------------------------------------------------------------------
# GET /api/enrichment/health
# ---------------------------------------------------------------------------


@router.get("/enrichment/health")
async def get_enrichment_health(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
    enricher_id: str | None = Query(default=None, description="Narrow to a single enricher id."),
) -> dict[str, Any]:
    """Per-enricher health (auto_disabled, circuit_state, consecutive_failures)."""
    corpus, _op = _corpus_and_operator(request, path)
    registry = HealthRegistry(corpus)
    await asyncio.to_thread(registry.load)
    snap = registry.all()
    if enricher_id is not None:
        h = snap.get(enricher_id)
        if h is None:
            return {"available": False, "reason": f"no record for {enricher_id!r}"}
        return {"enricher_id": enricher_id, **_health_to_dict(h)}
    return {
        "enrichers": {eid: _health_to_dict(h) for eid, h in snap.items()},
    }


# ---------------------------------------------------------------------------
# POST /api/enrichment/health/{enricher_id}/re-enable
# ---------------------------------------------------------------------------


@router.post("/enrichment/health/{enricher_id}/re-enable")
async def re_enable_enricher_route(
    enricher_id: str,
    request: Request,
    body: HealthReEnableRequest | None = None,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> dict[str, Any]:
    """Operator manual recovery — same effect as ``podcast enrich --re-enable``."""
    corpus, _op = _corpus_and_operator(request, path)
    body = body or HealthReEnableRequest()
    registry = HealthRegistry(corpus)
    await asyncio.to_thread(registry.load)
    record = registry.re_enable(enricher_id, reason=body.reason, clear_cooldown=True)
    await asyncio.to_thread(registry.save)
    # Audit-trail: append the manual-recovery event to the JSONL log so the
    # MCP enrichment_recent_events tool surfaces it alongside the executor's
    # own events. Best-effort — never break the response on log-write failure.
    jsonl_path = corpus / "enrichments" / "run.jsonl"
    try:
        await asyncio.to_thread(
            append_event,
            jsonl_path,
            build_health_re_enabled(
                enricher_id=enricher_id,
                operator_id=None,
                reset_counter=True,
                cleared_cooldown=True,
                reason=body.reason,
            ),
        )
    except OSError as exc:
        logger.warning("enrichment re_enable event append failed: %s", exc)
    return {"enricher_id": enricher_id, **_health_to_dict(record)}


# ---------------------------------------------------------------------------
# GET /api/enrichment/metrics
# ---------------------------------------------------------------------------


@router.get("/enrichment/metrics")
async def get_enrichment_metrics(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
    enricher_id: str | None = Query(default=None, description="Narrow to a single enricher id."),
    window: str = Query(default="24h", description="Time window (e.g. 1h / 24h / 7d)."),
) -> dict[str, Any]:
    """Return per-enricher windowed metrics.

    Chunk 1 ships the route + empty payload (no metrics history yet);
    chunk 6 viewer consumes; chunks 2-4 populate as enrichers ship.
    """
    corpus, _op = _corpus_and_operator(request, path)
    # The current run_summary provides the most-recent shape;
    # window/aggregation comes in a later chunk.
    summary_payload = await asyncio.to_thread(_read_run_summary, corpus)
    out: dict[str, Any] = {
        "window": window,
        "per_enricher": {},
    }
    per_enricher = (summary_payload or {}).get("per_enricher") or {}
    if enricher_id is not None:
        per_enricher = (
            {enricher_id: per_enricher[enricher_id]} if enricher_id in per_enricher else {}
        )
    out["per_enricher"] = per_enricher
    return out


# ---------------------------------------------------------------------------
# GET /api/enrichment/run-summary
# ---------------------------------------------------------------------------


@router.get("/enrichment/run-summary")
async def get_enrichment_run_summary(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> dict[str, Any]:
    corpus, _op = _corpus_and_operator(request, path)
    payload = await asyncio.to_thread(_read_run_summary, corpus)
    return payload or {"available": False, "reason": "no run yet"}


# ---------------------------------------------------------------------------
# GET /api/enrichment/events
# ---------------------------------------------------------------------------


@router.get("/enrichment/events")
async def get_enrichment_events(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
    enricher_id: str | None = Query(default=None, description="Filter by enricher_id."),
    event_type: str | None = Query(default=None, description="Filter by event_type."),
    limit: int = Query(default=50, ge=1, le=500, description="Max events to return."),
) -> dict[str, Any]:
    """Return a slice of the JSONL event stream (most recent first).

    Reads ``enrichments/run.jsonl`` directly. Missing file → empty list.
    """
    corpus, _op = _corpus_and_operator(request, path)
    events = await asyncio.to_thread(_read_events_tail, corpus, limit, enricher_id, event_type)
    return {"events": events, "count": len(events)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _health_to_dict(h: Any) -> dict[str, Any]:
    """Convert an ``EnricherHealth`` dataclass to a serialisable dict."""
    from dataclasses import asdict

    return asdict(h)


def _read_run_summary(corpus_root: Path) -> dict[str, Any] | None:
    """Defensive read for the run-summary file (None on missing/corrupt)."""
    path = enrichment_run_summary_path(corpus_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _read_events_tail(
    corpus_root: Path,
    limit: int,
    enricher_id: str | None,
    event_type: str | None,
) -> list[dict[str, Any]]:
    """Read the JSONL event stream tail, filter by ``enricher_id`` / ``event_type``."""
    path = corpus_root / "enrichments" / "run.jsonl"
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    out: list[dict[str, Any]] = []
    # Walk newest-first to populate up to ``limit`` matching events.
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if enricher_id is not None and event.get("enricher_id") != enricher_id:
            continue
        if event_type is not None and event.get("event_type") != event_type:
            continue
        out.append(event)
        if len(out) >= limit:
            break
    return out


__all__ = [
    "COMMAND_ENRICHMENT",
    "EnrichmentJobAccepted",
    "EnrichmentJobRequest",
    "HealthReEnableRequest",
    "router",
]
