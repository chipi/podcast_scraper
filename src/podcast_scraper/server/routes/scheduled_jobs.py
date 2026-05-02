"""GET /api/scheduled-jobs — list cron schedules + next-run preview (#708).

Read-only. Operators add/remove schedules by editing ``scheduled_jobs:`` in
``viewer_operator.yaml`` via the existing ``PUT /api/operator-config`` (the
PUT handler already triggers a scheduler reload).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.server.routes.index_rebuild import _resolve_corpus_root
from podcast_scraper.server.schemas import (
    ScheduledJobItem,
    ScheduledJobsListResponse,
)

router = APIRouter(tags=["scheduled-jobs"])


def _corpus(request: Request, path: str | None) -> Path:
    anchor = getattr(request.app.state, "output_dir", None)
    corpus = _resolve_corpus_root(path, anchor)
    if corpus is None:
        raise HTTPException(
            status_code=400,
            detail="Corpus path is required (query or server default).",
        )
    if not bool(getattr(request.app.state, "jobs_api_enabled", False)):
        raise HTTPException(status_code=500, detail="jobs_api is not enabled.")
    return corpus


@router.get("/scheduled-jobs", response_model=ScheduledJobsListResponse)
async def list_scheduled_jobs(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> ScheduledJobsListResponse:
    """List configured scheduled jobs and (when running) their next-run preview."""
    corpus = _corpus(request, path)
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        # codeql[py/path-injection] -- corpus from _resolve_corpus_root (anchor-guarded; Type 1).
        return ScheduledJobsListResponse(
            path=os.path.normpath(str(corpus.resolve())),
            scheduler_running=False,
            timezone="UTC",
            jobs=[],
        )

    def _snapshot() -> ScheduledJobsListResponse:
        items: list[ScheduledJobItem] = []
        for cfg in scheduler.jobs:
            items.append(
                ScheduledJobItem(
                    name=cfg.name,
                    cron=cfg.cron,
                    enabled=cfg.enabled,
                    next_run_at=scheduler.next_run_at(cfg.name) if cfg.enabled else None,
                )
            )
        # codeql[py/path-injection] -- corpus from _resolve_corpus_root (anchor-guarded; Type 1).
        return ScheduledJobsListResponse(
            path=os.path.normpath(str(corpus.resolve())),
            scheduler_running=scheduler.running,
            timezone=scheduler.timezone,
            jobs=items,
        )

    return await asyncio.to_thread(_snapshot)
