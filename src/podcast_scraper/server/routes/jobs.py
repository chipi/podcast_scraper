"""POST/GET /api/jobs — opt-in pipeline subprocess jobs (RFC-077 Phase 2)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Query, Request, status

from podcast_scraper.server.pipeline_jobs import (
    apply_reconcile,
    cancel_job,
    enqueue_pipeline_job,
    get_job,
    list_jobs_snapshot,
    schedule_post_submit,
)
from podcast_scraper.server.routes.index_rebuild import _resolve_corpus_root
from podcast_scraper.server.schemas import (
    PipelineJobAccepted,
    PipelineJobReconcileResponse,
    PipelineJobRecord,
    PipelineJobsListResponse,
)

router = APIRouter(tags=["jobs"])


async def _kickoff_job(app: FastAPI, corpus: Path, rec: dict) -> None:
    await schedule_post_submit(app, corpus, rec)


def _corpus_and_operator(request: Request, path: str | None) -> tuple[Path, Path]:
    anchor = getattr(request.app.state, "output_dir", None)
    corpus = _resolve_corpus_root(path, anchor)
    if corpus is None:
        raise HTTPException(
            status_code=400,
            detail="Corpus path is required (query or server default).",
        )
    raw_op = getattr(request.app.state, "operator_config_path", None)
    if raw_op is None:
        raise HTTPException(
            status_code=500,
            detail="operator_config_path is not configured for pipeline jobs.",
        )
    return corpus, Path(raw_op)


@router.post(
    "/jobs",
    response_model=PipelineJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_pipeline_job(
    request: Request,
    background_tasks: BackgroundTasks,
    path: str | None = Query(
        default=None,
        description="Corpus output directory (same anchor rules as other viewer routes).",
    ),
) -> PipelineJobAccepted:
    """Queue a pipeline CLI job for the corpus (202 + optional queue position)."""
    corpus, operator_yaml = _corpus_and_operator(request, path)
    rec = await asyncio.to_thread(enqueue_pipeline_job, corpus, operator_yaml)
    background_tasks.add_task(_kickoff_job, request.app, corpus, rec)
    qp = None
    if rec.get("status") == "queued":
        snap = await asyncio.to_thread(list_jobs_snapshot, corpus)
        for row in snap:
            if str(row.get("job_id")) == str(rec.get("job_id")):
                qp = row.get("queue_position")
                if isinstance(qp, int):
                    break
                qp = None
    return PipelineJobAccepted(
        job_id=str(rec["job_id"]),
        status=str(rec["status"]),
        corpus_path=str(corpus.resolve()),
        queue_position=qp,
    )


@router.get("/jobs", response_model=PipelineJobsListResponse)
async def list_pipeline_jobs(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> PipelineJobsListResponse:
    """List jobs for the corpus from the JSONL registry."""
    corpus, _op = _corpus_and_operator(request, path)
    rows = await asyncio.to_thread(list_jobs_snapshot, corpus)
    jobs = [PipelineJobRecord.model_validate(r) for r in rows]
    return PipelineJobsListResponse(path=str(corpus.resolve()), jobs=jobs)


@router.post("/jobs/reconcile", response_model=PipelineJobReconcileResponse)
async def reconcile_pipeline_jobs(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> PipelineJobReconcileResponse:
    """Reconcile stale/orphan *running* rows (dead PID, wall-clock stale)."""
    corpus, _op = _corpus_and_operator(request, path)
    n, details = await asyncio.to_thread(apply_reconcile, corpus)
    return PipelineJobReconcileResponse(path=str(corpus.resolve()), updated=n, details=details)


@router.get("/jobs/{job_id}", response_model=PipelineJobRecord)
async def get_pipeline_job(
    request: Request,
    job_id: str,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> PipelineJobRecord:
    """Return a single job by id (404 when missing)."""
    corpus, _op = _corpus_and_operator(request, path)
    rec = await asyncio.to_thread(get_job, corpus, job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return PipelineJobRecord.model_validate(rec)


@router.post("/jobs/{job_id}/cancel", response_model=PipelineJobRecord)
async def cancel_pipeline_job(
    request: Request,
    job_id: str,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> PipelineJobRecord:
    """Cancel a queued job or signal SIGTERM for a running child (idempotent if terminal)."""
    corpus, _op = _corpus_and_operator(request, path)
    outcome, rec = await asyncio.to_thread(cancel_job, corpus, job_id)
    if outcome == "not_found" or rec is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return PipelineJobRecord.model_validate(rec)
