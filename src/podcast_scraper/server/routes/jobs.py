"""POST/GET /api/jobs — opt-in pipeline subprocess jobs (Phase 2)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse

from podcast_scraper.server.jobs_log_path import (
    JobLogPathError,
    read_job_log_tail_utf8 as _read_job_log_tail_utf8,
    resolve_pipeline_job_log_path,
)
from podcast_scraper.server.operator_paths import (
    viewer_operator_extras_source,
    viewer_operator_yaml_path,
)
from podcast_scraper.server.pipeline_docker_factory import validate_operator_pipeline_extras
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
    PipelineJobLogTailResponse,
    PipelineJobReconcileResponse,
    PipelineJobRecord,
    PipelineJobsListResponse,
)

router = APIRouter(tags=["jobs"])


async def _serve_pipeline_job_log(corpus: Path, job_id: str) -> FileResponse:
    """Resolve registry row → log file on disk; same rules for path- and query-style routes."""
    verified_under = await _resolved_job_log_path(corpus, job_id)
    # codeql[py/path-injection] -- verified_under from resolve_pipeline_job_log_path (Type 1).
    return FileResponse(
        verified_under,
        media_type="text/plain; charset=utf-8",
        filename=os.path.basename(verified_under),
    )


async def _resolved_job_log_path(corpus: Path, job_id: str) -> str:
    try:
        return await resolve_pipeline_job_log_path(corpus, job_id)
    except JobLogPathError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


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
    if not bool(getattr(request.app.state, "jobs_api_enabled", False)):
        raise HTTPException(
            status_code=500,
            detail="jobs_api is not enabled.",
        )
    return corpus, viewer_operator_yaml_path(request.app, corpus)


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
    # #666 review #8: read exec mode from ``app.state`` (pinned at startup by
    # ``create_app``) instead of re-reading ``PODCAST_PIPELINE_EXEC_MODE`` here.
    # Re-reading would drift if the env is rotated mid-process.
    pipe_mode = getattr(request.app.state, "pipeline_exec_mode", "")
    # #666 review #13: extras validation is symmetric on *values* — if the
    # operator YAML declares ``pipeline_install_extras`` it must be one of
    # ``ml`` / ``llm`` in both modes. Requiring the field itself stays
    # mode-specific by design: Docker mode uses the declaration to pick a
    # compose service (``pipeline`` vs ``pipeline-llm``) and cannot infer
    # it, whereas subprocess mode uses whatever extras the API image was
    # built with.
    try:
        await asyncio.to_thread(
            validate_operator_pipeline_extras,
            viewer_operator_extras_source(request.app, corpus),
            pipe_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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
    # codeql[py/path-injection] -- corpus from _resolve_corpus_root (anchor-guarded; Type 1).
    return PipelineJobAccepted(
        job_id=str(rec["job_id"]),
        status=str(rec["status"]),
        corpus_path=os.path.normpath(str(corpus.resolve())),
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
    # codeql[py/path-injection] -- corpus from _resolve_corpus_root (anchor-guarded; Type 1).
    return PipelineJobsListResponse(
        path=os.path.normpath(str(corpus.resolve())),
        jobs=jobs,
    )


@router.post("/jobs/reconcile", response_model=PipelineJobReconcileResponse)
async def reconcile_pipeline_jobs(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> PipelineJobReconcileResponse:
    """Reconcile stale/orphan *running* rows (dead PID, wall-clock stale)."""
    corpus, _op = _corpus_and_operator(request, path)
    n, details = await asyncio.to_thread(apply_reconcile, corpus)
    # codeql[py/path-injection] -- corpus from _resolve_corpus_root (anchor-guarded; Type 1).
    return PipelineJobReconcileResponse(
        path=os.path.normpath(str(corpus.resolve())),
        updated=n,
        details=details,
    )


@router.get("/jobs/subprocess-log")
async def get_pipeline_job_log_query(
    request: Request,
    job_id: str = Query(..., description="Pipeline job id (UUID)."),
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> FileResponse:
    """Same as ``GET /jobs/{job_id}/log`` but query-based.

    Avoids some proxy/static 404s on ``…/log``.
    """
    corpus, _op = _corpus_and_operator(request, path)
    return await _serve_pipeline_job_log(corpus, job_id)


@router.get("/jobs/subprocess-log-tail", response_model=PipelineJobLogTailResponse)
async def get_pipeline_job_log_tail_query(
    request: Request,
    job_id: str = Query(..., description="Pipeline job id (UUID)."),
    path: str | None = Query(default=None, description="Corpus output directory."),
    max_bytes: int = Query(
        default=96_000,
        ge=4096,
        le=512_000,
        description="Max bytes read from end of log (UTF-8).",
    ),
) -> PipelineJobLogTailResponse:
    """Same as ``GET /jobs/{job_id}/log-tail`` but query-based (avoids some proxy 404s)."""
    corpus, _op = _corpus_and_operator(request, path)
    verified_under = await _resolved_job_log_path(corpus, job_id)
    # codeql[py/path-injection] -- verified_under from resolve_pipeline_job_log_path (Type 1).
    text, truncated = await asyncio.to_thread(_read_job_log_tail_utf8, verified_under, max_bytes)
    return PipelineJobLogTailResponse(text=text, truncated=truncated)


@router.get("/jobs/{job_id}/log")
async def get_pipeline_job_log(
    request: Request,
    job_id: str,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> FileResponse:
    """Return the job subprocess log as ``text/plain`` (for opening in a new browser tab)."""
    corpus, _op = _corpus_and_operator(request, path)
    return await _serve_pipeline_job_log(corpus, job_id)


@router.get("/jobs/{job_id}/log-tail", response_model=PipelineJobLogTailResponse)
async def get_pipeline_job_log_tail(
    request: Request,
    job_id: str,
    path: str | None = Query(default=None, description="Corpus output directory."),
    max_bytes: int = Query(
        default=96_000,
        ge=4096,
        le=512_000,
        description="Max bytes read from end of log (UTF-8).",
    ),
) -> PipelineJobLogTailResponse:
    """Return the tail of the job log as JSON (for dashboard metrics + summary preview)."""
    corpus, _op = _corpus_and_operator(request, path)
    verified_under = await _resolved_job_log_path(corpus, job_id)
    # codeql[py/path-injection] -- verified_under from resolve_pipeline_job_log_path (Type 1).
    text, truncated = await asyncio.to_thread(_read_job_log_tail_utf8, verified_under, max_bytes)
    return PipelineJobLogTailResponse(text=text, truncated=truncated)


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
