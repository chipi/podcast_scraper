"""POST/GET /api/jobs — opt-in pipeline subprocess jobs (Phase 2)."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse

from podcast_scraper.server.jobs import (
    apply_reconcile,
    cancel_job,
    enqueue_pipeline_job,
    get_job,
    list_jobs_snapshot,
    normalize_pipeline_stage,
    PIPELINE_STAGES_ALLOWED,
    schedule_post_submit,
    STATUS_RUNNING,
)
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
from podcast_scraper.server.profile_presets import (
    validate_operator_profile_allowed,
    validate_profile_name_allowed,
)
from podcast_scraper.server.queue_sweeper import (
    DEFAULT_SWEEP_INTERVAL_SECONDS,
    drain_is_paused,
    pause_drain,
    resume_drain,
)
from podcast_scraper.server.routes.index_rebuild import _resolve_corpus_root
from podcast_scraper.server.schemas import (
    PipelineJobAccepted,
    PipelineJobLogTailResponse,
    PipelineJobReconcileResponse,
    PipelineJobRecord,
    PipelineJobsListResponse,
    PipelineJobsStopResponse,
    PipelineQueueResumeResponse,
)

# Markers that identify a provider-secret error from Config validation.
# Only these are surfaced as 400; all other validation errors are ignored
# so a valid-at-runtime config is never wrongly rejected at submit time.
_PROVIDER_SECRET_RE = re.compile(
    r"API key required|key required|environment variable", re.IGNORECASE
)


def _feed_pinned_profiles(corpus: Path, feed_url: str | None) -> list[str]:
    """Profiles pinned in feeds.spec.yaml that THIS run will resolve.

    A single-feed run resolves only that feed's pin; a batch resolves all of them. Best
    effort — a malformed spec must not block submission, it simply means the pins go
    unchecked exactly as they did before (#1874 W6).
    """
    try:
        from podcast_scraper.rss.feeds_spec import load_feeds_spec_file

        spec = corpus / "feeds.spec.yaml"
        if not spec.is_file():
            return []
        from podcast_scraper.server.jobs import _normalise_feed_url

        entries = load_feeds_spec_file(str(spec)).feeds
        if feed_url:
            # Same normalisation as the argv path — a trailing-slash difference silently
            # skipped the pin's secret check while the run still used that pin.
            wanted = _normalise_feed_url(feed_url)
            entries = [e for e in entries if _normalise_feed_url(e.url) == wanted]
        return [p for e in entries if (p := (getattr(e, "profile", None) or "").strip())]
    except Exception:  # noqa: BLE001 — never block submit on spec parsing
        return []


def _check_provider_secrets(
    operator_yaml: Path,
    *,
    profile_override: str | None = None,
    extra_profiles: Sequence[str] | None = None,
) -> None:
    """Raise ValueError when a profile this run will use is missing a provider secret.

    Runs Config.model_validate against the resolved operator YAML (including
    its profile presets) so the same validation the pipeline CLI does at
    startup fires eagerly at submit time — no container spawned.

    #1874 W6: the run's profile is no longer only the corpus YAML's. A per-request
    override (#1872) and per-feed pins in feeds.spec.yaml both change which providers a
    run needs, and neither reached this check — so the documented guarantee ("missing
    secret → 400, no container spawned") did not hold for exactly the two layers that were
    added. A batch feed-pin failure was the worst case: no validation anywhere, so it died
    mid-run AFTER earlier feeds had already spent money. Each additional profile is checked
    the same way, against the same operator YAML body.

    Scoped: only provider-secret ValidationErrors ("API key required", …) are
    re-raised. Any other validation error is silently swallowed so that a
    config that is valid at runtime (rss/output_dir supplied by CLI flags,
    etc.) is never wrongly rejected at submit time.
    """
    try:
        from podcast_scraper.config import load_config_file
    except ImportError:  # pragma: no cover — always available in api process
        return

    try:
        data = load_config_file(str(operator_yaml))
    except (ValueError, OSError):
        # Missing file / parse error — not our job to reject here; let the
        # pipeline CLI handle it with its richer error messages.
        return

    # The corpus YAML's own profile, then every OTHER profile this run may resolve.
    variants: list[dict[str, Any]] = [dict(data)]
    seen: set[str] = set()
    for name in [profile_override, *(extra_profiles or [])]:
        key = (name or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        variant = dict(data)
        variant["profile"] = key
        variants.append(variant)

    for payload in variants:
        _validate_one_profile_payload(payload)


def _validate_one_profile_payload(data: dict[str, Any]) -> None:
    """Provider-secret check for ONE resolved config payload (see _check_provider_secrets)."""
    try:
        from podcast_scraper.config import Config
    except ImportError:  # pragma: no cover
        return

    try:
        import pydantic

        Config.model_validate(data)
    except (pydantic.ValidationError, ValueError) as exc:
        msg = str(exc)
        if _PROVIDER_SECRET_RE.search(msg):
            # Extract just the first matching error line to keep the 400 terse.
            for line in msg.splitlines():
                if _PROVIDER_SECRET_RE.search(line):
                    raise ValueError(line.strip()) from exc
            raise ValueError(msg) from exc
        # Not a provider-secret error; let the pipeline fail at runtime with
        # its own clear message (e.g. missing rss_url, which is CLI-injected).
    except Exception:  # noqa: BLE001 — defensive; unexpected validators must not block submit
        pass


router = APIRouter(tags=["jobs"])

_EPISODE_ORDERS = {"newest", "oldest"}


def _resolve_feed_url(corpus: Path, feed: str) -> str:
    """Resolve a ``feed`` scope (raw URL or stable feed slug) to its RSS URL (P1.4).

    A value containing ``://`` is taken as the URL verbatim; otherwise it's a
    ``feed_workspace_dirname`` slug matched against the corpus ``feeds.spec.yaml``. 404 if a slug
    matches no known feed — refuse to silently run the whole batch when one feed was asked for.
    """
    raw = feed.strip()
    if "://" in raw:
        return raw
    from podcast_scraper.rss.feeds_spec import FEEDS_SPEC_DEFAULT_BASENAME, load_feeds_spec_file
    from podcast_scraper.utils.filesystem import feed_workspace_dirname

    spec_path = corpus / FEEDS_SPEC_DEFAULT_BASENAME
    # codeql[py/path-injection] -- request path anchor-guarded (Type 1; CODEQL_DISMISSALS.md).
    if spec_path.is_file():
        try:
            doc = load_feeds_spec_file(spec_path)
        except Exception as exc:  # noqa: BLE001 — a malformed spec is a 400, not a 500
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not read {FEEDS_SPEC_DEFAULT_BASENAME}: {exc}",
            ) from exc
        for entry in doc.feeds:
            if entry.url == raw or feed_workspace_dirname(entry.url) == raw:
                return entry.url
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"feed {feed!r} not found in {FEEDS_SPEC_DEFAULT_BASENAME}.",
    )


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
    feed: str | None = Query(
        default=None,
        description="Scope the run to ONE feed (RSS URL or feed slug); omit for the whole batch.",
    ),
    skip_existing: bool = Query(
        default=False, description="Per-feed only: skip episodes already present (guid-keyed)."
    ),
    append: bool = Query(
        default=False, description="Per-feed only: append mode (episode_id-validated resume)."
    ),
    max_episodes: int | None = Query(
        default=None, ge=1, description="Per-feed only: cap episodes this run (cost guardrail)."
    ),
    episode_offset: int | None = Query(
        default=None, ge=0, description="Per-feed only: skip the newest N before selecting."
    ),
    episode_order: str | None = Query(
        default=None, description="Per-feed only: 'newest' or 'oldest'."
    ),
    profile: str | None = Query(
        default=None,
        description=(
            "Run THIS job on a named profile, overriding both the feed's pinned profile and "
            "the corpus operator YAML (#1872). Scoped to this run; nothing is persisted."
        ),
    ),
    pipeline_stage: str | None = Query(
        default=None,
        description=(
            "Run only part of the pipeline instead of the full ingest. REPROCESS modes reuse "
            "what is already on disk and never re-run ASR: 'rederive_only' (re-run the LLM "
            "stages — cleaning + GI + KG — from the existing transcript; use after a prompt or "
            "model change), 'relabel_only' (re-resolve speaker names on the frozen "
            "diarization), 'rediarize_only' (re-diarize the audio, aligned to the existing ASR "
            "text). PARTIAL modes: 'audio_only', 'download_only'. Reprocess modes are scoped to "
            "episodes already in the corpus automatically. Omit (or send 'full') for a normal "
            "ingest. 'enrich_only' is accepted as a deprecated alias of 'rederive_only'."
        ),
    ),
) -> PipelineJobAccepted:
    """Queue a pipeline CLI job for the corpus (202 + optional queue position).

    Default: the whole ``feeds.spec.yaml`` batch. With ``feed=`` the run is scoped to that one feed
    as a single-feed corpus-layout run plus the incremental knobs (P1.4) — cautious per-feed add.
    """
    corpus, operator_yaml = _corpus_and_operator(request, path)
    feed_url: str | None = None
    if feed is not None and feed.strip():
        feed_url = _resolve_feed_url(corpus, feed)
    if episode_order is not None and episode_order not in _EPISODE_ORDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"episode_order must be one of {sorted(_EPISODE_ORDERS)}.",
        )
    # Reject an unknown stage HERE with a 400 rather than letting build_pipeline_argv drop it
    # with a warning. Silently ignoring it would start a FULL ingest for a caller who asked for
    # a cheap reprocess — the expensive direction of the mistake, and invisible in the response.
    if pipeline_stage is not None and str(pipeline_stage).strip():
        requested = str(pipeline_stage).strip()
        if requested != "full" and normalize_pipeline_stage(requested) is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "pipeline_stage must be one of "
                    f"{sorted(PIPELINE_STAGES_ALLOWED)} (or 'full'/omitted for a normal run)."
                ),
            )
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
    # Defense-in-depth for #692 / RFC-081 §Layer 1: even if a stale viewer
    # bundle picked a profile hidden from the dropdown, refuse to enqueue
    # a pipeline run that would crash several minutes in (the chosen
    # profile's image isn't published in this env). ``operator_yaml`` is
    # the same path the run will execute against — single source of truth.
    try:
        await asyncio.to_thread(validate_operator_profile_allowed, operator_yaml)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    # #1872: the per-request override gets the SAME allowlist gate before anything is
    # enqueued. An unknown name must fail here rather than reach argv, where
    # Config._resolve_profile merely warns and runs on defaults — a job that looks
    # configured and is not is the failure mode this feature must not introduce.
    try:
        profile_override = await asyncio.to_thread(validate_profile_name_allowed, profile)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    # Preflight: validate the profile's required provider secrets are present
    # in the api process env. A missing secret → 400 with the missing key named,
    # no container spawned. Only provider-secret errors are surfaced (see
    # _check_provider_secrets); unrelated fields that CLI injects at runtime
    # (rss_url, output_dir, …) are not checked.
    # #1874 W6: validate every profile this run can resolve — the corpus YAML's, the
    # per-request override, and (for a batch) each feed's pin. A pinned feed missing a key
    # otherwise reaches no validation at all and dies mid-run, after earlier feeds have spent.
    # Pins are irrelevant when the operator overrode the profile for this run: the merge
    # ignores every pin in that case (profile_overrides_feed_pins), so validating them would
    # reject a legitimate "reprocess once on cloud_thin" because some feed's pin happens to
    # reference a provider whose key is absent — a 400 for a profile that will never run.
    pinned_profiles: list[str] = (
        [] if profile_override else await asyncio.to_thread(_feed_pinned_profiles, corpus, feed_url)
    )
    try:
        await asyncio.to_thread(
            _check_provider_secrets,
            operator_yaml,
            profile_override=profile_override,
            extra_profiles=pinned_profiles,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    rec = await asyncio.to_thread(
        enqueue_pipeline_job,
        corpus,
        operator_yaml,
        feed_url=feed_url,
        skip_existing=skip_existing,
        append=append,
        max_episodes=max_episodes,
        episode_offset=episode_offset,
        episode_order=episode_order,
        profile_override=profile_override,
        pipeline_stage=pipeline_stage,
    )
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


@router.get("/jobs/running", response_model=PipelineJobsListResponse)
async def list_running_pipeline_jobs(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> PipelineJobsListResponse:
    """What is executing right now (#1785) — the view that used to need ``docker ps`` over SSH.

    The corpus API cannot answer this (it serves newest-run-per-episode, so an in-flight run
    is invisible until episodes complete); the job registry can.
    """
    corpus, _op = _corpus_and_operator(request, path)
    rows = await asyncio.to_thread(list_jobs_snapshot, corpus)
    running = [PipelineJobRecord.model_validate(r) for r in rows if r.get("status") == "running"]
    # codeql[py/path-injection] -- corpus from _resolve_corpus_root (anchor-guarded; Type 1).
    return PipelineJobsListResponse(path=os.path.normpath(str(corpus.resolve())), jobs=running)


@router.post("/jobs/stop", response_model=PipelineJobsStopResponse)
async def stop_all_pipeline_jobs(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
    verify_seconds: float = Query(
        default=DEFAULT_SWEEP_INTERVAL_SECONDS,
        ge=0,
        le=120,
        description=(
            "How long to wait before re-checking that nothing survived. Defaults to one full "
            "sweep interval. 0 skips verification (the response then reports every stopped job "
            "as a survivor candidate unverified)."
        ),
    ),
) -> PipelineJobsStopResponse:
    """The emergency brake (#1785): hold the queue, SIGTERM running work, verify, report.

    ORDER MATTERS — the pause flag is set BEFORE any signal. The sweeper promotes queued work
    on a 30s loop, so signalling first just yields the freed slot to the next queued job and
    makes the stop look like it failed (the 2026-08-18 incident). Queued jobs are NOT
    cancelled: the pause holds them, and releasing is the operator's decision (``resume``).

    SIGTERM with grace rather than SIGKILL: the pipeline flushes in-flight provider cost on
    TERM; a KILL would make that spend invisible after the fact.
    """
    corpus, _op = _corpus_and_operator(request, path)
    # 1. Hold the queue FIRST.
    await asyncio.to_thread(pause_drain, corpus)
    # 2. Signal every running job through the same path the single-job cancel uses (prior-boot
    #    rows go through the docker-label stop; this-boot rows get SIGTERM on their pid).
    rows = await asyncio.to_thread(list_jobs_snapshot, corpus)
    running_ids = [str(r["job_id"]) for r in rows if r.get("status") == STATUS_RUNNING]
    stopped: list[PipelineJobRecord] = []
    for job_id in running_ids:
        _outcome, rec = await asyncio.to_thread(cancel_job, corpus, job_id)
        if rec is not None:
            stopped.append(PipelineJobRecord.model_validate(rec))
    # 3. Verify: after a wait, anything still running with a live pid survived the brake.
    survivors: list[PipelineJobRecord] = []
    if verify_seconds > 0 and running_ids:
        await asyncio.sleep(verify_seconds)
        from podcast_scraper.server.jobs import pid_alive

        recheck = await asyncio.to_thread(list_jobs_snapshot, corpus)
        survivors = [
            PipelineJobRecord.model_validate(r)
            for r in recheck
            if str(r.get("job_id")) in set(running_ids)
            and r.get("status") == STATUS_RUNNING
            and pid_alive(r.get("pid"))
        ]
    # codeql[py/path-injection] -- corpus from _resolve_corpus_root (anchor-guarded; Type 1).
    return PipelineJobsStopResponse(
        path=os.path.normpath(str(corpus.resolve())),
        queue_paused=True,
        stopped=stopped,
        survivors=survivors,
        all_stopped=not survivors,
        verified_after_seconds=verify_seconds if running_ids else 0.0,
    )


@router.post("/jobs/resume", response_model=PipelineQueueResumeResponse)
async def resume_pipeline_queue(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output directory."),
) -> PipelineQueueResumeResponse:
    """Release the hold the stop endpoint set — a brake that cannot be released is a defect."""
    corpus, _op = _corpus_and_operator(request, path)
    await asyncio.to_thread(resume_drain, corpus)
    # codeql[py/path-injection] -- corpus from _resolve_corpus_root (anchor-guarded; Type 1).
    return PipelineQueueResumeResponse(
        path=os.path.normpath(str(corpus.resolve())),
        queue_paused=drain_is_paused(corpus),
    )


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
