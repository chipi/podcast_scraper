"""Pipeline job queue, subprocess spawn, and registry updates (Phase 2)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, cast, Sequence

from podcast_scraper.rss.feeds_spec import FEEDS_SPEC_DEFAULT_BASENAME
from podcast_scraper.server.operator_paths import viewer_operator_yaml_path
from podcast_scraper.server.operator_yaml_profile import split_operator_yaml_profile
from podcast_scraper.server.pipeline_job_registry import (
    with_jobs_locked_mutate,
    with_jobs_locked_read,
)

logger = logging.getLogger(__name__)

COMMAND_FULL = "full_incremental_pipeline"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_STALE = "stale"

TERMINAL = frozenset(
    {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED, STATUS_STALE},
)

JobsSubprocessFactory = Callable[
    [Sequence[str], Path, Path],
    Awaitable[asyncio.subprocess.Process],
]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def max_concurrent_jobs() -> int:
    """Max concurrent *running* jobs per corpus (default 1)."""
    raw = os.environ.get("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "1").strip()
    try:
        n = int(raw)
    except ValueError:
        return 1
    return max(1, n)


def stale_after_seconds() -> int:
    """Wall-clock stale threshold for *running* jobs during reconcile."""
    raw = os.environ.get("PODCAST_JOB_STALE_SECONDS", str(86400)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 86400


def pid_alive(pid: int | None) -> bool:
    """Return True if ``pid`` responds to ``kill(..., 0)``."""
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def build_pipeline_argv(corpus_root: Path, operator_yaml: Path) -> list[str]:
    """Build CLI argv for a full pipeline run (README parity: ``--profile`` then ``--config``)."""
    exe = sys.executable
    argv: list[str] = [exe, "-m", "podcast_scraper.cli", "--output-dir", str(corpus_root)]
    try:
        op_text = operator_yaml.read_text(encoding="utf-8", errors="replace")
    except OSError:
        op_text = ""
    profile_name, _body = split_operator_yaml_profile(op_text)
    pn = profile_name.strip()
    if pn:
        argv.extend(["--profile", pn])
    argv.extend(["--config", str(operator_yaml)])
    spec = corpus_root / FEEDS_SPEC_DEFAULT_BASENAME
    if spec.is_file():
        argv.extend(["--feeds-spec", str(spec.resolve())])
    return argv


def argv_summary(argv: Sequence[str]) -> str:
    """Persistable argv representation for the registry row."""
    return json.dumps(list(argv), ensure_ascii=False)


def _running_count(jobs: list[dict[str, Any]]) -> int:
    return sum(1 for j in jobs if j.get("status") == STATUS_RUNNING)


def _sort_queued(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q = [j for j in jobs if j.get("status") == STATUS_QUEUED]
    q.sort(key=lambda j: j.get("created_at") or "")
    return q


def reconcile_jobs_inplace(jobs: list[dict[str, Any]], *, stale_seconds: int) -> list[str]:
    """Mutate *jobs* in place; return human-readable detail lines."""
    details: list[str] = []
    now = datetime.now(timezone.utc)
    for j in jobs:
        if j.get("status") != STATUS_RUNNING:
            continue
        pid = j.get("pid")
        stale_wall = False
        started = j.get("started_at")
        if stale_seconds > 0 and isinstance(started, str) and started.strip():
            try:
                ts = datetime.fromisoformat(started.replace("Z", "+00:00"))
                if (now - ts).total_seconds() > stale_seconds:
                    stale_wall = True
            except ValueError:
                pass
        # Prefer explicit PID orphan detection over wall-clock stale when both apply.
        if pid is not None and int(pid) > 0 and not pid_alive(int(pid)):
            j["status"] = STATUS_FAILED
            j["ended_at"] = _utc_iso()
            j["error_reason"] = "orphan_reconciled_dead_pid"
            j["exit_code"] = -1
            details.append(f"{j.get('job_id')}: failed (dead pid)")
            continue
        if stale_wall:
            j["status"] = STATUS_STALE
            j["ended_at"] = _utc_iso()
            j["error_reason"] = "wall_clock_stale"
            details.append(f"{j.get('job_id')}: marked stale (wall-clock timeout)")
    return details


def _new_job_record(
    *,
    job_id: str,
    argv: list[str],
    log_relpath: str,
    status: str,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "job_id": job_id,
        "command_type": COMMAND_FULL,
        "status": status,
        "created_at": _utc_iso(),
        "started_at": _utc_iso() if status == STATUS_RUNNING else None,
        "ended_at": None,
        "pid": None,
        "argv_summary": argv_summary(argv),
        "exit_code": None,
        "log_relpath": log_relpath,
        "error_reason": None,
        "cancel_requested": False,
    }
    return rec


def enqueue_pipeline_job(corpus_root: Path, operator_yaml: Path) -> dict[str, Any]:
    """Append a new job; promote to *running* immediately when under the concurrency cap."""

    def fn(jobs: list[dict[str, Any]]) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        log_relpath = f".viewer/jobs/{job_id}.log"
        argv = build_pipeline_argv(corpus_root, operator_yaml)
        cap = max_concurrent_jobs()
        if _running_count(jobs) < cap:
            rec = _new_job_record(
                job_id=job_id, argv=argv, log_relpath=log_relpath, status=STATUS_RUNNING
            )
            jobs.append(rec)
            return rec
        rec = _new_job_record(
            job_id=job_id, argv=argv, log_relpath=log_relpath, status=STATUS_QUEUED
        )
        rec["started_at"] = None
        jobs.append(rec)
        return rec

    return with_jobs_locked_mutate(corpus_root, fn)


def list_jobs_snapshot(corpus_root: Path) -> list[dict[str, Any]]:
    """Return all jobs with optional ``queue_position`` for queued rows."""

    def fn(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        queued = _sort_queued(jobs)
        pos: dict[str, int] = {str(j["job_id"]): i + 1 for i, j in enumerate(queued)}
        out: list[dict[str, Any]] = []
        for j in jobs:
            row = dict(j)
            jid = str(j.get("job_id", ""))
            if j.get("status") == STATUS_QUEUED and jid in pos:
                row["queue_position"] = pos[jid]
            out.append(row)
        out.sort(key=lambda r: r.get("created_at") or "")
        return out

    return with_jobs_locked_read(corpus_root, fn)


def get_job(corpus_root: Path, job_id: str) -> dict[str, Any] | None:
    """Return one job record or None."""

    def fn(jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
        for j in jobs:
            if str(j.get("job_id")) == job_id:
                return dict(j)
        return None

    return with_jobs_locked_read(corpus_root, fn)


def apply_reconcile(corpus_root: Path) -> tuple[int, list[str]]:
    """Reconcile registry under lock; return ``(updated_count, detail_lines)``."""

    def fn(jobs: list[dict[str, Any]]) -> tuple[int, list[str]]:
        details = reconcile_jobs_inplace(jobs, stale_seconds=stale_after_seconds())
        return len(details), details

    return with_jobs_locked_mutate(corpus_root, fn)


def cancel_job(corpus_root: Path, job_id: str) -> tuple[str, dict[str, Any] | None]:
    """Return (outcome, updated_record_or_none). outcome: cancelled | noop_terminal | not_found."""

    def fn(jobs: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
        for j in jobs:
            if str(j.get("job_id")) != job_id:
                continue
            st = j.get("status")
            if st in TERMINAL:
                return "noop_terminal", dict(j)
            if st == STATUS_QUEUED:
                j["status"] = STATUS_CANCELLED
                j["ended_at"] = _utc_iso()
                j["exit_code"] = None
                j["error_reason"] = "cancelled_before_start"
                return "cancelled", dict(j)
            if st == STATUS_RUNNING:
                j["cancel_requested"] = True
                return "signal_running", dict(j)
        return "not_found", None

    outcome, rec = with_jobs_locked_mutate(corpus_root, fn)
    if outcome == "not_found":
        return "not_found", None
    if outcome == "noop_terminal":
        return "noop_terminal", rec
    if outcome == "cancelled":
        return "cancelled", rec
    # signal_running
    pid = rec.get("pid") if rec else None
    if pid and int(pid) > 0:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError as exc:
            logger.warning("cancel sigterm failed job=%s pid=%s: %s", job_id, pid, exc)
    return "cancelled", rec


def set_job_pid(corpus_root: Path, job_id: str, pid: int) -> None:
    """Persist child PID after spawn."""

    def fn(jobs: list[dict[str, Any]]) -> None:
        for j in jobs:
            if str(j.get("job_id")) == job_id:
                j["pid"] = int(pid)
                return

    with_jobs_locked_mutate(corpus_root, fn)


def promote_queued_if_slot(corpus_root: Path, operator_yaml: Path) -> dict[str, Any] | None:
    """If under cap, flip oldest queued job to running; return that record or None."""

    def fn(jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
        if _running_count(jobs) >= max_concurrent_jobs():
            return None
        q = _sort_queued(jobs)
        if not q:
            return None
        j = q[0]
        argv = build_pipeline_argv(corpus_root, operator_yaml)
        j["status"] = STATUS_RUNNING
        j["started_at"] = _utc_iso()
        j["argv_summary"] = argv_summary(argv)
        j["cancel_requested"] = False
        j["pid"] = None
        return dict(j)

    return with_jobs_locked_mutate(corpus_root, fn)


async def spawn_pipeline_subprocess(
    app: Any,
    corpus_root: Path,
    job_id: str,
    argv: list[str],
    log_abs: Path,
) -> asyncio.subprocess.Process:
    """Spawn the pipeline child (or delegate to ``app.state.jobs_subprocess_factory``)."""
    factory = getattr(app.state, "jobs_subprocess_factory", None)
    if factory is not None:
        proc = await factory(argv, corpus_root, log_abs)
        return cast(asyncio.subprocess.Process, proc)

    def _open_log_binary() -> Any:
        log_abs.parent.mkdir(parents=True, exist_ok=True)
        return open(log_abs, "wb")

    log_f = await asyncio.to_thread(_open_log_binary)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=log_f,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(corpus_root),
        start_new_session=os.name != "nt",
    )
    setattr(proc, "_ps_log_fp", log_f)
    return proc


async def _finalize_job(
    corpus_root: Path,
    job_id: str,
    *,
    exit_code: int | None,
    cancelled: bool,
) -> None:
    def fn(jobs: list[dict[str, Any]]) -> None:
        for j in jobs:
            if str(j.get("job_id")) != job_id:
                continue
            if j.get("status") in TERMINAL:
                return
            j["ended_at"] = _utc_iso()
            if cancelled or j.get("cancel_requested"):
                j["status"] = STATUS_CANCELLED
                j["exit_code"] = exit_code
                j["error_reason"] = j.get("error_reason") or "cancelled"
                return
            code = int(exit_code) if exit_code is not None else -1
            j["exit_code"] = code
            if code == 0:
                j["status"] = STATUS_SUCCEEDED
            else:
                j["status"] = STATUS_FAILED
                j["error_reason"] = j.get("error_reason") or f"exit_code_{code}"

    await asyncio.to_thread(with_jobs_locked_mutate, corpus_root, fn)


async def monitor_subprocess(
    app: Any,
    corpus_root: Path,
    job_id: str,
    proc: asyncio.subprocess.Process,
) -> None:
    """Wait for the child, finalize registry, then try to promote queued work."""
    log_fp = getattr(proc, "_ps_log_fp", None)
    try:
        try:
            code = await proc.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("job wait failed job=%s: %s", job_id, exc)
            code = -1
        rec = await asyncio.to_thread(get_job, corpus_root, job_id)
        cancelled = bool(rec and rec.get("cancel_requested"))
        await _finalize_job(corpus_root, job_id, exit_code=code, cancelled=cancelled)
        await drain_queue_async(app, corpus_root)
    finally:
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass


async def start_job_if_running_record(
    app: Any,
    corpus_root: Path,
    operator_yaml: Path,
    job: dict[str, Any],
) -> None:
    """Spawn and monitor when ``job`` is already marked *running* in the registry."""
    if job.get("status") != STATUS_RUNNING:
        return
    job_id = str(job["job_id"])
    argv = build_pipeline_argv(corpus_root, operator_yaml)
    log_abs = corpus_root / str(job.get("log_relpath", f".viewer/jobs/{job_id}.log"))
    try:
        proc = await spawn_pipeline_subprocess(app, corpus_root, job_id, argv, log_abs)
    except Exception as exc:
        logger.exception("spawn failed job=%s", job_id)
        err_msg = str(exc)

        def _fail_mark_spawn_failed(jobs: list[dict[str, Any]]) -> None:
            for j in jobs:
                if str(j.get("job_id")) != job_id:
                    continue
                j["status"] = STATUS_FAILED
                j["ended_at"] = _utc_iso()
                j["error_reason"] = f"spawn_failed: {err_msg}"
                j["exit_code"] = -1

        await asyncio.to_thread(with_jobs_locked_mutate, corpus_root, _fail_mark_spawn_failed)
        await drain_queue_async(app, corpus_root)
        return

    if proc.pid:
        await asyncio.to_thread(set_job_pid, corpus_root, job_id, proc.pid)
    await monitor_subprocess(app, corpus_root, job_id, proc)


async def drain_queue_async(app: Any, corpus_root: Path) -> None:
    """Start queued jobs until the concurrency cap is reached."""
    operator_yaml = viewer_operator_yaml_path(app, corpus_root)
    while True:
        promoted = await asyncio.to_thread(promote_queued_if_slot, corpus_root, operator_yaml)
        if promoted is None:
            break
        await start_job_if_running_record(app, corpus_root, operator_yaml, promoted)


async def schedule_post_submit(app: Any, corpus_root: Path, rec: dict[str, Any]) -> None:
    """Background entry: spawn when the accepted job is already *running*."""
    operator_yaml = viewer_operator_yaml_path(app, corpus_root)
    if rec.get("status") == STATUS_RUNNING:
        await start_job_if_running_record(app, corpus_root, operator_yaml, rec)
