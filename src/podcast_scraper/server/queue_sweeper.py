"""Keep the job queue moving without waiting for another job to finish (#1653).

The queue had no heartbeat. Two functions existed and were correct, and nothing called them
on a schedule:

* ``drain_queue_async`` — promote a queued job when a slot frees. Called in exactly two
  places, both inside a job's own lifecycle: after its child process exits, and after a spawn
  failure. So promotion was **edge-triggered on another job finishing**.
* ``apply_reconcile`` — mark a ``running`` row failed when its pid is gone or it has blown the
  stale window. Called from ``POST /api/jobs/reconcile`` only, i.e. by hand.

Together those two gaps produce a queue that can stop permanently:

1. A job is ``running`` with a pid.
2. Its process dies without ``_finalize_job`` — the API container is killed, restarted, OOMs.
3. The row stays ``running`` forever, and ``_running_count`` still counts it, so it holds a
   concurrency slot.
4. Fill the slots with ghosts and **every queued job waits forever**, because the only thing
   that promotes a queued job is another job finishing — and nothing can start.
5. Recovery requires a human to notice and POST ``/api/jobs/reconcile``.

This is not hypothetical: enrichment job ``ef9f8f9c`` sat queued for 7.75 hours before it
ran. The registry is shared by every container, so the pipeline can enqueue its own follow-up
enrichment — but only this server can start one, which makes the server responsible for
noticing that work is waiting.

Reconcile runs BEFORE drain on every sweep, deliberately: freeing a ghost slot is what makes
a promotion possible in the wedged case, and doing it the other way round would drain against
a slot count that is still wrong.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: How often to reconcile + drain. Small enough that a stranded queue recovers on its own
#: well inside a human's attention span; large enough that the pid checks and one JSONL read
#: are irrelevant next to the work the jobs themselves do.
DEFAULT_SWEEP_INTERVAL_SECONDS = 30.0

#: Touch this file to stop the sweeper promoting queued work; delete it to resume.
PAUSE_FLAG_RELPATH = ".viewer/jobs.paused"


def drain_is_paused(corpus_root: Path) -> bool:
    """Is operator-controlled promotion currently held?

    Without this, starting the API server *is* starting whatever is queued: the startup sweep
    promotes within seconds of boot, every boot. That is the right default for normal
    operation and the wrong one for a corpus repair, where the operator needs to bring the
    stack up, look at it, and decide what runs.

    It also covers a gap the enqueue rework leaves open: a repair driven as a plain CLI run
    holds no registry slot, so nothing stops the sweeper promoting a queued enrichment pass
    that would read the very files the repair is rewriting — a smaller version of the hazard
    that replacing the detached ``Popen`` was meant to close.

    A missing corpus root reads as "not paused": the sweeper is already a no-op there, and
    failing closed on an unreadable path would silently stop the queue for the wrong reason.
    """
    try:
        return (corpus_root / PAUSE_FLAG_RELPATH).exists()
    except OSError:
        return False


def pause_drain(corpus_root: Path) -> None:
    """Hold promotion. The #1785 stop endpoint calls this BEFORE signalling anything — the
    sweeper's loop would otherwise promote the next queued job straight into the freed slot."""
    flag = corpus_root / PAUSE_FLAG_RELPATH
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()


def resume_drain(corpus_root: Path) -> None:
    """Release promotion (idempotent)."""
    try:
        (corpus_root / PAUSE_FLAG_RELPATH).unlink()
    except FileNotFoundError:
        pass


def _queued_count(corpus_root: Path) -> int:
    """Best-effort count of waiting jobs, for the paused-log line only."""
    try:
        from podcast_scraper.server.jobs import STATUS_QUEUED
        from podcast_scraper.server.pipeline_job_registry import with_jobs_locked_read

        return with_jobs_locked_read(
            corpus_root,
            lambda jobs: sum(1 for j in jobs if j.get("status") == STATUS_QUEUED),
        )
    except Exception:  # pragma: no cover — a log line must never break the sweep
        return -1


async def sweep_once(app: Any, corpus_root: Path) -> int:
    """Reconcile dead/stale rows, then promote whatever the freed slots allow.

    Returns the number of rows reconciled. Never raises: a housekeeping loop that dies on a
    transient registry error is worse than one that logs and tries again — the failure mode
    it exists to prevent is precisely "nothing is watching".
    """
    from podcast_scraper.server.jobs import apply_reconcile, drain_queue_async

    reconciled = 0
    try:
        count, details = await asyncio.to_thread(
            functools.partial(
                apply_reconcile,
                corpus_root,
                # An operator posting /api/jobs/reconcile may deliberately free the slot of a
                # job that is running long. A 30 s timer must not make that call on its own —
                # it would let a second writer into the corpus while the first is mid-write.
                stale_marks_live_processes=False,
            )
        )
        reconciled = int(count)
        if reconciled:
            # WARNING, not INFO: a reconciled row means a job died without finalising, which
            # is a real incident even though the queue has just recovered from it.
            logger.warning(
                "job queue: reconciled %d stranded row(s) — %s",
                reconciled,
                "; ".join(details[:5]),
            )
    except (OSError, ValueError) as exc:
        logger.warning("job queue: reconcile failed this sweep (%s); will retry", exc)

    if drain_is_paused(corpus_root):
        # Reconcile still ran, so the registry stays truthful while promotion is held.
        # The count goes through a thread: it takes the cross-process registry lock, and
        # blocking the event loop on a lock the pipeline container may hold — for a log
        # line — would be a poor trade.
        waiting = await asyncio.to_thread(_queued_count, corpus_root)
        logger.info(
            "job queue: drain paused by %s — %d queued job(s) left waiting",
            PAUSE_FLAG_RELPATH,
            waiting,
        )
        return reconciled

    try:
        await drain_queue_async(app, corpus_root)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("job queue: drain failed this sweep (%s); will retry", exc)

    return reconciled


async def _sweep_loop(app: Any, corpus_root: Path, interval_seconds: float) -> None:
    """Sweep forever until cancelled."""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            await sweep_once(app, corpus_root)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover — belt and braces
            # The loop must outlive any single bad sweep. If this ever swallows something
            # important the WARNING is the trail.
            logger.warning("job queue: sweep iteration failed (%s); loop continues", exc)


async def start_queue_sweeper(
    app: Any,
    *,
    interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
) -> Optional[asyncio.Task]:
    """Sweep once now, then every *interval_seconds*. Returns the task, or None if disabled.

    The immediate sweep matters as much as the loop: a restart is exactly when ghost
    ``running`` rows exist, left by whatever killed the previous process. Waiting a full
    interval to notice would mean the queue is wedged for the first sweep window of every
    restart — the moment an operator is most likely to be watching and least likely to guess
    that the fix is to wait.
    """
    if not getattr(app.state, "jobs_api_enabled", False):
        return None
    corpus_root = getattr(app.state, "output_dir", None)
    if corpus_root is None:
        return None

    root = Path(corpus_root)
    await sweep_once(app, root)
    task = asyncio.create_task(_sweep_loop(app, root, interval_seconds))
    logger.info("job queue: sweeper started (every %.0fs)", interval_seconds)
    return task


async def stop_queue_sweeper(task: Optional[asyncio.Task]) -> None:
    """Cancel the sweeper and wait for it to finish."""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
