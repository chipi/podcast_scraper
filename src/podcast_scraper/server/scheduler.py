"""In-process cron scheduler for feed sweeps (#708).

Reads ``scheduled_jobs:`` from the corpus's ``viewer_operator.yaml`` at app
startup (and after operator-config PUT) and fires APScheduler ``CronTrigger``s
that invoke the **same** internal job-spawn path as ``POST /api/jobs`` —
``enqueue_pipeline_job`` followed by ``schedule_post_submit``. No subprocess
or argv duplication.

Why API-level (not a host-side cron / systemd-timer): works on Codespace
pre-prod (no systemd) and VPS prod alike, the operator UX is the existing
viewer Configuration tab, and failure events flow through the existing
``emit_job_state_change`` webhook surface — see GH issue #708.

V1 schedule shape (Shape A from #708 — extend ``viewer_operator.yaml``)::

    scheduled_jobs:
      - name: morning-feed-sweep
        cron: "0 4 * * *"
        enabled: true

Per-schedule overrides (``profile``, ``feeds``, ``max_episodes``) are V2
work. Each schedule reuses the corpus's standing ``viewer_operator.yaml`` +
``feeds.spec.yaml`` exactly the way a manual viewer-triggered run does.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ``scheduled_jobs:`` lives at the YAML root alongside Config-shaped fields.
# It's listed in ``OPERATOR_ONLY_TOP_LEVEL_KEYS`` so the pipeline CLI strips
# it before Pydantic's ``extra="forbid"`` validates the rest.
SCHEDULED_JOBS_KEY = "scheduled_jobs"


class ScheduledJobConfig(BaseModel):
    """One entry in ``viewer_operator.yaml`` ``scheduled_jobs:``."""

    name: str = Field(..., min_length=1, max_length=64)
    cron: str = Field(..., min_length=1)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        # Keep it tight enough to use as a Prometheus label without escaping.
        # Cron job ids reuse this name; APScheduler tolerates anything but
        # operators copy this value into Slack alerts and dashboards too.
        v = v.strip()
        if not v:
            raise ValueError("scheduled job name must be non-empty")
        if not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("scheduled job name may only contain letters, digits, '-' and '_'")
        return v

    @field_validator("cron")
    @classmethod
    def _validate_cron(cls, v: str) -> str:
        # Defer the real validation to APScheduler at registration time —
        # but reject obviously empty strings up front so a malformed YAML
        # entry surfaces during config load, not silently at first fire.
        v = v.strip()
        if not v:
            raise ValueError("cron expression must be non-empty")
        return v


class ScheduledJobsParseError(Exception):
    """Raised when ``scheduled_jobs:`` is present but malformed."""


def parse_scheduled_jobs(yaml_text: str) -> list[ScheduledJobConfig]:
    """Extract validated schedule list from operator YAML text.

    Returns ``[]`` when the key is absent. Raises :class:`ScheduledJobsParseError`
    when the key is present but malformed; the scheduler treats this as
    fail-loud during reload so a typo doesn't silently disable the schedule.
    """
    if not yaml_text or not yaml_text.strip():
        return []
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ScheduledJobsParseError(f"invalid YAML: {exc}") from exc
    if parsed is None or not isinstance(parsed, dict):
        return []
    raw = parsed.get(SCHEDULED_JOBS_KEY)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ScheduledJobsParseError(f"{SCHEDULED_JOBS_KEY} must be a list of objects")
    out: list[ScheduledJobConfig] = []
    seen_names: set[str] = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ScheduledJobsParseError(f"{SCHEDULED_JOBS_KEY}[{idx}] must be a mapping")
        try:
            cfg = ScheduledJobConfig.model_validate(entry)
        except Exception as exc:
            raise ScheduledJobsParseError(f"{SCHEDULED_JOBS_KEY}[{idx}] invalid: {exc}") from exc
        if cfg.name in seen_names:
            raise ScheduledJobsParseError(f"{SCHEDULED_JOBS_KEY}: duplicate name {cfg.name!r}")
        seen_names.add(cfg.name)
        out.append(cfg)
    return out


def _read_operator_yaml(operator_yaml: Path) -> str:
    """Read the operator YAML file; return ``""`` when missing."""
    if not operator_yaml.is_file():
        return ""
    try:
        return operator_yaml.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("scheduler: cannot read %s (%s)", operator_yaml, exc)
        return ""


def load_scheduled_jobs(operator_yaml: Path) -> list[ScheduledJobConfig]:
    """Parse ``scheduled_jobs:`` from the operator YAML at ``operator_yaml``.

    Returns ``[]`` when the file is missing or doesn't contain the key.
    """
    return parse_scheduled_jobs(_read_operator_yaml(operator_yaml))


# --- Prometheus counters -------------------------------------------------
#
# Defined at module scope so they survive scheduler reloads (each reload
# tears down jobs but reuses the same counter registry). When
# ``prometheus_client`` is missing — `[server]` extra not installed — the
# counters degrade to no-ops so the scheduler still works in `[dev]`-only
# environments (e.g., unit-test runners).

_TRIGGERED_COUNTER: Any | None = None
_FAILED_COUNTER: Any | None = None


def _ensure_counters() -> None:
    """Lazy-init the Prometheus counters; idempotent."""
    global _TRIGGERED_COUNTER, _FAILED_COUNTER
    if _TRIGGERED_COUNTER is not None:
        return
    try:
        from prometheus_client import Counter
    except ImportError:
        return
    _TRIGGERED_COUNTER = Counter(
        "podcast_scheduled_jobs_triggered_total",
        "Total scheduled feed-sweep firings.",
        ["name"],
    )
    _FAILED_COUNTER = Counter(
        "podcast_scheduled_jobs_failed_total",
        "Scheduled feed-sweep firings that failed before a job_id was issued.",
        ["name", "reason"],
    )


def _record_triggered(name: str) -> None:
    if _TRIGGERED_COUNTER is None:
        return
    try:
        _TRIGGERED_COUNTER.labels(name=name).inc()
    except Exception:
        pass


def _record_failed(name: str, reason: str) -> None:
    if _FAILED_COUNTER is None:
        return
    try:
        _FAILED_COUNTER.labels(name=name, reason=reason).inc()
    except Exception:
        pass


# --- SchedulerService ---------------------------------------------------


class SchedulerService:
    """Owns the ``BackgroundScheduler`` and the spawn callback wiring.

    One instance per FastAPI app; lives on ``app.state.scheduler``. Started
    in the app's lifespan; reload triggered by ``operator-config`` PUT.

    The spawn callback is injected so unit tests can substitute a fake
    without standing up the full subprocess pipeline.
    """

    def __init__(
        self,
        corpus_root: Path,
        operator_yaml: Path,
        spawn: Any,
    ) -> None:
        """``spawn`` is called as ``spawn(name, corpus_root, operator_yaml)``.

        It must perform the equivalent of ``POST /api/jobs`` — enqueue +
        schedule_post_submit — and is responsible for its own error logging.
        Raising lets the scheduler increment the failure counter.
        """
        self._corpus_root = corpus_root
        self._operator_yaml = operator_yaml
        self._spawn = spawn
        self._scheduler: Any | None = None
        self._jobs: list[ScheduledJobConfig] = []
        _ensure_counters()

    @property
    def jobs(self) -> list[ScheduledJobConfig]:
        return list(self._jobs)

    @property
    def running(self) -> bool:
        sch = self._scheduler
        return bool(sch is not None and getattr(sch, "running", False))

    @property
    def timezone(self) -> str:
        return _scheduler_timezone()

    def next_run_at(self, name: str) -> str | None:
        """Return the next scheduled fire time for ``name`` as UTC ISO-8601, or None."""
        sch = self._scheduler
        if sch is None:
            return None
        try:
            job = sch.get_job(name)
        except Exception:
            return None
        if job is None:
            return None
        nrt = getattr(job, "next_run_time", None)
        if nrt is None:
            return None
        try:
            # APScheduler returns timezone-aware datetimes; normalize to UTC ISO.
            from datetime import timezone as _tz

            return str(nrt.astimezone(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            return None

    def start(self) -> None:
        """Idempotent start: load jobs, build the scheduler, register triggers."""
        if self._scheduler is not None:
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError as exc:
            logger.warning(
                "scheduler: apscheduler not installed (%s); scheduled_jobs disabled. "
                "Install [server] extras to enable.",
                exc,
            )
            return
        try:
            self._jobs = load_scheduled_jobs(self._operator_yaml)
        except ScheduledJobsParseError as exc:
            logger.error("scheduler: %s; no jobs registered", exc)
            self._jobs = []
        if not any(j.enabled for j in self._jobs):
            logger.info("scheduler: no enabled scheduled_jobs; scheduler not started")
            return
        sch = BackgroundScheduler(timezone=_scheduler_timezone())
        for cfg in self._jobs:
            if not cfg.enabled:
                continue
            self._add_apscheduler_job(sch, cfg)
        sch.start()
        self._scheduler = sch
        logger.info(
            "scheduler: started with %d enabled job(s) (corpus=%s)",
            sum(1 for j in self._jobs if j.enabled),
            self._corpus_root,
        )

    def shutdown(self) -> None:
        """Idempotent shutdown: stop scheduler if running."""
        sch = self._scheduler
        self._scheduler = None
        if sch is None:
            return
        try:
            sch.shutdown(wait=False)
        except Exception as exc:
            logger.warning("scheduler: shutdown error (%s)", exc)

    def reload(self) -> None:
        """Re-read operator YAML and rebuild triggers.

        Called from operator-config PUT. Cheaper than a full restart and
        keeps any actively-firing job from being interrupted.
        """
        self.shutdown()
        self.start()

    def _add_apscheduler_job(self, sch: Any, cfg: ScheduledJobConfig) -> None:
        try:
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            return
        try:
            trigger = CronTrigger.from_crontab(cfg.cron, timezone=_scheduler_timezone())
        except (ValueError, TypeError) as exc:
            logger.error(
                "scheduler: job %r has invalid cron %r (%s); skipping",
                cfg.name,
                cfg.cron,
                exc,
            )
            _record_failed(cfg.name, "invalid_cron")
            return
        sch.add_job(
            self._fire,
            trigger,
            id=cfg.name,
            name=cfg.name,
            args=[cfg.name],
            # ``misfire_grace_time``: if the host was asleep when the trigger
            # was supposed to fire (Codespace suspended, VPS rebooting), give
            # APScheduler a 1-hour window to "catch up" once it wakes. Beyond
            # that, the slot is silently skipped — operators can re-trigger
            # manually if a missed sweep matters.
            misfire_grace_time=3600,
            coalesce=True,
            max_instances=1,
            replace_existing=True,
        )

    def _fire(self, name: str) -> None:
        """APScheduler-thread entry point — must not raise to APScheduler.

        APScheduler swallows exceptions and only emits a warning, so we own
        the error→counter→log flow ourselves.
        """
        _record_triggered(name)
        try:
            self._spawn(name, self._corpus_root, self._operator_yaml)
        except Exception as exc:
            logger.exception("scheduler: job %r spawn failed (%s)", name, exc.__class__.__name__)
            _record_failed(name, exc.__class__.__name__)


def _scheduler_timezone() -> str:
    """Pick the scheduler timezone.

    ``PODCAST_SCHEDULER_TZ`` overrides for operators who want UTC
    everywhere; otherwise we use ``TZ`` (standard cloud-init convention)
    or fall back to ``UTC`` so cron expressions are unambiguous in logs.
    """
    for env in ("PODCAST_SCHEDULER_TZ", "TZ"):
        v = os.environ.get(env, "").strip()
        if v:
            return v
    return "UTC"


# --- App-side spawn callback wiring -------------------------------------


def make_app_spawn_callback(app: Any) -> Any:
    """Return a ``spawn(name, corpus_root, operator_yaml)`` callback bound to ``app``.

    The callback runs on the APScheduler worker thread; it offloads to the
    FastAPI event loop with ``run_coroutine_threadsafe`` so the existing
    async ``schedule_post_submit`` path is reused verbatim. This keeps a
    scheduled fire indistinguishable from a ``POST /api/jobs`` call from
    the registry's point of view.
    """
    import asyncio

    from podcast_scraper.server.pipeline_jobs import (
        enqueue_pipeline_job,
        schedule_post_submit,
    )

    def _spawn(name: str, corpus_root: Path, operator_yaml: Path) -> None:
        loop: Optional[asyncio.AbstractEventLoop] = getattr(app.state, "event_loop", None)
        if loop is None or not loop.is_running():
            logger.warning("scheduler: event loop unavailable when firing %r; skipping", name)
            raise RuntimeError("event loop unavailable")
        rec = enqueue_pipeline_job(corpus_root, operator_yaml)
        logger.info(
            "scheduler: job %r fired -> pipeline job_id=%s status=%s",
            name,
            rec.get("job_id"),
            rec.get("status"),
        )

        async def _kickoff() -> None:
            try:
                await schedule_post_submit(app, corpus_root, rec)
            except Exception as exc:
                logger.exception(
                    "scheduler: post_submit for %r failed (%s)",
                    name,
                    exc.__class__.__name__,
                )

        asyncio.run_coroutine_threadsafe(_kickoff(), loop)

    return _spawn
