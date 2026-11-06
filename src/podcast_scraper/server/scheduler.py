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

Each schedule carries a ``kind`` (``pipeline`` | ``enrichment``, #1069) and
otherwise reuses the corpus's standing ``viewer_operator.yaml`` +
``feeds.spec.yaml`` exactly the way a manual viewer-triggered run does. Other
per-schedule overrides (``profile``, ``feeds``, ``max_episodes``) remain V2 work.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from podcast_scraper.extensions import job_runner, load_extensions

logger = logging.getLogger(__name__)

# ``scheduled_jobs:`` lives at the YAML root alongside Config-shaped fields.
# It's listed in ``OPERATOR_ONLY_TOP_LEVEL_KEYS`` so the pipeline CLI strips
# it before Pydantic's ``extra="forbid"`` validates the rest.
SCHEDULED_JOBS_KEY = "scheduled_jobs"

# Job kinds a schedule can fire. ``pipeline`` (ingestion) is the default for
# back-compat with existing ``scheduled_jobs:`` entries; ``enrichment`` makes the
# enrichment run schedulable exactly like ingestion (#1069 consistency).
JOB_KIND_PIPELINE = "pipeline"
JOB_KIND_ENRICHMENT = "enrichment"
# ``digest`` fires the per-user "Your Week" digest enqueue (#1415) — enqueues DeliveryEnvelopes to
# the outbox for the infra worker to deliver; extractive, no pipeline job. Idempotent per period.
#: The platform's own kinds. Installed extensions add theirs (the player's ``digest``, ADR-168).
_PLATFORM_JOB_KINDS = frozenset({JOB_KIND_PIPELINE, JOB_KIND_ENRICHMENT})


def _valid_job_kinds() -> frozenset[str]:
    kinds = set(_PLATFORM_JOB_KINDS)
    for ext in load_extensions():
        kinds.update(ext.job_kinds)
    return frozenset(kinds)


class ScheduledJobConfig(BaseModel):
    """One entry in ``viewer_operator.yaml`` ``scheduled_jobs:``."""

    name: str = Field(..., min_length=1, max_length=64)
    cron: str = Field(..., min_length=1)
    enabled: bool = True
    # Which job to fire on the schedule — ``pipeline`` (ingestion) or
    # ``enrichment``. Both run through the identical enqueue → spawn path.
    kind: str = Field(default=JOB_KIND_PIPELINE)

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, v: str) -> str:
        v = (v or "").strip().lower()
        valid = _valid_job_kinds()
        if v not in valid:
            raise ValueError(f"scheduled job kind must be one of {sorted(valid)}")
        return v

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
# ``prometheus_client`` is missing — metrics extras not installed — the
# counters degrade to no-ops so the scheduler still works in minimal
# environments (e.g., stripped-down containers without prometheus_client).

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


# --- Missed-slot detection ----------------------------------------------

#: Per-job fire/report times, beside the job registry. APScheduler's in-memory jobstore forgets
#: every slot that passed while the process was down, so this file is the only record of it.
SCHEDULER_STATE_RELPATH = ".viewer/scheduler_state.json"
_SLOT_SCAN_LIMIT = 20000
_SKIP_NOT_RUNNING = "scheduler_not_running_at_slot"
_SKIP_MISFIRE = "misfire"


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _last_slot_between(trigger: Any, start: datetime, now: datetime) -> Optional[datetime]:
    """The latest cron slot in ``[start, now)``, or None."""
    last: Optional[datetime] = None
    slot = trigger.get_next_fire_time(None, start)
    for _ in range(_SLOT_SCAN_LIMIT):
        if slot is None or slot >= now:
            break
        last = slot
        slot = trigger.get_next_fire_time(slot, slot + timedelta(microseconds=1))
    return last


def _emit_skipped(job: str, kind: str, slot: datetime, reason: str) -> None:
    try:
        from podcast_scraper.obs.events import emit_event

        emit_event(
            "pipeline_run.skipped",
            job=job,
            kind=kind,
            slot_at=_iso_utc(slot),
            reason=reason,
            detected_at=_iso_utc(datetime.now(timezone.utc)),
        )
    except Exception:  # noqa: BLE001 - telemetry must never break the scheduler
        logger.debug("scheduler: pipeline_run.skipped emit failed", exc_info=True)


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
        """``spawn`` is called as ``spawn(name, corpus_root, operator_yaml, kind)``.

        It must perform the equivalent of ``POST /api/jobs`` (``kind="pipeline"``)
        or ``POST /api/jobs/enrichment`` (``kind="enrichment"``) — enqueue +
        schedule_post_submit — and is responsible for its own error logging.
        Raising lets the scheduler increment the failure counter.
        """
        self._corpus_root = corpus_root
        self._operator_yaml = operator_yaml
        self._spawn = spawn
        self._scheduler: Any | None = None
        self._jobs: list[ScheduledJobConfig] = []
        self._state_lock = threading.Lock()
        # The last slot reported per job, also kept here: when the state file cannot be written
        # (a read-only corpus), a reload must still not report the same slot again.
        self._reported: dict[str, str] = {}
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
                "Install with ``pip install -e '.[dev]'`` (or add apscheduler explicitly).",
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
        try:
            self._report_missed_slots()
        except Exception:  # noqa: BLE001 - missed-slot telemetry never blocks the nightly
            logger.debug("scheduler: missed-slot report failed", exc_info=True)
        try:
            from apscheduler.events import EVENT_JOB_MISSED

            sch.add_listener(self._on_job_missed, EVENT_JOB_MISSED)
        except ImportError:  # pragma: no cover - apscheduler imported above
            pass
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

    def _state_path(self) -> Path:
        return self._corpus_root / SCHEDULER_STATE_RELPATH

    def _load_state(self) -> dict[str, dict[str, str]]:
        try:
            raw = json.loads(self._state_path().read_text(encoding="utf-8"))
            jobs = raw.get("jobs") if isinstance(raw, dict) else None
        except (OSError, ValueError):
            return {}
        if not isinstance(jobs, dict):
            return {}
        return {k: v for k, v in jobs.items() if isinstance(v, dict)}

    def _save_state(self, state: dict[str, dict[str, str]]) -> None:
        path = self._state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".scheduler_state.")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump({"jobs": state}, fh, sort_keys=True)
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.warning("scheduler: could not write %s (%s)", path, exc.__class__.__name__)

    def _record_fire(self, name: str) -> None:
        with self._state_lock:
            state = self._load_state()
            state.setdefault(name, {})["last_fire"] = _iso_utc(datetime.now(timezone.utc))
            self._save_state(state)

    def _report_missed_slots(self) -> None:
        """Emit ``pipeline_run.skipped`` for each enabled job whose latest slot never fired.

        A job seen for the first time only looks at slots after that moment, so enabling a new
        schedule reports nothing. A disabled job's entry is dropped so re-enabling it starts fresh.
        """
        try:
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            return
        now = datetime.now(timezone.utc)
        with self._state_lock:
            state = self._load_state()
            before = json.dumps(state, sort_keys=True)
            for cfg in self._jobs:
                if not cfg.enabled:
                    state.pop(cfg.name, None)
                    continue
                entry = state.setdefault(cfg.name, {})
                entry.setdefault("since", _iso_utc(now))
                self._merge_reported(cfg.name, entry)
                try:
                    trigger = CronTrigger.from_crontab(cfg.cron, timezone=_scheduler_timezone())
                    marks = [
                        _parse_iso(entry.get(k))
                        for k in ("since", "last_fire", "last_reported_slot")
                    ]
                    start = max(m for m in marks if m is not None) + timedelta(microseconds=1)
                    slot = _last_slot_between(trigger, start, now)
                except Exception:  # noqa: BLE001 - never block scheduler start on detection
                    logger.debug(
                        "scheduler: missed-slot scan failed for %r", cfg.name, exc_info=True
                    )
                    continue
                if slot is None:
                    continue
                entry["last_reported_slot"] = _iso_utc(slot)
                self._reported[cfg.name] = entry["last_reported_slot"]
                _emit_skipped(cfg.name, cfg.kind, slot, _SKIP_NOT_RUNNING)
            if json.dumps(state, sort_keys=True) != before:
                self._save_state(state)

    def _merge_reported(self, name: str, entry: dict[str, str]) -> None:
        """Take the in-memory last reported slot when it is later than the file's."""
        mem = _parse_iso(self._reported.get(name))
        disk = _parse_iso(entry.get("last_reported_slot"))
        if mem is not None and (disk is None or mem > disk):
            entry["last_reported_slot"] = self._reported[name]

    def _on_job_missed(self, event: Any) -> None:
        """APScheduler ``EVENT_JOB_MISSED`` listener: the slot was reached but past its grace."""
        try:
            name = str(event.job_id)
            slot = event.scheduled_run_time
            with self._state_lock:
                state = self._load_state()
                entry = state.setdefault(name, {})
                self._merge_reported(name, entry)
                reported = _parse_iso(entry.get("last_reported_slot"))
                if reported is not None and reported >= slot:
                    return
                entry["last_reported_slot"] = _iso_utc(slot)
                self._reported[name] = entry["last_reported_slot"]
                self._save_state(state)
            _emit_skipped(name, self._kind_for(name), slot, _SKIP_MISFIRE)
        except Exception:  # noqa: BLE001 - listener must not raise into APScheduler
            logger.debug("scheduler: misfire listener failed", exc_info=True)

    def _kind_for(self, name: str) -> str:
        """The configured job kind (``pipeline`` | ``enrichment``) for a schedule."""
        for job in self._jobs:
            if job.name == name:
                return job.kind
        return JOB_KIND_PIPELINE

    def _fire(self, name: str) -> None:
        """APScheduler-thread entry point — must not raise to APScheduler.

        APScheduler swallows exceptions and only emits a warning, so we own
        the error→counter→log flow ourselves.
        """
        _record_triggered(name)
        try:
            self._record_fire(name)
        except Exception:  # noqa: BLE001 - bookkeeping must not stop the fire
            logger.debug("scheduler: could not record fire of %r", name, exc_info=True)
        try:
            self._spawn(name, self._corpus_root, self._operator_yaml, self._kind_for(name))
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

    from podcast_scraper.server.jobs import (
        enqueue_enrichment_job,
        enqueue_pipeline_job,
        schedule_post_submit,
    )

    def _spawn(
        name: str, corpus_root: Path, operator_yaml: Path, kind: str = JOB_KIND_PIPELINE
    ) -> None:
        # A kind an extension owns runs through that extension, not the job queue (the player's
        # digest enqueues delivery envelopes to the outbox). No event loop / post_submit.
        if kind not in _PLATFORM_JOB_KINDS:
            runner = job_runner(kind)
            if runner is None:
                logger.warning(
                    "scheduler: job %r has kind %r, which nothing installed runs", name, kind
                )
                return
            runner(name, corpus_root, app)
            return
        loop: Optional[asyncio.AbstractEventLoop] = getattr(app.state, "event_loop", None)
        if loop is None or not loop.is_running():
            logger.warning("scheduler: event loop unavailable when firing %r; skipping", name)
            raise RuntimeError("event loop unavailable")
        # Same enqueue → schedule_post_submit path as POST /api/jobs (pipeline)
        # / POST /api/jobs/enrichment — a scheduled fire is indistinguishable
        # from a manual one at the registry.
        if kind == JOB_KIND_ENRICHMENT:
            rec = enqueue_enrichment_job(corpus_root, operator_yaml=operator_yaml)
        else:
            rec = enqueue_pipeline_job(corpus_root, operator_yaml, trigger="nightly")
        logger.info(
            "scheduler: job %r fired -> %s job_id=%s status=%s",
            name,
            kind,
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
