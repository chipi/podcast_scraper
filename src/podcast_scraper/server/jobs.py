"""Job queue, subprocess spawn, and registry updates.

Serves multiple ``command_type`` values via the JSONL job registry:

- ``full_incremental_pipeline`` — the original pipeline runs (Phase 2).
- ``corpus_enrichment`` — enrichment-layer runs (Epic #1101, RFC-088).

The promote-queued / cancel / stale-reconcile / pid-alive logic is
``command_type``-agnostic; new job kinds just register new
``COMMAND_*`` constants and ``build_*_argv`` / ``enqueue_*_job`` /
``spawn_*_subprocess`` helpers alongside the existing pipeline ones.

Renamed from ``pipeline_jobs.py`` in Epic #1101 chunk 1 (per O4) to
reflect the multi-kind reality. Module API + behaviour unchanged.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
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
COMMAND_ENRICHMENT = "corpus_enrichment"  # RFC-088 / Epic #1101 chunk 1 sub-6.
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
    """Max concurrent *running* jobs per corpus (default 1).

    #666 review #14: when ``PODCAST_VIEWER_MAX_PIPELINE_JOBS`` is unparsable,
    log a warning so operators see that their env var is being ignored.
    """
    raw = os.environ.get("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "1").strip()
    try:
        n = int(raw)
    except ValueError:
        logger.warning(
            "PODCAST_VIEWER_MAX_PIPELINE_JOBS=%r is not an int; using default 1",
            raw,
        )
        return 1
    return max(1, n)


def stale_after_seconds() -> int:
    """Wall-clock stale threshold for *running* jobs during reconcile.

    #666 review #14: log a warning on parse failure rather than silently
    falling back to the 24h default.
    """
    raw = os.environ.get("PODCAST_JOB_STALE_SECONDS", str(86400)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "PODCAST_JOB_STALE_SECONDS=%r is not an int; using default 86400",
            raw,
        )
        return 86400


# #666 review #10: default per-job log cap (50 MiB). Runaway pipelines hit this
# and the pump writes a truncation marker + /dev/null's any further output so
# disk space is bounded. 0 disables the cap entirely.
_LOG_MAX_BYTES_DEFAULT = 50 * 1024 * 1024

# Docker pull-progress lines emitted during ``docker compose run`` image pulls
# consume the capped log window and can truncate the real pipeline error away.
# These lines are layer-digest / status chatter that has no diagnostic value.
# Tight match: only the docker-compose pull verbs, not generic log lines that
# happen to contain those words.
_DOCKER_PULL_RE = re.compile(
    rb"^(?:"
    rb"Pulling\s+"  # Pulling from … / Pulling fs layer …
    rb"|Waiting\b"
    rb"|Downloading\b"
    rb"|Extracting\b"
    rb"|Verifying Checksum\b"
    rb"|Download complete\b"
    rb"|Pull complete\b"
    rb"|Already exists\b"
    rb"|Pulling fs layer\b"
    rb"|[0-9a-f]{12}:\s+"  # layer-digest prefix lines
    rb")",
    re.IGNORECASE,
)

_ERROR_REASON_MAX_LEN = 300


def _parse_error_reason_from_log(log_path: Path) -> str | None:
    """Scan the captured log for a human-readable failure cause.

    Priority order (first match wins):
      1. A line containing "API key required" or "required" from a config
         validator — these are the missing-secret messages.
      2. The final line of a ``Traceback`` block (the exception class + message).
      3. The last line starting with ``ERROR``/``Error:``/``CRITICAL``.

    Returns None when the log is absent, empty, or yields nothing parseable.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None

    lines = text.splitlines()

    # Priority 1: config-validation error (key-required messages).
    for line in lines:
        ls = line.strip()
        if "API key required" in ls or (
            "required" in ls.lower()
            and ("key" in ls.lower() or "provider" in ls.lower())
            and len(ls) < 400
        ):
            return ls[:_ERROR_REASON_MAX_LEN]

    # Priority 2: last traceback final line (exception type + message).
    last_exc: str | None = None
    in_tb = False
    for line in lines:
        ls = line.rstrip()
        if ls.strip() == "Traceback (most recent call last):":
            in_tb = True
            continue
        if in_tb:
            # Indented continuation lines are the traceback frames.
            if ls.startswith(" ") or ls.startswith("\t"):
                continue
            # A non-indented line after a traceback = the exception line.
            if ls:
                last_exc = ls.strip()
            in_tb = False

    if last_exc:
        return last_exc[:_ERROR_REASON_MAX_LEN]

    # Priority 3: last ERROR/CRITICAL line.
    for line in reversed(lines):
        ls = line.strip()
        if ls.startswith(("ERROR", "Error:", "CRITICAL")):
            return ls[:_ERROR_REASON_MAX_LEN]

    return None


def job_log_max_bytes() -> int:
    """Max bytes a single job may write to its ``.viewer/jobs/*.log``.

    Operator override: ``PODCAST_JOB_LOG_MAX_BYTES`` (integer bytes, ``0`` =
    unlimited). Parse failure logs a warning and falls back to the default.
    """
    raw = os.environ.get("PODCAST_JOB_LOG_MAX_BYTES", str(_LOG_MAX_BYTES_DEFAULT)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "PODCAST_JOB_LOG_MAX_BYTES=%r is not an int; using default %d",
            raw,
            _LOG_MAX_BYTES_DEFAULT,
        )
        return _LOG_MAX_BYTES_DEFAULT


async def _pump_subprocess_to_log(
    stream: asyncio.StreamReader,
    log_abs: Path,
    *,
    max_bytes: int,
    job_id: str,
) -> None:
    """Pump ``stream`` → ``log_abs`` with a per-job byte cap (#666 review #10).

    When ``max_bytes`` is exceeded the remaining bytes are drained from the
    subprocess stream (so the child does not block on a full pipe) but not
    written. A truncation marker is emitted to the log the first time the
    cap is hit.

    Docker pull-progress lines (layer digests, "Pulling", "Waiting", …) are
    dropped before the byte counter advances — they consume the cap without
    adding diagnostic value, causing real pipeline errors to be truncated away.
    """
    log_abs.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    truncated = False
    # ``max_bytes == 0`` disables the cap (operators who want unlimited logs).
    uncapped = max_bytes <= 0
    # Line-buffer: carry the trailing partial line across chunk boundaries so
    # the pull-progress filter sees complete lines, not split prefixes.
    buf = b""

    def _emit(data: bytes) -> None:
        """Write data respecting the cap; emits truncation marker once."""
        nonlocal written, truncated
        if _DOCKER_PULL_RE.match(data):
            return
        if uncapped:
            out.write(data)
            return
        if written >= max_bytes:
            return
        remaining = max_bytes - written
        if len(data) <= remaining:
            out.write(data)
            written += len(data)
        else:
            out.write(data[:remaining])
            written = max_bytes
            if not truncated:
                marker = (
                    f"\n[LOG TRUNCATED at {max_bytes} bytes "
                    "(set PODCAST_JOB_LOG_MAX_BYTES=0 to disable)]\n"
                ).encode("utf-8")
                out.write(marker)
                truncated = True
                logger.warning(
                    "job log truncated job=%s after %d bytes",
                    job_id,
                    max_bytes,
                )

    with open(log_abs, "wb") as out:
        while True:
            try:
                chunk = await stream.read(65536)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("log pump read failed job=%s: %s", job_id, exc)
                break
            if not chunk:
                # EOF: flush any trailing partial line (process wrote without newline).
                if buf:
                    _emit(buf)
                    buf = b""
                break
            buf += chunk
            # Emit complete lines; hold the trailing partial line in buf.
            while b"\n" in buf:
                nl = buf.index(b"\n")
                line = buf[: nl + 1]
                buf = buf[nl + 1 :]
                _emit(line)
            out.flush()


def pid_alive(pid: int | None) -> bool:
    """Return True if ``pid`` responds to ``kill(..., 0)``."""
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


#: Identifies THIS process. Stamped at import, which for the API server is process start.
_BOOT_ID = uuid.uuid4().hex


def current_boot_id() -> str:
    """The id of the currently-running server process.

    A pid only means something to the process that observed it. Two facts make an unscoped
    ``pid_alive`` check actively wrong once it runs unattended on a timer rather than when an
    operator asks for it:

    * In Docker exec mode — which is what production runs
      (``compose/docker-compose.prod.yml`` sets ``PODCAST_PIPELINE_EXEC_MODE=docker``) — the
      recorded pid is the ``docker compose run`` *client* inside the API container, while the
      real work runs in a container on the host daemon. Kill the API container and the client
      dies with it; the job container keeps going. A restart would then see a dead pid and
      mark a **live** job failed, freeing its slot for a second concurrent corpus writer.
    * The new container starts a fresh PID namespace that reuses low pid numbers, so an
      unrelated process can make a dead job look alive — the ghost keeps its slot, and
      ``cancel_job`` would SIGTERM whatever now owns that number.

    So the pid rule applies only to rows this boot created. Prior-boot rows need real evidence
    (``docker_job_alive``), never a pid.
    """
    return _BOOT_ID


def build_pipeline_argv(
    corpus_root: Path,
    operator_yaml: Path,
    *,
    run_id: str | None = None,
    feed_url: str | None = None,
    skip_existing: bool = False,
    append: bool = False,
    max_episodes: int | None = None,
    episode_offset: int | None = None,
    episode_order: str | None = None,
) -> list[str]:
    """Build CLI argv for a full pipeline run (README parity: ``--profile`` then ``--config``).

    When *run_id* is given it is passed as ``--run-id`` so the pipeline's self-generated run id ==
    the Jobs API job_id — a single join key across the Jobs API and observability (podcast_obs
    correlate --run-id), instead of scraping ``[run=…]`` from the log tail (P1.6). The docker exec
    path preserves the CLI-flag tail, so the flag flows through both local and docker modes.

    When *feed_url* is given the run is scoped to that ONE feed as a single-feed corpus-layout run
    (``--rss <url> --single-feed-uses-corpus-layout``) instead of the whole ``--feeds-spec`` batch —
    the cautious per-feed incremental add (P1.4). The incremental knobs (skip_existing / append /
    max_episodes / episode_offset / episode_order) apply only in this scoped mode. Whole-batch is
    unchanged when *feed_url* is None.

    Profile resolution order:

    1. ``profile:`` line in ``viewer_operator.yaml`` (operator's saved choice).
    2. ``PODCAST_DEFAULT_PROFILE`` env var (validated against on-disk profile
       names + allowlist via ``env_default_profile``). Lets a fresh corpus run
       through cloud_thin (or any preprod default) even if the operator
       triggered a job before clicking Save in the profile menu.
    3. No ``--profile`` flag at all (CLI falls back to ``Config._resolve_profile``
       defaults — same as today's pre-RFC-081 behavior).
    """
    # Local import — module-level would create a circular: profile_presets
    # imports nothing from server (today), but the codebase has had churn
    # and a cycle would be hard to detect; defensive.
    from podcast_scraper.server.profile_presets import env_default_profile

    exe = sys.executable
    argv: list[str] = [exe, "-m", "podcast_scraper.cli", "--output-dir", str(corpus_root)]
    if run_id:
        argv.extend(["--run-id", str(run_id)])
    try:
        op_text = operator_yaml.read_text(encoding="utf-8", errors="replace")
    except OSError:
        op_text = ""
    profile_name, _body = split_operator_yaml_profile(op_text)
    pn = profile_name.strip()
    if not pn:
        fallback = env_default_profile()
        if fallback:
            pn = fallback
    if pn:
        argv.extend(["--profile", pn])
    argv.extend(["--config", str(operator_yaml)])
    if feed_url:
        # Scoped per-feed incremental add: just this feed into the shared corpus, corpus layout.
        # The URL MUST be the positional ``rss`` arg — that is what populates ``config.rss_url``
        # (the single-feed scraping stage reads it). ``--rss`` binds to ``rss_extra`` (the plural
        # multi-feed list), which leaves ``config.rss_url`` None and the run dies at the scraping
        # stage with "RSS URL is required". A single ``rss_extra`` entry does NOT trigger the
        # multi-feed loop that would otherwise set rss_url per feed.
        argv.extend([str(feed_url), "--single-feed-uses-corpus-layout"])
        if skip_existing:
            argv.append("--skip-existing")
        if append:
            argv.append("--append")
        if max_episodes is not None:
            argv.extend(["--max-episodes", str(int(max_episodes))])
        if episode_offset is not None:
            argv.extend(["--episode-offset", str(int(episode_offset))])
        if episode_order:
            argv.extend(["--episode-order", str(episode_order)])
    else:
        spec = corpus_root / FEEDS_SPEC_DEFAULT_BASENAME
        # codeql[py/path-injection] -- request path anchor-guarded (Type 1; CODEQL_DISMISSALS.md).
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


def reconcile_jobs_inplace(
    jobs: list[dict[str, Any]],
    *,
    stale_seconds: int,
    prior_boot_alive: Callable[[dict[str, Any]], bool | None] | None = None,
    prior_boot_reason: str = "orphan_reconciled_no_container",
    stale_marks_live_processes: bool = True,
) -> list[str]:
    """Mutate *jobs* in place; return human-readable detail lines.

    Since #1653 this runs unattended every 30 s instead of only when an operator posts
    ``/api/jobs/reconcile``, which raises the bar on every rule here: a wrong call now
    happens silently and repeatedly. Two rules changed as a result.

    **The pid rule is boot-scoped** (see ``current_boot_id``). For a row from a previous boot
    the recorded pid is meaningless — in Docker exec mode it belonged to a compose client that
    died with the old API container while its job container kept running, and pid numbers get
    reused in the new PID namespace anyway.

    **Prior-boot rows need real evidence**, and *absence* of evidence must not free the slot.
    ``prior_boot_alive`` supplies it (see ``default_prior_boot_probe``): True leaves the row
    running, False marks it failed, and None means "cannot tell" — in which case the row keeps
    its slot for now rather than risk a second writer, and only the wall-clock rule below can
    eventually release it. This matters because a prior-boot ``running`` row has no monitor
    task any more, so nothing will *ever* finalize it on its own; reconcile is its only route
    to a terminal state, which is why the answer is correct reconciliation and not suppression.

    **A live process loses its slot to the wall clock only when a human asked.** The stale
    rule fires on rows whose liveness check just proved them *alive*, freeing the slot of a
    job that is still writing. Via ``POST /api/jobs/reconcile`` that is a deliberate operator
    override and stays available; on the sweeper's 30 s timer it would silently manufacture
    two concurrent corpus writers, so the sweeper passes
    ``stale_marks_live_processes=False`` and gets a WARNING instead. The default keeps the
    manual endpoint's long-standing behaviour — automation is what had to become cautious,
    not the operator.

    ``prior_boot_reason`` names the evidence the probe actually used, so the recorded
    ``error_reason`` stays truthful: a pid-based probe must not stamp a row "no container".
    """
    details: list[str] = []
    now = datetime.now(timezone.utc)
    boot_id = current_boot_id()
    for j in jobs:
        if j.get("status") != STATUS_RUNNING:
            continue
        job_id = str(j.get("job_id"))
        pid = j.get("pid")
        this_boot = j.get("boot_id") == boot_id
        stale_wall = False
        started = j.get("started_at")
        if stale_seconds > 0 and isinstance(started, str) and started.strip():
            try:
                ts = datetime.fromisoformat(started.replace("Z", "+00:00"))
                if (now - ts).total_seconds() > stale_seconds:
                    stale_wall = True
            except ValueError:
                pass

        alive: bool | None
        if this_boot and pid is not None and int(pid) > 0:
            alive = pid_alive(int(pid))
        elif not this_boot:
            # Prior-boot row: ask the probe, which knows what counts as evidence in this
            # exec mode. Never this function's own pid check.
            alive = prior_boot_alive(j) if prior_boot_alive is not None else None
        else:
            # This boot, spawned but no pid recorded yet — the promote→spawn window.
            alive = True

        # Prefer explicit orphan detection over wall-clock stale when both apply.
        if alive is False:
            reason = "orphan_reconciled_dead_pid" if this_boot else prior_boot_reason
            j["status"] = STATUS_FAILED
            j["ended_at"] = _utc_iso()
            j["error_reason"] = reason
            j["exit_code"] = -1
            # ``error_reason`` carries the slug for grouping; the detail line is read by a
            # human in a UI toast, so it says "dead pid" rather than the slug.
            human = reason.removeprefix("orphan_reconciled_").replace("_", " ")
            details.append(f"{job_id}: failed ({human})")
            continue
        if stale_wall:
            if alive and not stale_marks_live_processes:
                # Freeing a live process's slot is the one thing an *unattended* sweep must
                # not do. Louder than the reconcile it replaced, and non-destructive.
                logger.warning(
                    "job %s has run past the stale window (%ss) but is still alive; leaving it "
                    "running — cancel it explicitly if it is hung",
                    job_id,
                    stale_seconds,
                )
                continue
            j["status"] = STATUS_STALE
            j["ended_at"] = _utc_iso()
            j["error_reason"] = "wall_clock_stale"
            details.append(f"{job_id}: marked stale (wall-clock timeout)")
    return details


def _new_job_record(
    *,
    job_id: str,
    argv: list[str],
    log_relpath: str,
    status: str,
    command_type: str = COMMAND_FULL,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "job_id": job_id,
        "command_type": command_type,
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


def build_enrichment_argv(
    corpus_root: Path,
    *,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    corpus_only: bool = False,
    operator_yaml: Path | None = None,
    log_level: str = "INFO",
    profile: str | None = None,
    force: bool = False,
    with_ml: bool = False,
) -> list[str]:
    """Build CLI argv for an enrichment job (RFC-088 / Epic #1101).

    Mirrors ``build_pipeline_argv`` shape — the child process is
    ``python -m podcast_scraper.cli enrich`` (the ``enrich`` main-CLI subcommand,
    #1069 consistency), so it invokes, schedules, and runs in docker exactly like
    the pipeline. The subcommand delegates to the enrichment CLI verbatim.

    Note on ``sys.executable``: the argv is persisted on the registry row and executed later,
    possibly by a different process, so it embeds the interpreter path of whoever *built* it.
    Harmless today — in Docker mode ``_cli_argv_tail`` strips the interpreter prefix and
    compose supplies its own, and in subprocess mode builder and runner are the same container.
    It would break if a row enqueued from one container were later spawned via subprocess exec
    in another with a different layout (#1653 review).
    """
    argv: list[str] = [
        sys.executable,
        "-m",
        "podcast_scraper.cli",
        "enrich",
        "--output-dir",
        str(corpus_root),
        "--log-level",
        log_level,
    ]
    if only:
        argv.extend(["--only", ",".join(only)])
    if skip:
        argv.extend(["--skip", ",".join(skip)])
    if corpus_only:
        argv.append("--corpus-only")
    if operator_yaml is not None:
        argv.extend(["--config", str(operator_yaml)])
    # The profile decides WHICH enrichers run (``enricher_set_for_profile``). Without it the
    # child resolves an empty set — which is how every enrichment run in the corpus's life
    # became a 3 ms no-op reporting success (#1648). It now raises instead of no-opping, so an
    # omitted profile is loud rather than silent, but the fix is to pass it.
    if profile:
        argv.extend(["--profile", str(profile)])
    if force:
        argv.append("--force")
    # Some enrichers declare a provider_requirement and need an injected ML provider; without
    # this they are warned-and-skipped, which reads as "ran, produced nothing".
    if with_ml:
        argv.append("--with-ml")
    return argv


def _matching_queued_enrichment(
    jobs: list[dict[str, Any]], argv: list[str]
) -> dict[str, Any] | None:
    """An already-queued enrichment row whose stored argv is identical, if any.

    Compares the stored argv verbatim, which includes the builder's ``sys.executable``. The
    case this exists for — N per-feed pipeline runs each enqueueing a follow-up — always
    comes from the same image, so the interpreter path matches and they coalesce. A row
    enqueued from a *different* interpreter (host CLI vs container) will not match one from
    the container, and is left as a separate job rather than silently folded into it.
    """
    wanted = argv_summary(argv)
    for j in jobs:
        if (
            j.get("status") == STATUS_QUEUED
            and j.get("command_type") == COMMAND_ENRICHMENT
            and j.get("argv_summary") == wanted
        ):
            return j
    return None


def enqueue_enrichment_job(
    corpus_root: Path,
    *,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    corpus_only: bool = False,
    operator_yaml: Path | None = None,
    profile: str | None = None,
    force: bool = False,
    with_ml: bool = False,
    force_queued: bool = False,
) -> dict[str, Any]:
    """Enqueue a ``corpus_enrichment`` job; promote to running when slot free.

    Identical concurrency / queue / log-path semantics as
    ``enqueue_pipeline_job`` — the only difference is the
    ``command_type`` constant on the registry row and the argv
    builder. The promote-queued / cancel / reconcile / pid-alive
    paths in this module are ``command_type``-agnostic and reuse
    automatically.

    ``force_queued`` makes the row land as ``queued`` even when a slot is free (#1653). The
    RUNNING status is a promise that a process was started, and only the API server can keep
    it — ``start_job_if_running_record`` lives here and needs the app. A caller in another
    process (the pipeline, enqueueing its own follow-up enrichment) must therefore enqueue as
    QUEUED and let this server's drain promote it, or it would write a "running" row for a
    process nobody ever spawned.

    **This is not guaranteed to create a new row.** When an identical enrichment pass is
    already queued, the existing record is returned instead — callers must therefore read
    ``job_id`` from the return value rather than assume one enqueue means one new job.
    """

    def fn(jobs: list[dict[str, Any]]) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        log_relpath = f".viewer/jobs/{job_id}.log"
        argv = build_enrichment_argv(
            corpus_root,
            only=only,
            skip=skip,
            corpus_only=corpus_only,
            operator_yaml=operator_yaml,
            profile=profile,
            force=force,
            with_ml=with_ml,
        )
        # Coalesce against an identical pass that is already waiting. Enrichment reads the
        # whole corpus as it finds it, so two queued passes with the same argv do not produce
        # two different results — the second just re-does the first's work at full cost. This
        # became reachable when the post-pipeline chain started enqueueing (#1653): a reprocess
        # driven as N per-feed pipeline jobs would otherwise leave N identical corpus-wide
        # enrichment passes lined up behind it. Only ``queued`` rows coalesce; a ``running``
        # row is already reading files and a follow-up genuinely needs to run after it.
        existing = _matching_queued_enrichment(jobs, argv)
        if existing is not None:
            logger.info(
                "enrichment already queued as %s with identical argv; not enqueueing a duplicate",
                existing.get("job_id"),
            )
            return dict(existing)
        cap = max_concurrent_jobs()
        if not force_queued and _running_count(jobs) < cap:
            rec = _new_job_record(
                job_id=job_id,
                argv=argv,
                log_relpath=log_relpath,
                status=STATUS_RUNNING,
                command_type=COMMAND_ENRICHMENT,
            )
            jobs.append(rec)
            return rec
        rec = _new_job_record(
            job_id=job_id,
            argv=argv,
            log_relpath=log_relpath,
            status=STATUS_QUEUED,
            command_type=COMMAND_ENRICHMENT,
        )
        rec["started_at"] = None
        jobs.append(rec)
        return rec

    return with_jobs_locked_mutate(corpus_root, fn)


def enqueue_pipeline_job(
    corpus_root: Path,
    operator_yaml: Path,
    *,
    feed_url: str | None = None,
    skip_existing: bool = False,
    append: bool = False,
    max_episodes: int | None = None,
    episode_offset: int | None = None,
    episode_order: str | None = None,
) -> dict[str, Any]:
    """Append a new job; promote to *running* immediately when under the concurrency cap.

    When *feed_url* is given the run is scoped to that one feed (P1.4) — see build_pipeline_argv.
    """

    def fn(jobs: list[dict[str, Any]]) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        log_relpath = f".viewer/jobs/{job_id}.log"
        argv = build_pipeline_argv(
            corpus_root,
            operator_yaml,
            run_id=job_id,
            feed_url=feed_url,
            skip_existing=skip_existing,
            append=append,
            max_episodes=max_episodes,
            episode_offset=episode_offset,
            episode_order=episode_order,
        )
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


def docker_exec_mode() -> bool:
    """True when jobs run as sibling containers rather than child processes."""
    return os.environ.get("PODCAST_PIPELINE_EXEC_MODE", "").strip().lower() == "docker"


def default_prior_boot_probe() -> Callable[[dict[str, Any]], bool | None]:
    """How to tell whether a job from a *previous* server boot is still running.

    The honest answer differs by exec mode, which is why this is a function and not a constant:

    * **Docker mode** (what production runs) — the recorded pid was a compose client that died
      with the old API container, and the new container's PID namespace recycles those
      numbers, so the pid is worse than useless: it is actively misleading in both directions.
      Ask the daemon which containers still carry the job's label.
    * **Subprocess mode** — the child is a real process spawned with ``start_new_session``, so
      it is reparented rather than killed when the server exits, and it lives in the same PID
      namespace the new server sees. There the pid remains real evidence, and using it keeps
      the pre-existing behaviour for this mode.

    Known limitation, stated rather than papered over: subprocess mode *inside a container
    that restarts* gets a fresh PID namespace too, and this probe would then trust a recycled
    pid. Nothing in the repo runs that combination — prod sets
    ``PODCAST_PIPELINE_EXEC_MODE=docker`` (``compose/docker-compose.prod.yml``) — and fixing
    it properly needs a namespace identity the registry does not record today.
    """
    if docker_exec_mode():
        from podcast_scraper.server.pipeline_docker_factory import docker_job_alive

        return lambda row: docker_job_alive(str(row.get("job_id")))

    def _by_pid(row: dict[str, Any]) -> bool | None:
        pid = row.get("pid")
        if pid is None or int(pid) <= 0:
            return None
        return pid_alive(int(pid))

    return _by_pid


def apply_reconcile(
    corpus_root: Path,
    *,
    prior_boot_alive: Callable[[dict[str, Any]], bool | None] | None = None,
    stale_marks_live_processes: bool = True,
) -> tuple[int, list[str]]:
    """Reconcile registry under lock; return ``(updated_count, detail_lines)``.

    The liveness probe runs BEFORE the lock is taken. It shells out to ``docker ps`` in Docker
    mode, and the registry lock is cross-process — held here, it would stall the pipeline
    container's own enqueue for the duration of every probe. So this reads a snapshot, asks
    about the prior-boot rows outside the lock, then reconciles against the cached answers. A
    row that appears in between simply gets ``None`` (unknown) and keeps its slot until the
    next sweep, which is the safe direction.
    """
    probe = prior_boot_alive if prior_boot_alive is not None else default_prior_boot_probe()
    boot_id = current_boot_id()

    snapshot: list[dict[str, Any]] = with_jobs_locked_read(
        corpus_root, lambda jobs: [dict(j) for j in jobs]
    )
    answers: dict[str, bool | None] = {}
    for row in snapshot:
        if row.get("status") != STATUS_RUNNING or row.get("boot_id") == boot_id:
            continue
        try:
            answers[str(row.get("job_id"))] = probe(row)
        except Exception as exc:  # pragma: no cover — a probe must never break reconcile
            logger.warning("liveness probe failed job=%s: %s", row.get("job_id"), exc)
            answers[str(row.get("job_id"))] = None

    def fn(jobs: list[dict[str, Any]]) -> tuple[int, list[str]]:
        details = reconcile_jobs_inplace(
            jobs,
            stale_seconds=stale_after_seconds(),
            prior_boot_alive=lambda row: answers.get(str(row.get("job_id"))),
            prior_boot_reason=(
                "orphan_reconciled_no_container"
                if docker_exec_mode()
                else "orphan_reconciled_dead_pid"
            ),
            stale_marks_live_processes=stale_marks_live_processes,
        )
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
    this_boot = rec is not None and rec.get("boot_id") == current_boot_id()
    if not this_boot:
        # Do NOT signal a pid this process did not record. Same reasoning as the reconcile
        # rule (see ``current_boot_id``), but the consequence here is worse than a wrong
        # status: after a restart that pid number is very likely owned by something else in
        # the new PID namespace, and SIGTERM would kill an innocent process.
        stopped = _stop_prior_boot_job(job_id)
        if not stopped:
            logger.warning(
                "cancel job=%s: row is from a previous boot, so its recorded pid (%s) is not "
                "safe to signal and no container could be stopped; the row is marked cancelled "
                "but any surviving work must be stopped by hand",
                job_id,
                pid,
            )
        return "cancelled", rec
    if pid and int(pid) > 0:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError as exc:
            logger.warning("cancel sigterm failed job=%s pid=%s: %s", job_id, pid, exc)
    return "cancelled", rec


def _stop_prior_boot_job(job_id: str) -> bool:
    """Stop a job left over from an earlier boot by its container label. True if stopped.

    Only Docker exec mode has a durable handle on such a job — the label survives the API
    container that spawned it, which is exactly what the pid does not do.
    """
    if not docker_exec_mode():
        return False
    from podcast_scraper.server.pipeline_docker_factory import docker_stop_job

    return docker_stop_job(job_id)


def set_job_pid(corpus_root: Path, job_id: str, pid: int) -> None:
    """Persist child PID after spawn, tagged with the boot that observed it.

    ``boot_id`` travels with the pid because it is what makes the pid interpretable later:
    see ``current_boot_id``. A row without one is a pre-#1653 row, and is treated as
    prior-boot (i.e. its pid is not trusted as evidence).
    """

    def fn(jobs: list[dict[str, Any]]) -> None:
        for j in jobs:
            if str(j.get("job_id")) == job_id:
                j["pid"] = int(pid)
                j["boot_id"] = current_boot_id()
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
        # Promotion only flips the lifecycle state — it must NOT rewrite the
        # command. The enqueue-time (command-typed) ``argv_summary`` is the
        # source of truth; overwriting it here is what made queued enrichment
        # jobs spawn the pipeline.
        j["status"] = STATUS_RUNNING
        j["started_at"] = _utc_iso()
        j["cancel_requested"] = False
        j["pid"] = None
        return dict(j)

    return with_jobs_locked_mutate(corpus_root, fn)


def _factory_accepts_job_id(factory: Any) -> bool:
    """True when *factory* declares a ``job_id`` parameter (or accepts ``**kwargs``)."""
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # pragma: no cover — builtins / C callables
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "job_id" in params


async def spawn_pipeline_subprocess(
    app: Any,
    corpus_root: Path,
    job_id: str,
    argv: list[str],
    log_abs: Path,
) -> asyncio.subprocess.Process:
    """Spawn the pipeline child (or delegate to ``app.state.jobs_subprocess_factory``).

    The default path pipes subprocess stdout through a capped pump
    (:func:`_pump_subprocess_to_log`, #666 review #10) so runaway pipelines
    cannot fill the disk with log output. The Docker factory manages log
    capture via compose and is not routed through the pump here.
    """
    factory = getattr(app.state, "jobs_subprocess_factory", None)
    if factory is not None:
        # The Docker factory wants ``job_id`` so it can label the container it spawns, which
        # is what lets reconcile ask the daemon whether a prior-boot job is really still
        # running. Older/test factories take the three positional arguments only, so the
        # keyword is offered rather than imposed — checked by signature rather than by
        # catching TypeError, which would silently swallow a TypeError raised *inside* a
        # factory and retry the spawn.
        if _factory_accepts_job_id(factory):
            proc = await factory(argv, corpus_root, log_abs, job_id=job_id)
        else:
            proc = await factory(argv, corpus_root, log_abs)
        return cast(asyncio.subprocess.Process, proc)

    log_abs.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(corpus_root),
        start_new_session=os.name != "nt",
    )
    # Start the pump concurrently. ``monitor_subprocess`` awaits both
    # ``proc.wait()`` and this task via the ``_ps_log_pump`` attribute.
    assert proc.stdout is not None
    pump_task = asyncio.create_task(
        _pump_subprocess_to_log(
            proc.stdout,
            log_abs,
            max_bytes=job_log_max_bytes(),
            job_id=job_id,
        ),
        name=f"pipeline-log-pump-{job_id}",
    )
    setattr(proc, "_ps_log_pump", pump_task)
    return proc


async def _finalize_job(
    corpus_root: Path,
    job_id: str,
    *,
    exit_code: int | None,
    cancelled: bool,
) -> str | None:
    """Finalize the job registry row and return the log_relpath on failure (else None).

    The returned relpath is used by the caller (monitor_subprocess) to parse
    the job log for a human-readable error_reason AFTER the pump finishes
    draining — the file is not fully written until after the pump task completes,
    so parsing cannot happen inside this function.
    """
    log_relpath_holder: list[str] = []

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
                lr = j.get("log_relpath")
                if lr:
                    log_relpath_holder.append(str(lr))

    await asyncio.to_thread(with_jobs_locked_mutate, corpus_root, fn)
    return log_relpath_holder[0] if log_relpath_holder else None


def _patch_error_reason_from_log(corpus_root: Path, job_id: str, log_relpath: str) -> None:
    """Replace the placeholder ``exit_code_N`` error_reason with a parsed cause.

    Called (in a thread) after the log pump finishes so the file is complete.
    No-op when parsing yields nothing — the exit_code_N fallback stays.
    """
    log_abs = corpus_root / log_relpath
    parsed = _parse_error_reason_from_log(log_abs)
    if not parsed:
        return

    def fn(jobs: list[dict[str, Any]]) -> None:
        for j in jobs:
            if str(j.get("job_id")) != job_id:
                continue
            # Only overwrite the auto-generated ``exit_code_N`` placeholder;
            # leave any reason set by reconcile / spawn-failed / cancel as-is.
            current = j.get("error_reason") or ""
            if current.startswith("exit_code_"):
                j["error_reason"] = parsed

    with_jobs_locked_mutate(corpus_root, fn)


async def monitor_subprocess(
    app: Any,
    corpus_root: Path,
    job_id: str,
    proc: asyncio.subprocess.Process,
) -> None:
    """Wait for the child, drain the log pump, finalize registry, then try
    to promote queued work.

    ``_ps_log_fp`` (legacy file-handle path, used by the Docker factory) and
    ``_ps_log_pump`` (default PIPE+pump path, #666 review #10) are mutually
    exclusive; whichever is present is cleaned up in the finally block.
    """
    log_fp = getattr(proc, "_ps_log_fp", None)
    log_pump = getattr(proc, "_ps_log_pump", None)
    failed_log_relpath: str | None = None
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
        failed_log_relpath = await _finalize_job(
            corpus_root, job_id, exit_code=code, cancelled=cancelled
        )
        await drain_queue_async(app, corpus_root)
    finally:
        # #666 review #10: wait for the pump task to finish draining stdout
        # AFTER the child exited — otherwise the last buffered chunks are
        # lost. The pump owns the file handle internally and closes it.
        if log_pump is not None:
            try:
                await log_pump
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("job log pump task failed job=%s: %s", job_id, exc)
        # #666 review #11: log cleanup failures instead of silently
        # swallowing them — disk-full / readonly-fs conditions leave the
        # operator with no signal that pipeline output was truncated.
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("job log close failed job=%s: %s", job_id, exc)

    # Parse the fully-written log for a human-readable failure cause, then
    # fire the downstream webhook/prometheus notification with the final state.
    # Both run after the finally block so the log file is guaranteed complete.
    if failed_log_relpath:
        try:
            await asyncio.to_thread(
                _patch_error_reason_from_log, corpus_root, job_id, failed_log_relpath
            )
        except Exception:
            logger.exception("error_reason log parse failed job=%s", job_id)

    # Fire-and-forget downstream notification. No-op when
    # PODCAST_JOB_WEBHOOK_URL is unset (default). Failures are logged
    # but never propagate — webhook outages must not break finalize.
    # See src/podcast_scraper/server/job_webhook.py + RFC-081 §Layer 4.
    rec_after = await asyncio.to_thread(get_job, corpus_root, job_id)
    if rec_after is not None:
        try:
            from podcast_scraper.server.pipeline_run_prometheus import (
                observe_pipeline_terminal_metrics,
            )

            await asyncio.to_thread(observe_pipeline_terminal_metrics, corpus_root, rec_after)
        except Exception:
            logger.exception(
                "pipeline prometheus observation failed job=%s",
                job_id,
            )
        from podcast_scraper.server.job_webhook import emit_job_state_change

        await emit_job_state_change(rec_after)


def argv_from_record(job: dict[str, Any]) -> list[str] | None:
    """The exact command to spawn, taken from the job's stored ``argv_summary``.

    The registry row is the single source of truth for *what to run*: the argv
    is built once at enqueue (command-typed via ``build_pipeline_argv`` /
    ``build_enrichment_argv`` / …) and persisted. Every spawn executes it
    verbatim, so each ``command_type`` runs its own CLI instead of always the
    pipeline. Returns ``None`` for a legacy/blank row so the caller can fall
    back to rebuilding the pipeline argv.
    """
    raw = job.get("argv_summary")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, list) and parsed and all(isinstance(a, str) for a in parsed):
        return [str(a) for a in parsed]
    return None


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
    # Source of truth = the row's stored (command-typed) argv; rebuild only for
    # a legacy row that predates argv persistence. This is what lets an
    # enrichment (or any non-pipeline) job spawn its own CLI, not the pipeline.
    argv = argv_from_record(job) or build_pipeline_argv(corpus_root, operator_yaml, run_id=job_id)
    log_abs = corpus_root / str(job.get("log_relpath", f".viewer/jobs/{job_id}.log"))
    try:
        proc = await spawn_pipeline_subprocess(app, corpus_root, job_id, argv, log_abs)
    except Exception as exc:
        # #666 review #9: full ``str(exc)`` can include absolute paths,
        # environment variable names, or internal stack-trace fragments
        # that are forwarded to the viewer via ``error_reason``. Capture
        # only the exception type in the registry; the full message stays
        # server-side in ``logger.exception`` above.
        logger.exception("spawn failed job=%s", job_id)
        err_code = type(exc).__name__

        def _fail_mark_spawn_failed(jobs: list[dict[str, Any]]) -> None:
            for j in jobs:
                if str(j.get("job_id")) != job_id:
                    continue
                j["status"] = STATUS_FAILED
                j["ended_at"] = _utc_iso()
                j["error_reason"] = f"spawn_failed: {err_code}"
                j["exit_code"] = -1

        await asyncio.to_thread(with_jobs_locked_mutate, corpus_root, _fail_mark_spawn_failed)
        await drain_queue_async(app, corpus_root)
        return

    if proc.pid:
        await asyncio.to_thread(set_job_pid, corpus_root, job_id, proc.pid)
    _watch_in_background(app, corpus_root, job_id, proc)


#: Strong references to in-flight monitor tasks. ``asyncio`` only holds a *weak* reference to
#: a running task, so a task nobody keeps can be garbage-collected mid-await — the job would
#: then never be finalized and its row would strand exactly like the ghosts #1653 exists to
#: clear. Each task discards itself on completion, so this set tracks live jobs, not history.
_MONITOR_TASKS: set[asyncio.Task] = set()


def _watch_in_background(
    app: Any,
    corpus_root: Path,
    job_id: str,
    proc: asyncio.subprocess.Process,
) -> None:
    """Await the child's exit in a background task instead of in the caller.

    ``monitor_subprocess`` blocks on ``proc.wait()`` for the job's entire lifetime. Awaiting
    it here would make every caller of ``drain_queue_async`` block for as long as the job it
    just promoted runs — hours. The HTTP route never noticed because it dispatches through
    ``background_tasks.add_task``, but two callers are not shielded:

    * the queue sweeper's startup sweep, which runs inside the FastAPI lifespan — so a queued
      row plus a free slot at boot (precisely the wedged state the sweeper exists to fix)
      would stall startup until the queue drained, uvicorn would never serve, and a
      healthcheck would kill the container — taking out the compose client of the job it had
      just promoted;
    * the sweeper's own loop, whose 30 s reconcile would pause for the duration of any job it
      promoted — i.e. exactly while jobs are running and can die unfinalized.

    Backgrounding the *monitor* rather than the whole spawn keeps ``drain_queue_async``'s
    contract intact: when it returns, every promotable job has actually been spawned and its
    pid recorded. Only the waiting moved.
    """

    async def _run() -> None:
        await monitor_subprocess(app, corpus_root, job_id, proc)

    task = asyncio.create_task(_run(), name=f"job-monitor-{job_id}")
    _MONITOR_TASKS.add(task)

    def _done(t: asyncio.Task) -> None:
        _MONITOR_TASKS.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            # Nobody awaits this task, so without this the traceback would surface only as
            # asyncio's "exception was never retrieved" at GC time, long after the fact.
            logger.error("job monitor task failed job=%s: %s", job_id, exc, exc_info=exc)

    task.add_done_callback(_done)


async def drain_queue_async(app: Any, corpus_root: Path) -> None:
    """Start queued jobs until the concurrency cap is reached.

    Returns once every job the cap allows has been *spawned*; each one's completion is then
    watched in the background (see ``_watch_in_background``). The loop cannot overshoot
    ``max_concurrent_jobs``: ``promote_queued_if_slot`` flips the row to RUNNING inside the
    registry lock before returning it, so the next iteration counts it.
    """
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
