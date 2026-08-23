# ADR-152: Job-queue supervision — who moves the queue, and what counts as evidence a job is alive

- **Status**: Proposed
- **Date**: 2026-08-15
- **Authors**: Marko Dragoljevic
- **Related RFCs**: [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) §Phase 2
  (job registry, stale detection, reconcile)
- **Related ADRs**: [ADR-084](ADR-084-full-stack-docker-compose-topology.md) (Docker job
  execution), [ADR-116](ADR-116-privilege-split-public-control-api.md) (enqueue-only control API)
- **Issues**: #1653

## Context & Problem Statement

The job queue had no heartbeat. Two correct functions existed and nothing called them on a
schedule:

- `drain_queue_async` promotes a queued job when a slot frees. It was called from exactly two
  places, both inside a job's own lifecycle — after its child exits, and after a spawn failure.
  Promotion was therefore **edge-triggered on another job finishing**.
- `apply_reconcile` releases a `running` row whose process is gone. It was called from
  `POST /api/jobs/reconcile` only, i.e. by hand.

Together they let the queue stop permanently: a job dies without finalising (container killed,
OOM, API restart), its row stays `running` and keeps counting toward `max_concurrent_jobs`,
and every queued job then waits on an event that can no longer happen. Observed, not
theorised — enrichment job `ef9f8f9c` sat queued for 7.75 hours. RFC-077 §Phase 2 anticipated
this and called for a "periodic lightweight check (optional background task in server)"; it was
never built.

This became urgent rather than untidy when the post-pipeline enrichment chain switched from a
detached `subprocess.Popen` to enqueueing a real job (#1653): work that used to bypass the
queue entirely now depends on the queue actually moving.

## Decision

### 1. A periodic sweeper owns queue liveness

`server/queue_sweeper.py` runs `reconcile → drain` every 30 s, plus once immediately at
startup. Order matters: freeing a ghost slot is what makes a promotion possible in the wedged
case, so draining first would compute against a slot count it is about to invalidate. The
startup sweep matters as much as the loop — a restart is precisely when ghost rows exist.

### 2. Monitoring a job never blocks the caller that started it

`monitor_subprocess` awaits `proc.wait()` for the job's whole lifetime. Awaiting it inline made
`drain_queue_async` return only once the job it promoted had *finished*. The HTTP route never
noticed because it dispatches through `background_tasks`; the sweeper is not behind anything,
and its startup sweep runs inside the FastAPI lifespan. A queued row plus a free slot at boot —
the exact state the sweeper exists to fix — would have stalled startup until the queue drained,
so uvicorn would never have begun serving and a healthcheck would have killed the container,
taking out the compose client of the job it had just promoted.

The monitor now runs in a background task with a strong reference held (`asyncio` keeps only a
weak one, and a collected monitor would strand the job it was watching). Backgrounding the
*monitor* rather than the whole spawn preserves the drain contract: when `drain_queue_async`
returns, every promotable job has been spawned and its pid recorded.

### 3. A pid is evidence only for the boot that recorded it

This is the load-bearing decision, and it is a direct consequence of ADR-084. In Docker exec
mode — what production runs (`compose/docker-compose.prod.yml`) — the pid stored on a job row
belongs to the `docker compose run` **client** process inside the API container. The work runs
in a container on the host daemon. Therefore:

- **False-failed.** Replace the API container and every recorded pid dies with it, while the
  job containers keep running. An automatic dead-pid rule would mark a live job failed and free
  its slot for a second concurrent corpus writer.
- **False-alive.** The new container's PID namespace recycles low pid numbers, so an unrelated
  process can make a dead job look alive — the ghost keeps its slot, and `cancel_job` would
  SIGTERM whatever now owns that number.

Both faults are *created by automating* a check that was previously a human's judgement call.
So `set_job_pid` records a per-process `boot_id` alongside the pid, and the dead-pid rule
applies only to rows whose `boot_id` matches the current boot. Rows without one (written before
this change) count as prior-boot.

### 4. Prior-boot rows are judged by ground truth, and unknown never frees a slot

Boot-scoping alone would leave a genuinely dead prior-boot row holding its slot until the
24 h wall-clock window — the original wedge, unfixed, for the restart case that causes it most
often. Suppressing reconciliation is not an option either: a prior-boot `running` row has no
monitor task any more, so nothing will *ever* finalize it, even on a clean container exit.
Reconcile is its only route to a terminal state. The answer has to be *correct* reconciliation.

Every job container is therefore labelled `ps.job_id=<job_id>` at spawn, and reconcile asks the
daemon (`docker ps --filter label=…`) about prior-boot rows. The probe is tri-state:

| Answer | Meaning | Action |
| --- | --- | --- |
| `True` | container present | leave `running`, keep the slot |
| `False` | no container | mark `failed` |
| `None` | docker CLI missing, daemon unreachable, probe timed out | **keep the slot** |

`None` is deliberately distinct from `False`. Guessing "dead" on absent evidence admits a
second writer to the corpus; guessing "alive" costs one sweep interval. In subprocess exec
mode the probe falls back to the pid, which there is a real worker in a stable PID namespace.

The probe runs **before** the registry lock is taken — it shells out, and the lock is
cross-process, so holding it would stall the pipeline container's own enqueue.

### 5. Cancel never signals a pid this process did not record

The same rule, with sharper consequences. `cancel_job` sends SIGTERM to the pid on the row.
Reconcile getting boot identity wrong produces a wrong *status*; cancel getting it wrong
**kills an unrelated process** that inherited the recycled pid number. A prior-boot cancel
therefore stops the job by container label (`docker stop`) in Docker mode, and in subprocess
mode marks the row cancelled while logging loudly that surviving work must be stopped by hand
— never a blind signal.

### 6. Only a human may evict a live long-running job

The wall-clock rule marks a row `stale` and frees its slot even when the liveness check just
proved the process **alive**. Through `POST /api/jobs/reconcile` that is a deliberate operator
override and it stays. On a 30 s timer it would silently manufacture two concurrent corpus
writers, so the sweeper passes `stale_marks_live_processes=False` and logs a WARNING instead.
Automation had to become cautious; the operator did not.

### 7. Promotion is pausable

`.viewer/jobs.paused` holds promotion while still reconciling, so the registry stays truthful
during the pause. Without it, starting the API server *is* starting whatever is queued —
acceptable normally, wrong during a corpus repair, and wrong when a repair driven as a plain
CLI run holds no registry slot and so cannot serialise against a queued enrichment pass.

### 8. Identical queued enrichment passes coalesce

Enrichment reads the whole corpus as it finds it, so two *queued* passes with equal argv cannot
produce two different results. A reprocess driven as N per-feed pipeline jobs would otherwise
line up N identical corpus-wide passes. `enqueue_enrichment_job` returns the waiting row
instead of appending — so callers must read `job_id` from the return value rather than assume
one call means one new job. Only `queued` rows coalesce; a `running` pass is already reading
files and a follow-up genuinely needs to run after it.

## Invariants

- **RUNNING is a promise that a process was started**, and only the API server can keep it
  (`start_job_if_running_record` needs the app). Cross-process callers enqueue as `queued`.
- **An unattended sweep never frees the slot of a process it believes alive.**
- **Absence of evidence is not evidence of death** — tri-state liveness, never a boolean.
- Promotion cannot overshoot `max_concurrent_jobs`: `promote_queued_if_slot` flips the row to
  RUNNING inside the registry lock before returning it, so concurrent drains count it.

## Consequences

- Job completion is no longer synchronous with the request that promoted it. Tests must poll
  for a terminal state rather than assume it (`TestClient` also needs its context manager, or
  the per-request portal is torn down before the monitor task can finish).
- `error_reason` gains `orphan_reconciled_no_container` alongside `orphan_reconciled_dead_pid`,
  naming which evidence was actually used.
- A hung-but-alive job now needs an explicit cancel; it will not time itself out. The 30 s
  WARNING is the signal.

## Known limitations

Stated rather than papered over:

- **Subprocess exec mode inside a restarting container** gets a fresh PID namespace, and the
  fallback probe would trust a recycled pid. Nothing in the repo runs that combination — prod
  is Docker mode — and fixing it properly needs a namespace identity the registry does not
  record.
- **`docker compose run` client survival** — that an attached client's death leaves the
  container running is standard daemon behaviour, but it has not been verified end-to-end in
  this stack. Layer 3/4 above are designed so that being wrong about it is safe: the row keeps
  its slot rather than losing it.
- **`PODCAST_JOB_STALE_SECONDS` defaults to 24 h.** The longest observed job ran ~2.16 h, but
  that was not a 678-episode reprocess. If a single job can exceed the window, raise it for the
  duration or chunk the work; the live-pid guard above keeps the failure mode to "row stays
  running" rather than "two writers".

## Non-Goals

- Not a general supervisor — it does not restart failed jobs, only reports them accurately.
- Not a scheduler — ordering stays FIFO by `created_at`.
