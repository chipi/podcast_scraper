# ADR-150: Supervision bounds, crash-only recovery, and absence-detection for the pipeline

- **Status**: Accepted (Phase 1 partially implemented, unverified by CI)
- **Date**: 2026-08-12
- **Authors**: Marko Dragoljevic, Claude
- **Related RFC**: [RFC-117](../rfc/RFC-117-pipeline-supervision-and-absence-detection.md) (the full argument)
- **Related ADRs**: [ADR-119](ADR-119-vendor-neutral-event-emission.md) (events, not scraped metrics), [ADR-142](ADR-142-litellm-prod-gateway.md) (prod-local gateway), [ADR-145](ADR-145-channel-agnostic-outbox-seam.md) (tailnet seam pattern)
- **Related issues**: [#1620](https://github.com/chipi/podcast_scraper/issues/1620) (programme), [#379](https://github.com/chipi/podcast_scraper/issues/379) (non-enforcing timeout), [#429](https://github.com/chipi/podcast_scraper/issues/429) (fail-fast)

## Context & Problem Statement

Two production outages on 2026-08-12 were **one defect with two endings**. `CostCapExceeded`
raised in the main thread from a region of `orchestration.py` with no `try/finally`, orphaning
a non-daemon worker whose continue-predicate defaults to `True`. With nothing queued it span
for 4 h 15 m in total silence; with one job queued it raised
`RuntimeError: cannot schedule new futures after interpreter shutdown` and discarded the run.

Three separate properties were absent: nothing **bounded** the failure, nothing **recovered**
from it, and nothing **detected** it. Full evidence and argument in RFC-117.

## Decisions

### D1 — Supervision bounds are unconditional, and liveness is one of them

The worker loop terminates on **main-thread death** or a **wall-clock budget**, evaluated
*before* any queue-state logic. A worker must never outlive its parent.

**Why unconditional ordering matters:** the incident was precisely the case where queue-state
branches could never fire. A bound reachable only through those branches would not have helped.

Default budget 4 h (~2× the longest legitimate run), overridable via
`processing_loop_budget_seconds`, disabled only by an explicit `<= 0`. **A malformed value
falls back to the default, never to unbounded** — a typo must not be able to recreate the
incident.

### D2 — Scheduling is defended, not just work

Both `executor.submit()` sites are guarded. A scheduling failure un-marks the episode, stops
submission, and returns cleanly rather than propagating.

The prior asymmetry is the lesson worth generalising: `_process_single_processing_job` wrapped
the *work* in `try/except` while the *scheduling* of that work was bare. Defensiveness in the
small does not compose into safety in the large.

### D3 — Abandoning a stuck unit must not wait on it

`ThreadPoolExecutor` is managed explicitly rather than via `with`, because `__exit__` calls
`shutdown(wait=True)` and would block on the very future the bounds exist to escape. The abort
path uses `shutdown(wait=False, cancel_futures=True)`; the normal path keeps `wait=True`.

### D4 — Crash-only recovery; the corpus is the ledger

The only recovery verb is **kill and resubmit**, bounded by an attempt count. GUID-keyed
`skip_existing` already makes this free and idempotent — it is why both outages cost time and
zero data.

**Rejected:** re-attaching to an orphaned child after an api restart. The stdout pipe cannot be
rewired, and crash-only relaunch is already proven safe in production. **Rejected:** a queue
broker; unjustified on a single VPS when the corpus already serves as the ledger.

### D5 — Budget exhaustion is a planned outcome, not an exception

`CostCapExceeded` becomes a graceful stop — set stop event, drain, finalize, report a distinct
`budget_exhausted` terminal reason. A guardrail firing is a *success* of the guardrail and must
not be expressed as an exception racing three threads.

### D6 — Deadlines belong at the transport layer

Priority order: transport timeout on every outbound call (tight) → episode budget (loose) →
job budget (looser). Hangs live in network calls, which should complete in seconds-to-minutes
regardless of episode length; episode-level deadlines must otherwise be so loose they detect a
hang an hour late.

Enforced by a **CI lint** that every outbound HTTP call in pipeline paths carries an explicit
`timeout=`. One un-timeouted call is the entire hang class.

**`utils/timeout.py` is not a deadline.** It cannot interrupt — `TimeoutError` is raised only
after the block returns. It is retained solely as a *detection* signal (an ERROR log while the
operation is still stuck) and documented as such. It must never again be counted as protection.

### D7 — Liveness and progress are different signals; alerting needs progress

**A heartbeat alone would not have caught the wedge** — the loop was iterating the whole time
and would have reported "alive" for four hours. Therefore:

- **liveness** — the loop iterates (catches death, hard blocks)
- **progress watermark** — monotonic completed-unit count plus `last_progress_at` (catches
  *alive but stuck*)

The **child emits**, the **api sweeper evaluates**, **systemd + external alerting cover the
api**. Each layer is watched by a dumber, more reliable one.

Alert on progress *age*, not on episode-completion timing:

```promql
podcast_job_running == 1 and (time() - podcast_job_last_progress_timestamp_seconds) > 1800
```

A flat 30-minute threshold is only safe **once D6 exists** — transport timeouts are what
guarantee sub-episode watermark movement at minutes-scale despite a 4–90 minute episode
spread. D6 is a prerequisite for cheap alerting, not parallel work.

### D8 — `stale` is set by the api sweeper, never self-reported

A 60 s asyncio task extends `reconcile` with progress-age logic, sets `stale`, SIGTERMs per
policy, and fires the existing `emit_job_state_change` webhook. The child never self-marks — it
is the thing that dies.

`reconcile`'s existing pid-liveness test is **retained but insufficient**: it returned
`updated: 0` during the wedge because the pid was alive. Progress-age is the addition, not a
replacement.

### D9 — Notification reuses the existing webhook; no new dashboards

Route `stale`/`failed` through `PODCAST_JOB_WEBHOOK_URL` to a phone push, plus vmalert →
Alertmanager as an api-independent second channel. The failure being fixed is "a human had to
go looking" — another dashboard reproduces it.

### D10 — Cost cap stays `abort`, with the threshold raised

Raised to `$10.00` per run in the box's `viewer_operator.yaml` (was `$5.00`, inherited from
`config/profiles/cloud_balanced.yaml:177-178`).

**`action: warn` was considered and rejected.** It would make the buggy `abort` path
unreachable — attractive short-term — but disables a guardrail to dodge a bug that D5 fixes
properly. Real spend is separately bounded by the podcast-prod LiteLLM key cap.

Note the cap measures **modelled** cost including transcription billed against a free Deepgram
allowance, so it constrains a number that does not correspond to the actual bill. Recalibrating
it against real spend is deferred, not decided.

## Consequences

**Positive.** Both incident presentations become impossible. Unknown future failures surface
within ~30 minutes instead of four hours. Most of the work is un-gating and wiring existing
primitives rather than new subsystems.

**Negative / accepted.**

- Abandoned futures leak threads until process exit. Accepted: the process is short-lived and
  crash-only recovery is the containment.
- A wall-clock budget can truncate a legitimately slow run. Mitigated by a generous default and
  an explicit opt-out.
- Auto-resubmit plus a deterministically-fatal episode is an infinite loop until the poison-pill
  memo (Phase 2) exists. **This is a real hazard introduced by D4 and must land before
  auto-resubmit is enabled.**

**Unverified, flagged.** Whether `skip_existing` treats a half-written episode as present is
load-bearing for D4 and has not been checked. Phase-1 code landed on
`fix/pipeline-resilience-supervision` has never been executed — the authoring environment had
no venv or pytest, so `ast.parse` is the entire extent of local verification.

## Alternatives considered

**Fix the bugs first, add containment later.** Rejected. The containment *is* the fix — the
minimal correct change to the loop (bounds, guarded submits, coordinated shutdown) is the same
diff. Shipping a watchdog that watches a loop known to be broken is process theatre. But note
the converse was also rejected: "containment before bugfix" as a general principle was a
rationalisation in this case, because the two were never separable.

**Per-episode isolation as the primary fix.** Rejected as already-present. Isolation exists
(`processing.py:1726-1745`); what was missing is *cancellation*. An isolated unit that hangs
forever isolates nothing.

**A supervision framework / external orchestrator.** Rejected as disproportionate for a
single-VPS deployment whose api already supervises subprocesses and whose corpus already
functions as a ledger.
