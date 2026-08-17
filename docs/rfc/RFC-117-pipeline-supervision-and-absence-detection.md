# RFC-117: Pipeline supervision — containment, recovery, and absence-detection

- **Status**: Draft — Phase 1 partially landed on `fix/pipeline-resilience-supervision`, unverified by CI
- **Authors**: Marko Dragoljevic (chipi), Claude
- **Stakeholders**: Operator (sign-off), pipeline + platform maintainers
- **Related ADRs**: [ADR-150](../adr/ADR-150-supervision-bounds-and-absence-detection.md) (the decisions this RFC argues for), [ADR-119](../adr/ADR-119-vendor-neutral-event-emission.md) (pipeline is an events source), [ADR-142](../adr/ADR-142-litellm-prod-gateway.md) (prod-local gateway)
- **Related Documents**: `docs/wip/INCREMENTAL-ROLLOUT-FOLLOWUPS-2026-08-11.md` (incident log, F1–F15 + G1/G2)
- **Anchors**: [#1620](https://github.com/chipi/podcast_scraper/issues/1620) (this programme), [#379](https://github.com/chipi/podcast_scraper/issues/379) (the timeout that never worked), [#429](https://github.com/chipi/podcast_scraper/issues/429) (fail-fast / max-failures)

> Written after two production outages on 2026-08-12 that turned out to be **one defect with
> two endings**. The proximate bugs matter less than the fact that nothing bounded them and
> nothing noticed them.

---

## 1. Context

A corpus backfill drove the pipeline from 180 to ~500 episodes via the operator API. Two runs
failed. Both had the same origin.

`CostCapExceeded` is raised in the **main thread** by `check_cost_soft_cap_at_stage`
(`orchestration.py:2207`). That call sits inside a ~150-line region between
`processing_thread.start()` (2110) and `processing_thread.join()` (2251) which has **no
`try/finally`**. When it raises, control unwinds past `transcription_complete_event.set()`
(2246), leaving a non-daemon `ProcessingProcessor` thread alive with a continue-predicate,
`_should_continue_processing` (`processing.py:1555-1562`), that ends in `return True`.

| | Trigger | Ending |
| --- | --- | --- |
| Incident 1 | `09:28:30Z` `$12.4599 > $5.0000` | nothing left to submit → spin at 0.05 s/iter for **4 h 15 m**, live pid, ~2.5 % CPU, **zero log output**, cancelled by hand |
| Incident 2 | `16:29:13Z` `$5.6425 > $5.0000` | 12 s later one job was queued → `executor.submit` into a shutting-down interpreter → `RuntimeError: cannot schedule new futures after interpreter shutdown` |

Which ending occurs is decided by whether a job happens to be queued when teardown begins.
There is no third, healthy branch.

**The guardrail was working.** A `$5.00` per-run soft cap
(`config/profiles/cloud_balanced.yaml:177-178`, `action: abort`) correctly detected an
overrun. The defect is that a *planned* outcome — budget exhaustion — is expressed as an
exception thrown across a three-thread pipeline with no coordinated shutdown.

### 1.1 What did not go wrong

Worth stating, because it constrains the design. **No data was lost in either incident.**
Corpus coverage after both was `with_gi == with_kg == total_episodes`, `with_neither: 0`.
GUID-keyed `skip_existing` made cancel-and-relaunch free and idempotent; 32/36 and 12/15
episodes were intact and fully enriched.

That primitive is the foundation everything below is built on, and must not be traded away.

---

## 2. The reframing

The instinct after an outage is to add defenses. That instinct is wrong here.

This codebase already contains: a per-job `try/except` with per-episode failure recording
(`processing.py:1726-1745`), a retryable/terminal classifier (`processing.py:175`), a
`timeout_context` on every episode (`processing.py:1684-1688`), a `stale` job status, a
`reconcile` endpoint, a `.pipeline_status.json` writer (`monitor/status.py:57-90`),
`run.jsonl` `episode_finished` events, a `pipeline_progress` event (ADR-119), and a job
webhook (`server/jobs.py:893-895`).

Every one of them is gated off, non-enforcing, pull-only, or terminal-only.

The sharpest example: `utils/timeout.py` starts a `threading.Timer` that sets a flag; the
`TimeoutError` is raised **after the `yield` returns**, i.e. only once the operation has
already finished on its own. It cannot interrupt a blocked socket read. Issue #379 built it
"to prevent hangs". A 1200 s per-episode deadline was active throughout both incidents and
prevented neither.

> **The system is not undefended. It is defensively decorated. It does not need more
> defenses — it needs a supervisor that makes the ones it already has fire.**

This roughly halves the cost of the programme: most of the work is wiring, un-gating, and
enforcement, not new subsystems.

---

## 3. Three pillars

They fail independently and must be designed independently.

| Pillar | Question it answers | How it failed on 2026-08-12 |
| --- | --- | --- |
| **Containment** | does one unit's failure stay local? | nothing had an enforced deadline; one raise killed a whole run |
| **Recovery** | do we get back to work? | recovery was a human noticing and re-POSTing |
| **Detection** | do we *know*, and are we *told*? | 4 h 15 m of silence read as healthy |

A perfectly contained pipeline that freezes silently is still a four-hour outage.

### 3.1 Containment

The atomic unit for **retry** is the episode — GUID-keyed, proven. The atomic unit for
**deadlines** is not: episodes legitimately run 4–90 minutes, so an episode-level deadline
must be so loose it detects a hang an hour late.

Hangs live in **network calls**, which should complete in seconds-to-minutes regardless of
episode length. The deadline hierarchy is therefore:

1. **transport timeout on every outbound call** — tight; kills the hang class at source
2. **episode budget** — loose; catches pathological loops
3. **job budget** — looser; catches everything else

Isolation already exists. What is missing is **cancellation**: an isolated unit that can hang
forever isolates nothing.

### 3.2 Recovery

Crash-only. `skip_existing` already makes "kill and resubmit" safe, so the supervisor's only
recovery verb should be exactly that, bounded by an attempt count. **The corpus is the
ledger**; no queue broker is required on a single VPS.

This collapses a large amount of would-be design. It also creates one new hazard —
see §5 poison pill.

### 3.3 Detection

The generalised failure: **every detection mechanism here is exception-triggered.** Logs
record what raises; Sentry records what raises; job status changes when something raises or
exits. Nothing is absence-triggered. A process that stops working while staying alive is
invisible, and silence is indistinguishable from healthy quiet operation.

Measured, during the wedge versus a healthy run:

```
{job="podcast-pipeline"} 10:30–14:30Z (wedged)   ->   0 log lines
{job="podcast-pipeline"} 15:00–15:35Z (healthy)  ->  66 log lines
```

**Liveness is not enough, and this is the subtle part.** During the wedge the loop was
*iterating* the entire time. A heartbeat emitted by that loop would have reported "alive"
cheerfully for four hours. Two distinct signals are required:

- **liveness** — the process/loop is running (catches death and hard blocks)
- **progress watermark** — a monotonic count of completed units, plus `last_progress_at`
  (catches *alive but stuck*, which is what actually happened)

Note also that `podcast_pipeline_run_cost_usd_total` increments only at **terminal** state,
so `increase(...[6h]) == 0` is the expected reading for any in-flight run. It looked like
corroborating evidence during the incident and was not. There is currently **zero**
first-party in-flight signal.

---

## 4. Design

### 4.1 Shutdown coordination

Orchestration owns a `stop_event`. A `try/finally` around the worker-thread lifecycle sets
both `stop_event` and `transcription_complete_event` on **any** exit path, then joins with a
bound. Workers check `stop_event` in their continue-predicate.

The predicate must evaluate supervision bounds **first and unconditionally** — not as another
branch reachable only through queue state, since the incident was precisely the case where
queue-state branches could never fire.

### 4.2 Progress watermark

The **child** emits (it owns the truth), the **api sweeper** evaluates (it survives the
child), **systemd + external alerting** cover the api itself. Each layer watched by a dumber,
more reliable one.

Un-gate `.pipeline_status.json` (drop the `cfg.monitor` condition) and extend it with
`episodes_done`, `episodes_total`, `last_progress_at`, and `in_flight` (idx + stage +
started_at). Day one can degrade to `run.jsonl` `episode_finished` events plus file mtime with
no child changes at all.

### 4.3 Absence alerting

The api exports `podcast_job_running`, `podcast_job_episodes_done`, and
`podcast_job_last_progress_timestamp_seconds`. Then:

```
- alert: PipelineJobStalled
  expr: podcast_job_running == 1
        and (time() - podcast_job_last_progress_timestamp_seconds) > 1800
  for: 5m
```

Do **not** key the threshold to episode-completion times — that is what forces per-feed
tuning against a 4–90 minute spread. Once transport timeouts exist, every healthy in-flight
episode emits sub-episode watermark movement at minutes-scale regardless of episode length,
so a flat 30-minute progress-age threshold is safe. **Transport timeouts are a prerequisite
for cheap alerting, not parallel work.**

### 4.4 Notification

`emit_job_state_change` already fires a webhook on terminal transitions
(`PODCAST_JOB_WEBHOOK_URL`). Route `stale`/`failed` through it to ntfy or Telegram for a phone
push; add vmalert → Alertmanager as an api-independent second channel. **No new dashboards** —
the failure mode being fixed is "a human had to go looking."

### 4.5 Who sets `stale`

The api sweeper — a 60 s asyncio task extending `reconcile` with progress-age logic — sets
`stale` and SIGTERMs per policy. The child never self-marks; it is the thing that dies. When
the api is itself the dead thing, systemd `Restart=always` plus startup-reconcile covers
resurrection and an `absent()`/scrape-staleness rule covers detection.

### 4.6 Budget exhaustion is not an error

`CostCapExceeded` must become a graceful stop: set `stop_event`, drain in-flight work,
finalize, and report a distinct terminal reason (`budget_exhausted`) — not an exception racing
three threads. This is the direct fix for the trigger shared by both incidents.

---

## 5. Categories that must not be forgotten

- **Supervisor death / orphans.** An api restart leaves rows `running` forever; nothing at
  startup reconciles or drains. The child's stdout pipe loses its pump, and once the ~64 KB
  buffer fills the child **blocks on write** — a fresh wedge. Worse, when a live orphan later
  succeeds, the next reconcile sees a dead pid and records `failed / orphan_reconciled_dead_pid`
  — success recorded as failure. Needs startup reconcile + drain, and kill-orphan-group-on-boot
  (`start_new_session=True` already gives a process group) followed by auto-resubmit.
- **Poison pill.** Auto-resubmit plus a deterministically-fatal episode is an infinite loop:
  completed episodes skip, the poison one always re-enters. Needs a per-GUID failure memo
  (attempts, last error) persisted in the corpus and a `quarantined` mark after N.
- **Partial-artifact atomicity.** Individual writes are atomic but an episode is several
  artifacts (transcript, metadata, summary, GI, KG) with no commit marker. Behaviour of
  `skip_existing` against a half-written episode depends on which artifact its predicate
  checks — **unverified**, and whoever implements must verify it. `corpus_completeness.py`
  exists and should be promoted to a post-run gate and crash-test oracle.
- **Error taxonomy.** `error_reason` is free text parsed from logs. Resubmit decisions need
  machine-readable classes: `retryable`, `terminal`, `quarantine`, `budget`.
- **Backpressure.** Real but lowest priority — `_submit_new_jobs` dumps everything available
  into a bounded pool, harmless at current scale.

---

## 6. Testing

Both incidents reproduce deterministically. No chaos tooling is warranted at this scale.

**Build:**

1. **Loop-contract unit tests** — drive the loop with a `process_job_func` that blocks on an
   Event; assert exit within budget and the unit marked failed. Test the predicate truth table
   directly, including `transcription_complete_event=None` and never-set.
   *Blocked on extracting `_run_parallel_processing_loop` to module scope — it is currently
   nested inside `process_processing_jobs_concurrent` and cannot be imported.*
2. **Interpreter-shutdown repro** — spawn a mini-main that starts the worker then raises
   mid-window; assert clean exit and no `cannot schedule new futures` on stderr. Red before
   the fix, green after; pins incident 2 forever.
3. **Scripted-fault provider** at the provider seam (hang / slow / retryable / terminal,
   selected per episode index) — exercises deadline enforcement and quarantine.
4. **Enumerated crash-point kill test** — SIGKILL at each stage boundary on a 3-episode
   fixture, rerun with `skip_existing`, assert `corpus_completeness` passes and nothing is
   duplicated or double-charged. Enumerated beats property-based for CI determinism.
5. **Supervisor lifecycle test** — kill the api mid-run, restart, assert startup reconcile
   kills the orphan group, marks the row, and resubmits within budget.
6. **A no-timeout lint** — CI gate that every outbound HTTP call in pipeline paths carries an
   explicit `timeout=`. One un-timeouted call is the entire hang class.

**Skip:** chaos frameworks, toxiproxy/network-namespace injection, automated alert-rule
integration tests (unit-test exporter values; validate the rule once by hand), load testing at
this scale.

---

## 7. Sequencing

**Phase 0 — evidence.** Done. Root cause confirmed from VictoriaLogs; `CostCapExceeded`
identified as the shared trigger. For the next wedge, `py-spy dump --pid` is the one-command
diagnosis and belongs in the runbook now.

**Phase 1 — cheap, high leverage.** Bound the amplifier and gain eyes.

1. Loop bounds + guarded submits + explicit executor lifecycle — **landed, unverified**
2. Orchestration `try/finally` + `stop_event`; graceful `budget_exhausted`
3. Transport timeouts on every outbound call + the no-timeout lint; retire or fix
   `timeout_context` — **documented honestly, landed**
4. Progress watermark; surface on `GET /api/jobs/{id}`; export the three gauges
5. API sweeper (60 s): reconcile + progress-age → `stale` → SIGTERM → webhook; one vmalert rule
6. Startup reconcile + drain; kill-orphan-group-on-boot; `attempts`; auto-resubmit ≤ 2

**—— cut line ——**

**Phase 2 — structural.** Poison-pill memo + `quarantined`; episode commit marker +
`corpus_completeness` as gate; crash-point tests + scripted-fault provider; job-level error
taxonomy; bounded submission; extract the loop to module scope; index content fingerprint
(C2).

**The cut is defensible:** everything above the line either removes the amplifier both
incidents shared or makes the next unknown failure visible within 30 minutes instead of four
hours — using primitives the codebase already owns.

---

## 8. Open questions

1. **Is `stale` a terminal state or a transition?** If the sweeper SIGTERMs and the child then
   exits cleanly, does the row become `stale` or `failed`? Affects the resubmit decision.
2. **What is the per-job wall-clock budget?** The current 24 h stale threshold exceeds any
   plausible run. A budget derived from `max_episodes` is tempting but episode length varies
   ~20×.
3. **How many auto-resubmits?** Two is proposed. Interacts with poison-pill quarantine; the
   wrong pair burns budget silently.
4. **Does `skip_existing` treat a half-written episode as present?** Unverified and
   load-bearing for crash-only recovery.
5. **Should the cost cap remain `abort`?** `abort` is currently the *broken* path; `warn`
   would make the bug unreachable at the cost of losing the stop. Resolved for now by raising
   the cap to `$10` on the box and keeping `abort` — see ADR-150.
