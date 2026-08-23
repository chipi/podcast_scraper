# INCIDENT-2026-08-18 — a reprocess outlived the job that launched it and spent unattended for 14 hours

| Field | Value |
| --- | --- |
| Date | 2026-08-18 → 2026-08-19 |
| Duration | 2026-08-18 22:41z → 2026-08-19 12:18z (~13h37m of unattended paid work); active response ~2h |
| Severity | SEV-2 (no user-facing outage; direct financial loss, corpus quality regression on 19% of episodes) |
| Affected services | prod-podcast VPS pipeline; Deepgram ASR account; corpus enrichment coverage. Public API/viewer/player unaffected throughout. |
| Author(s) | operator, agent (Claude Code), prod agent (SSH) |
| Status | draft |
| Last updated | 2026-08-19 |

## Summary

A 32-episode corpus repair was dispatched via GitHub Actions; the Actions job was killed at its
360-minute timeout, but the `docker compose run` it had started on the prod box kept running for
another ~8 hours, re-transcribing episodes with `--no-transcript-cache` and billing Deepgram for
every one. It was found and stopped only after the operator noticed their prepaid balance falling
a second day running. A second, unrelated container was discovered alongside it that had been
"Up 7 days" since raising `CostCapExceeded` on 2026-08-12 and failing to exit.

## Impact

- **Customer-facing**: no. The public API, viewer, and player served normally throughout; only
  batch pipeline containers were involved.
- **Financial**: ~$40 of unbudgeted Deepgram spend on 2026-08-19, on top of ~$48 on 2026-08-18
  (the preceding incident, same root code). A `cost_soft_cap_usd_per_run: 5.0` with
  `cost_soft_cap_action: abort` was configured and active the entire time.
- **Data lost or corrupted**: no data lost. But a quality REGRESSION: 127 episodes (19% of the
  678-episode corpus) were rewritten across the two days and **all 127 lost their enrichment** —
  `enrichments_available` went from `{insight_density: true, insight_sentiment: true}` to
  `{false, false}`. 8 of them also lost their summary. GI and KG remained complete (127/127).
- **Repair delivered: ZERO of 32, confirmed.** Forensics on the corpus volume: the work-list held
  32 `substack:post` ids; the run processed 127 episodes across 8 mainstream feeds (megaphone x3,
  simplecast x2, npr, acast, flightcast); **the intersection is empty**. Every dollar bought
  episodes nobody asked for, and there is nothing to salvage — the 32 targets must be re-run.
- **The work-list was ignored, and the targets were simply NOT REACHED YET.** The running image
  (`sha-cd22625`) predates `8143a121`, so `--reprocess-episode-ids` did not restrict the set; the
  run ground through the corpus feed by feed instead. It completed **8 of the 14 feeds** before it
  was stopped, and both substack-hosted feeds — which hold the 32 targets — were among the 6 it
  never got to.

  **CORRECTION (2026-08-19, after this PIR was first written).** An earlier revision of this
  section claimed the substack feed was "not in the spec" and that the targets were therefore
  unreachable by any work-list fix. That was WRONG, and it was an inference from the prod agent's
  zero-overlap measurement rather than a check. One query against the corpus API disproves it:

  | feed | episodes |
  | --- | --- |
  | Lenny's Podcast — `api.substack.com/feed/podcast/10845.rss` | 51 |
  | The Pragmatic Engineer — `api.substack.com/feed/podcast/458709.rss` | 49 |

  140 episodes in the corpus carry `substack:`-prefixed episode_ids. The 32 targets are ordinary
  episodes in ordinary feeds. **The repair needs no configuration change** — only the fixed image
  and a run. The error is recorded rather than quietly edited out because it is the same failure
  shape as the incident's own leading hypothesis: reasoning from an absence in a partial view
  instead of looking at the whole.
- **Time to detect (TTD)**: ~13.5 hours from the orphan starting (22:41z) to the operator raising
  it (~12:05z). No automated signal fired at any point.
- **Time to resolve (TTR)**: ~13 minutes from the operator escalating to both containers stopped
  (12:05z → 12:18z), once an agent with SSH was involved.
- **Time on incident response**: ~2 agent-hours + ~30 operator-minutes across two agents.

---

## Phase 1: Facts (timeline)

| Time (UTC) | Event | Source |
| --- | --- | --- |
| 2026-08-12 08:16:40 | Container `aa5fd46e` created (`--max-episodes 36 --episode-offset 25`, investlikethebest) | `docker ps -a` |
| 2026-08-12 09:28 | That process raises `CostCapExceeded: cost soft cap exceeded: $12.4599 > $5.0000`; container does not exit | container logs |
| 2026-08-18 22:36 | 32-episode reprocess dispatched via `reprocess-prod.yml` | Actions run |
| 2026-08-18 22:41:29 | Container `c8ff8104` created on the box by that job: `--no-transcript-cache --reprocess-episode-ids /app/output/preprocessing_repair_worklist.txt`, image `sha-cd22625` | `docker inspect` |
| 2026-08-19 ~04:36 | Actions job killed at the inherited 360-minute default timeout; ssh client dies, container continues | Actions run |
| 2026-08-19 05:09 | `preprocessing_repair_worklist.txt` mtime (36 lines) | prod filesystem |
| 2026-08-19 07:00 | Last GitHub Actions run of any kind completes | `gh run list` |
| 2026-08-19 10:33:35 | Newest corpus run directory written | corpus API |
| 2026-08-19 ~10:40 | ASR/expensive work finishes; the container spends its remaining time writing podcast artwork (`corpus-art/*.jpg`, cheap) | corpus mtimes |
| 2026-08-19 ~12:05 | Operator reports balance falling again, states work is ongoing | operator |
| 2026-08-19 12:06 | Agent confirms nothing running locally and no Actions run after 07:00z | `ps`, `gh run list` |
| 2026-08-19 ~12:10 | Agent cannot reach prod: ssh host-key verification fails; `/api/jobs` requires an admin key not held; Deepgram key lacks `usage:read` | tool output |
| 2026-08-19 ~12:15 | Handover brief issued to an agent with SSH | `docs/wip/2026-08-19-prod-spend-diagnosis-brief.md` |
| 2026-08-19 ~12:18 | Both containers found; `.viewer/jobs.paused` set; both stopped with SIGTERM + 20s grace; zero respawned after a 45s wait | prod agent |
| 2026-08-19 ~12:18 | Both containers self-remove — they were `docker compose run --rm`, so `docker stop` deleted them and their stdout with them | prod agent |
| 2026-08-19 ~12:25 | Wedge mechanism identified in `_process_episodes_with_threading`; fixed on the hotfix branch (`11b11d81`) | commit |

---

## Phase 2: Analysis

### Root cause

**Killing a GitHub Actions job kills the ssh client, not the work.** `reprocess-prod.yml` ran
`ssh … docker compose run --rm pipeline-llm python -m podcast_scraper.cli …`. When the Actions job
hit its timeout, the runner terminated; the ssh session dropped; the container on the prod box —
which has no restart policy and is a genuine one-shot — simply carried on with no parent, no
observer, and no bound. `--no-transcript-cache` meant every episode it reached was re-sent to
Deepgram at full price.

Three independent properties had to hold for this to run 8 hours unnoticed, and all three did:

1. **No wall-clock bound anywhere.** The workflow declared no `timeout-minutes` and inherited
   GitHub's 360-minute default — a bound nobody chose. Nothing bounded the remote side at all.
2. **No cost bound that could fire.** `stages/transcription.py` contained no cost check of any
   kind; the first enforcement point that observes ASR spend is `check_cost_soft_cap_at_stage`
   after `transcription_thread.join()`, i.e. after every episode in a feed is already paid for.
   (Root-caused in the preceding incident, same day.)
3. **No liveness signal.** Nothing on the box, in the corpus, or in any dashboard indicated that
   a run was in progress. The only evidence was the provider's own billing.

### Contributing factors

- **The work-list did not restrict.** The deployed image `sha-cd22625` predates `8143a121`, so
  `--reprocess-episode-ids` only forced its members past `skip_existing` rather than limiting the
  set. The run was therefore free to work through the corpus at large.
- **A second orphan had been present for a week and nobody knew.** `aa5fd46e` raised
  `CostCapExceeded` on 2026-08-12 and could not exit — see the wedge below. Its presence for 7
  days without detection is the same blindness that let this one run for 14 hours.
- **THE WEDGE (a distinct defect, found during this response).** In
  `workflow/orchestration.py::_process_episodes_with_threading`, Step 9 calls
  `check_cost_soft_cap_at_stage` (which raises) and Step 9.5 sets
  `transcription_complete_event`. The raise unwound **past** the set. The ProcessingProcessor
  thread's continue-predicate waits on that event, so it never terminated, the process never
  exited, and the container never died. `DEFAULT_PROCESSING_LOOP_BUDGET_SECONDS` (4h, from the
  #1180 supervision work) bounds the spin but does not remove it; `aa5fd46e` predates that bound.
  **This means a cost abort that WORKED produced a zombie.**
- **Every prod workflow could start work; none could stop it.** There was no mechanism, anywhere,
  to halt a running pipeline without SSH.
- **Diagnosis was blocked at three doors simultaneously**: no ssh (host-key verification), no
  admin key for `/api/jobs`, no `usage:read` scope on the available Deepgram key. The responding
  agent could not answer basic questions about the state of its own production system.

### Why detection took as long as it did

There was no signal to miss — that is the honest answer. The corpus API cannot be used for
liveness (it serves "newest run per episode", so an in-flight run is invisible until episodes
complete; a two-sample check during this incident wrongly suggested nothing was running). Nothing
writes a heartbeat. The pipeline emits cost telemetry, but a cap that cannot fire generates no
alert. Detection came from the provider's billing email — an out-of-band, human, second-day
signal.

### Why recovery took as long as it did

Recovery itself was fast (~13 minutes). The delay was upstream: the responding agent had no
production access and had to write a handover brief for a second agent. Time was also lost to a
wrong hypothesis — the queue sweeper was proposed first, on the strength of having recently read
that code, and the prod diagnostic (zero container-create events in 8h) killed it immediately.
The competing hypothesis in the same brief was correct.

### What the hotfix would have changed, measured

Re-running the incident's exact topology against the hotfix branch (8 feeds, 32 work-list ids
present in none of them, `tests/integration/workflow/test_selection_gate_end_to_end.py::
test_INCIDENT_3_worklist_ids_absent_from_EVERY_configured_feed`): **0 episodes selected, $0.00
spent, and the run logs `repaired 0/32 · 32 NOT FOUND in any feed's corpus`.** The same test run
against the deployed `sha-cd22625` behaviour fails — episodes are selected and money is spent.

So the hotfix converts this incident from "$40 and silence" into "$0 and a correct error". It does
NOT make the repair succeed: the 32 remain unreachable until their feed is in the feeds spec.

### Counterfactuals (what didn't break that could have)

- **The cost cap never fired on container A.** If it had, the wedge would have converted it into a
  zombie exactly like `aa5fd46e` — quiet, unbounded, and still holding the corpus.
- **`pipeline-llm` has no restart policy.** With `restart: unless-stopped` (as `api` and `viewer`
  have), `docker stop` would have resurrected it and the response would have failed.
- **Only two containers accumulated.** Nothing prevents more; two runs were writing the same
  corpus volume concurrently for seven days with no lock and no complaint.
- **The operator was watching their balance.** Absent that habit, this would have run until the
  account emptied — which is what happened on 2026-08-18.
- **The expensive phase had already finished by ~10:40z.** The stop at 12:18z prevented a
  re-trigger rather than large in-flight ASR; had it been caught 90 minutes earlier the saving
  would have been substantial, and 6 hours earlier, near-total.
- **Container stdout was lost entirely.** `docker compose run --rm` means `docker stop` removed
  both containers and their logs; the VPS Alloy ships only systemd-journal and docker daemon
  events, not container stdout; and the box-local log file is part of the *undeployed* hotfix. The
  surviving evidence is the corpus itself. This is a direct argument for that log file.

---

## Phase 3: Improvement plan

### Prevention (would have stopped this happening)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| Bound the remote command with `timeout`, and refuse to start if `timeout(1)` is absent | `a309de17` | agent | hotfix |
| Explicit `timeout-minutes` on the job instead of an inherited default | `a309de17` | agent | hotfix |
| Pre-authorise every ASR call against the run budget (refuse before spending) | `e0556084` | agent | hotfix |
| Pre-flight gate: price a selection and refuse to start when it exceeds the budget | `a87ad130` | agent | hotfix |
| Run budget spans the whole invocation, not one feed, and counts at the provider choke point | `ea076154` | agent | hotfix |
| Work-list RESTRICTS the episode set | `8143a121` (on main; NOT in the deployed image) | agent | deploy |
| A cost abort must not wedge the process (try/finally around Step 9) | `11b11d81` | agent | hotfix |
| Nothing prevents two pipeline runs against one corpus | #59 (task) | — | follow-up |

### Detection (would have surfaced the problem sooner)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| A long-running batch writes no liveness signal — nothing to alert on | (new) | — | follow-up |
| Corpus read endpoints (integrity, preprocessing) so a repair can be verified without SSH | #1688 | — | ops-api |
| Job status API — know what is running without SSH | #1691 | — | ops-api |
| Report `repaired N/M` with unmatched ids, so a zero-repair run says so | (hotfix branch) | agent | hotfix |
| Reprocessing silently drops enrichment; nothing reports it | #60 (task) | — | follow-up |

### Mitigation (would have reduced impact / recovery time)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| Emergency brake workflow: list, stop, verify — not approval-gated, not in the prod-corpus concurrency group | `9f954ee8` (needs to land on `main` to be dispatchable) | operator | now |
| Batch halts on a tripped cap instead of continuing to the next feed | `a946215b` | agent | hotfix |
| Logs survive a kill (`PYTHONUNBUFFERED`, box-local tee) | `a309de17` | agent | hotfix |
| An operator-facing STOP endpoint, aligned with the ops-api epic | #1687 | — | ops-api |

### Process (would have changed how we respond)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| Responding agent had no prod access and no credentials for its own control plane | #1687 | — | ops-api |
| Runbook: how to determine whether prod is doing paid work, and how to stop it | (new) | — | follow-up |
| Do not use the corpus API for liveness — document why it cannot answer that question | this PIR | agent | done |

---

## What went well

- **The handover brief worked.** A structured brief with an explicit discriminating command
  produced a correct diagnosis and a complete stop in ~13 minutes by an agent with no prior
  context. The single most useful element was naming *both* hypotheses and the one command that
  separates them.
- **The prod agent stopped things in the right order** — hold the queue, then stop the container,
  then verify past a full sweep interval — and reported the second container unprompted.
- **`docker stop` with grace rather than `kill`** preserved in-flight cost accounting.
- **The 7-day zombie was found by accident and was worth more than the leak.** It exposed a defect
  that would have made the hotfix actively harmful.
- **`pipeline-llm` being restart-less** turned out to be load-bearing, and was a deliberate earlier
  choice.

## What went wrong

- A workflow that starts remote work owned that work only through an ssh session, and the system
  treated the death of the session as unrelated to the death of the work.
- A cost cap that had been configured, reviewed, and believed-in for months could not fire on the
  stage that spends the most money, and produced a hung process on the rare occasion it did.
- The responding agent's first hypothesis was drawn from code it had just been reading rather than
  from the evidence, and it spent two tool calls confirming a fact the operator had already stated.
- The operator had to be the monitoring system, twice, two days running.

## Lessons learned

- **The recurring shape is "work outlives its owner, and nobody notices."** Three incidents in
  eight days: an Actions job dies and the container lives; an exception unwinds and the thread
  lives; a cap trips and the run hangs. Point fixes were applied to each; the class deserves one
  structural answer — most likely a run lease/heartbeat in the corpus that a live run refreshes
  and any observer (API, workflow, agent) can read.
- **A safeguard that has never fired is not a safeguard, it is an assumption.** The $5 cap was
  correct in config, correct in pricing, and unreachable in practice. Nothing tested it end to
  end; every test called the enforcer directly with a hand-built metrics object. The wiring — not
  the mechanism — is what failed, and the tests were shaped to exercise exactly the half that
  worked.
- **Fixing a dormant safeguard is a behaviour change, not a bug fix.** Making the cap work made a
  previously-unreachable unwind path reachable. Anything that makes a rare path common must be
  reviewed as though that path were new.
- **Operator access asymmetry is an incident amplifier.** Every minute of this response spent
  writing a brief for someone else was a minute the meter ran. The ops-api epic (#1687) is not a
  convenience feature; it is incident tooling.

---

## References

- Preceding incident, same root code: 2026-08-18 32-episode reprocess (~$48) — see
  `docs/wip/2026-08-19-session-handover.md` §7
- Hotfix branch: `hotfix/cost-containment-and-scope-gate`
- Commits: `a309de17`, `ea076154`, `a87ad130`, `a946215b`, `e0556084`, `11b11d81`, `9f954ee8`
- Issues: #1757 (safety release), #1687 + #1688 + #1691 (ops-api), #1676
- Handover brief: `docs/wip/2026-08-19-prod-spend-diagnosis-brief.md`
- Prior PIR with a related theme (work continuing unnoticed for days):
  `INCIDENT-2026-08-05-prod-disk-image-pileup.md`
