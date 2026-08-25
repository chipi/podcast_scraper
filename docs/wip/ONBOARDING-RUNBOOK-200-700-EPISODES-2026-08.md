# Onboarding runbook: +200–700 episodes across new shows

Status: DRAFT for operator review (2026-08-25, `feat/onboarding-readiness` arc).
Owner gate: **every prod dispatch below is operator-approved per instance.** Nothing here
self-executes.

The repair mission improvised its operational discipline mid-incident. Onboarding is the same
class of operation — money, disk, corpus purity — planned this time.

## 0. Hard preconditions (in order)

| # | Precondition | State (2026-08-25) |
| --- | --- | --- |
| 1 | The `feat/onboarding-readiness` PR is merged and images built | in progress (this arc) |
| 2 | **Deploy** those images (`deploy-all-prod.yml`, gated) | NOT DONE — prod runs `b9fb37d`, which lacks the dedup gate, the #1720 failover fix, and the STOP endpoint |
| 3 | Audio decay-clock backfill (`backfill-audio-prod.yml`): dry-run reviewed → real run | dry-run dispatched 2026-08-25, results pending |
| 4 | Pre-onboarding snapshot: `backup-corpus-prod.yml` (or `snapshot-prod.yml`) completed green, run id noted in this file | NOT DONE |
| 5 | Health drill green (`make health-drill` against prod o11y) | drill exists; run it the same day |
| 6 | Decide ML enrichers: they are **disabled** in prod `viewer_operator.yaml` since the 2026-08-24 incident (#1817 gold-set re-eval owns re-enable). Onboarding runs WITHOUT them; enrichment for new episodes is re-runnable later | decision on record: stay disabled |

Rollback path: `prod-restore-corpus.yml` from the step-4 backup. The drill variant
(`drill-restore-corpus.yml`) has proven the restore path; if the last green drill is older
than a month, re-run it BEFORE onboarding, not after a bad batch.

## 1. Money

Measured baseline (2026-08-19, live 678-episode corpus — the numbers behind
`cost_soft_cap_usd_per_run: 25.0` in `cloud_balanced.yaml`):

- mean episode 51 min, p90 91 min
- **$0.238/episode all-in** ($0.218 ASR + $0.020 LLM)

| Scope | Mean estimate | With 1.25× retry allowance |
| --- | --- | --- |
| 200 episodes | ~$48 | ~$60 |
| 700 episodes | ~$167 | ~$210 |

Controls, all live in the arc code:

- **Selection gate** prices every dispatch from feed durations before any download and
  refuses over-cap selections ("N of M · H audio-hours · est. $X" — read this line on every
  batch).
- **Run budget ledger** counts at the provider choke point (incl. ASR) and aborts mid-run.
- **Batch = a cost, not a count.** Target **~$10–25 per dispatch** (roughly 40–100 mean
  episodes). `write_work_list(chunk_budget_usd=...)` mechanizes the split when driving by
  work-list; per-feed `max_episodes` bounds it when driving by feed.
- **Batch one is small**: one show, ~10–20 episodes, then a full verification pass (§4)
  before scaling up. This is #1757's stated limit — the cap firing against real providers is
  only proven by a real run.

## 2. Disk

- Local at-rest audio ≈ 0 by construction: per-run offload-to-cold + eviction is deployed and
  live-verified ("evicted 13 (1.16 GB); kept 0 not-yet-in-cold", zero errors). Peak local
  usage = one feed-run's audio.
- Cold archive: Hetzner Storage Box (bx11, 1 TB). 700 episodes ≈ 70–100 GB — fits with
  wide margin next to the existing corpus audio.
- Check `df -h /` on the box before batch one; the box was at 49% on 2026-08-23.

## 3. Corpus purity

- **Content-dedup gate (#1656)** is active from the first post-deploy run: identical bytes
  under a new GUID are refused before ASR. The index starts EMPTY — it protects against
  duplicates *within and after* onboarding, not against the pre-existing corpus (its audio is
  in cold; hashing it retroactively is a separate decision).
- Back-catalogs are where republishes live. Expect and read the
  "content-duplicate audio (#1656)" warnings per batch; each one is money not spent.
- Entity duplication and ranking drift at the new scale are measured, not guessed:
  #1683 (Wave-0 baseline BEFORE onboarding) → #1684/#1685 after.

## 4. Batch mechanics + per-batch verification

Mechanism: per-feed, bounded submissions — operator API `POST /api/jobs?feed=<slug>&
max_episodes=N` (server-side selection gate + run budget + dedup gate all apply), or the
scheduled sweep once a feed is trusted. One feed at a time for new shows.

Per-batch checklist (10 minutes, same queries every time):

1. Selection line: `selected N of M`, estimate sane, no refusal.
2. Completion: `repaired/processed N/N` against the denominator; job status `completed`.
3. Spend: LiteLLM SpendLogs delta ≈ estimate (the 2026-08-24 chain proved SpendLogs ==
   OpenRouter to the cent); Deepgram console delta for ASR.
4. Eviction: VictoriaLogs `instance:prod-podcast "audio eviction"` shows the run's summary
   with `kept 0 not-yet-in-cold`, `0 unlink-failed`.
5. Dedup: count of `#1656` warnings (money saved, and a signal about the feed's hygiene).
6. Errors: GlitchTip untriaged events for the window; `enrich-common-run-failed` alert quiet.
7. Disk: `df -h /` unchanged-ish.

Cadence: batches are independent; run 1–3 per day with the checklist between, not a single
700-episode weekend. The per-run cap ($25) and the selection gate make a runaway batch
refuse-at-start rather than abort-mid-flight.

## 5. Abort paths (verified to exist, in preference order)

1. `POST /api/jobs/stop` (#1785, after deploy) — pauses the queue FIRST, SIGTERMs running
   work, verifies, names survivors. `POST /api/jobs/resume` releases.
2. `stop-prod-pipeline.yml` — works against today's prod, no deploy needed; deliberately not
   approval-gated (a brake that queues behind the runaway is not a brake).
3. SSH + `docker rm -f` — last resort, the 2026-08-25 wedge path.

## NOT covered by this runbook (explicit)

- Re-enabling `topic_consensus`/`topic_similarity` (#1817 gold-set re-eval owns it).
- Retroactive fingerprinting of the pre-existing corpus.
- Feed *selection* (which shows to onboard) — editorial, operator's call.
- Player/consumer-side capacity (search index size, viewer performance at 1,400 episodes) —
  #1683's measurements inform this, but no work is scheduled here.
