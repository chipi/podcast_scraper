# Incremental rollout — consolidated follow-ups (F1–F15)

**Status:** living document — updated as feeds and episodes are onboarded.
**Reconstructed locally 2026-08-12** on the homelab from the operator's handover digest.
The original lives in the prod repo (`ai-ml-improvements` worktree); if the two diverge,
the prod copy is the source of record for F1–F15 and this copy is the source of record
for the rollout log in the final section.

Legend: `[OPEN]` needs work · `[DONE]` resolved · `[INFO]` by-design / no action.

---

## A. Provenance / metadata labels

### A1 (= F2 + F12) — Deepgram output mislabeled as Whisper — `[OPEN, medium]`

Episode metadata records `transcript_source="whisper_transcription"` and
`whisper_model="nova-3"` even when **Deepgram** did the transcription.

Metadata-only; no viewer or API consumer reads these fields. Harms:

1. Misleads provenance audits — already caused one false "wrong profile" scare.
2. Latent: only bites in a mixed Whisper + Deepgram corpus, where you can't target a
   single engine for reprocessing.
3. Analytics misattribution.

Does **not** affect UI, extraction quality, routing, or cost.

**Fix:** provider-neutral field names, or populate from `config_snapshot.ml_providers`
(which already holds `"deepgram"`). Note `transcript_source` is a `Literal` enum, so a
true rename touches both the schema and every reader.

### A2 (= F13) — `key_quotes` / `named_entities` always empty — `[OPEN, low]`

`summary.key_quotes` and `summary.named_entities` are always `0`. The real quotes and
entities live in the search index (`doc_type` `quote` / `kg_entity`) and the KG artifact,
not the summary block.

**Decide:** remove the dead sub-fields, or populate them.

---

## B. Discovery / add semantics

### B1 (= F4 + F15) — offset counts RSS position, not corpus absence — `[OPEN docs / INFO]`

`episode_offset` counts position in the RSS feed, which is **not** the same as "not in the
corpus". Separately, the discovery script's corpus-guid set (read from
`search/metadata.json`) lags the live corpus. So a "contiguous-N-new" window sometimes
includes one already-present episode and the feed adds N-1.

Observed: WSJ / NVIDIA +4 of 5, Flightcast fewer.

`skip_existing` (corpus-wide, D7) is the exact guard — no duplicate, no error. **Benign.**

**Optional fix:** re-read corpus guids immediately before each feed, or widen the window
and cap on adds. Either way, document the offset semantics.

### B2 — `[WITHDRAWN 2026-08-12]` "Latent Space feed serves only 2 items" — measurement error

**Raised and disproved within the same session. Recorded so nobody re-raises it.**

The claim was that the Flightcast/Latent Space feed served only 2 `<item>` elements and was
therefore permanently capped at 13 episodes. **This was false.** The feed serves **219
items** and is healthy.

**Cause of the error:** discovery counted items with `grep -c "<item>"`, which counts matching
**lines**, not occurrences. The Latent Space feed is 13.2 MB of XML minified onto very few
lines, so it reported 2. The other eight feeds serve multi-line XML where line count happens
to equal occurrence count, so they were unaffected — which made a tooling bug look like a
feed-specific defect. Correct form: `grep -o "<item>" | wc -l`.

**Lesson worth keeping:** when one item in a survey looks catastrophically different from the
others, suspect the measurement before the subject. The "dead feed" story was coherent,
matched an existing known issue (B1's Flightcast note), and was entirely wrong.

**Latent Space's `+0` in the +50 batch** therefore reverts to the original explanation:
`episode_offset=0` targeted the newest 10 episodes, which were already among its 13, and
`skip_existing` skipped them. It can reach any reasonable target using a real offset.

---

## C. Outcome / index signals (cosmetic)

### C1 (= F1) — skips tally as `failed`, not `skipped` — `[OPEN, low but affects EXIT gates]`

A clean $0 all-skip run reports `{failed: 1}`, which fails the Step-0/Step-1 EXIT criteria.
No functional harm. The skip-existing path (including the D7 corpus-layout branch) never
sets `status="skipped"` — only the exception path does
(`episode_processor.py:1655` / `:2644`).

**Fix:** set `status="skipped"` on both skip branches, and add a test.

> **Operational note (2026-08-12):** this mislabel does **not** propagate to job-level
> status. The WSJ re-run below was a complete all-skip and still returned
> `status=succeeded`. Watchers keyed on job status are safe; only per-episode outcome
> tallies are affected.

### C2 (= F3) — `reindex_recommended: true` after an all-skip reindex — `[OPEN, low]`

Run-summary mtime advances even though the index didn't change.

**Fix:** base staleness on the episode / fingerprint **set**, not mtime.

### C3 (= F6) — first post-D8 reindex re-embeds the whole corpus — `[INFO, by-design]`

The fingerprints file didn't exist yet, so the first reindex after the D8 deploy re-embeds
everything once; every reindex after that skips unchanged episodes. Not a bug — note it in
ops so a post-deploy ~10 min reindex isn't misread as a regression.

### C4 — `/api/corpus/feeds` lags behind the catalog — `[OPEN, low]`

> **Revised 2026-08-12, same session.** First written as "under-reports episode counts",
> implying permanent misattribution. **That was too strong.** The counts *do* catch up — after
> the next run completed, the endpoint showed Invest at 57 and The Journal at 34, summing
> exactly to the catalog's 270. It is a **staleness/refresh** problem, not a data problem.
> Severity dropped medium → low. The operational warning below still stands, because the lag
> window is long enough to mislead offset planning.

**Found 2026-08-12.** After a run added 32 episodes to Invest Like the Best,
`/api/corpus/feeds` still reported that feed at its pre-run count of **25**, while
`/api/corpus/stats` reported `catalog_episode_count: 262` and `/api/corpus/coverage` confirmed
`total_episodes: 262, with_gi: 262, with_kg: 262`.

So 32 fully-processed episodes were **counted by the catalog and by coverage, but attributed
to no feed**. Per-feed counts summed to 230 against a catalog of 262.

This matters more than cosmetics suggest: **per-feed counts are what any
equal-representation plan computes offsets from.** A feed that under-reports will be given an
offset that re-reads episodes already ingested — `skip_existing` prevents duplicates, but the
window is wasted and the target is never reached. Anyone automating "top up each feed to N"
against this endpoint will silently stall.

**Suspected cause (not investigated):** the endpoint aggregates from an episode-metadata scan
that may not associate episodes written under the `--single-feed-uses-corpus-layout` / D7
branch back to their feed. Worth checking against the same code path C1 implicates.

**Workaround in the meantime:** trust `catalog_episode_count` + `coverage` for totals, and
treat per-feed counts as a lower bound.

---

## G. Job execution and supervision

### G1 — long jobs wedge silently; `status` stays `running` forever — `[OPEN, high]`

**Found 2026-08-12.** A `max_episodes=36` job ran for **6 h 20 m**, of which the last
**4 h 15 m** did nothing at all, while continuing to report:

```
"status": "running"   "error_reason": null   "exit_code": null   "pid": 14043
```

Every internal signal said healthy. The evidence it was dead came only from **outside** the
job record:

| Signal | Value |
| --- | --- |
| `increase(podcast_pipeline_run_cost_usd_total[6h])` | **0** (was $28.11 for the same window earlier) |
| `node_cpu` on `prod-podcast` | **2.5 %** — box idle |
| `catalog_episode_count` | frozen at 262 for 4 h 15 m |

**`POST /api/jobs/reconcile` did not help** — it returned `updated: 0`, because the pid still
existed. The process had not crashed; it was **wedged**, most likely blocked on a network call
with no timeout (feed fetch, transcription, or LLM). Reconcile only detects *dead* processes,
not *hung* ones.

`POST /api/jobs/{id}/cancel` worked cleanly (`exit_code: 130`), and the 32 episodes completed
before the wedge were intact and fully enriched — no data loss, no partial artifacts.

**Why this is high severity:** an unattended batch loses its whole remaining window to a hang
and reports success-shaped state throughout. A scheduled or overnight run would burn hours
silently, and the operator has no signal that anything is wrong.

**Suggested fixes, cheapest first:**

1. **A watchdog on progress, not liveness.** If a running job's episode count has not advanced
   in N minutes, mark it `stale`. The `stale` status already exists in the status enum and is
   currently unreachable for this failure mode.
2. **Timeouts on outbound calls** in the pipeline — the wedge is almost certainly an untimed
   socket read.
3. **Expose per-job progress** (episodes done / total) on `GET /api/jobs/{id}`. Today a job's
   only observable is a status string that cannot distinguish "working" from "hung".

**Operational mitigation now in use:** smaller windows (`max_episodes=15` rather than 36) so a
hang costs minutes rather than hours, plus a client-side stall detector that alarms when
`catalog_episode_count` is static for ~30 min while status is `running`.

### G2 — `RuntimeError: cannot schedule new futures after interpreter shutdown` — `[OPEN, high]`

**Found 2026-08-12.** First hard job failure of the rollout.

```
job_id:       97180b23-e8e5-4714-8475-2d2341a70803
feed:         Latent Space (Flightcast)
status:       failed        exit_code: 1
started:      15:37:59Z     ended: 16:29:28Z   (~51 min)
args:         --max-episodes 15 --episode-offset 13 --episode-order newest --skip-existing
error_reason: RuntimeError: cannot schedule new futures after interpreter shutdown
```

#### It is NOT a load or capacity problem — measured, not assumed

| Signal | Value | Reading |
| --- | --- | --- |
| `node_memory_MemAvailable` (prod), now | 73.4 % | healthy |
| Same, **min over the 6 h containing the failure** | **66.4 %** | never pressured; no OOM |
| prod CPU during window | ~2.5 % | idle |
| `podcast_pipeline_run_cost_usd_created` | `1786468912.96`, **unchanged** | the api container did **not** restart |

#### It left NO dirty state

`/api/corpus/coverage` after the failure: `total_episodes: 296, with_gi: 296, with_kg: 296,
with_both: 296, with_neither: 0`. The job completed **12 of 15** episodes and every one is
fully enriched. There are no partial artifacts to clean up, and the run is safely re-runnable
because `skip_existing` will pass over the 12.

#### What actually happened

The error is raised by CPython's `concurrent.futures.thread` when `executor.submit()` is
called after the interpreter's shutdown hook (`_python_exit`) has run. So something invoked
the summarizer's parallel chunk path **while the job process was already tearing down** —
this is a lifecycle race inside the job's own process, unrelated to the api or the host.

**The fragility that turns it into a hard failure** is at
`providers/ml/summarizer.py:2795-2799`:

```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_chunk = {
        executor.submit(_summarize_chunk, (i, chunk)): i for i, chunk in enumerate(chunks, 1)
    }
```

`_summarize_chunk` has its own `try/except Exception` and degrades a failed chunk to
`(chunk_idx, None)`. **The submission loop has no such guard.** A `submit()` that raises
propagates straight out of `_summarize_chunks_parallel` and fails the entire run — so the
code is defensive about *doing* the work and undefended about *scheduling* it.

#### Suggested fixes

1. **Guard the submit loop** — wrap submission so a scheduling failure degrades to
   sequential/in-thread summarization (or to `(idx, None)` per chunk) instead of killing the
   run. Cheapest fix, highest value, no behaviour change on the happy path.
2. **Find the shutdown trigger.** Something began interpreter teardown while summarization was
   live. Worth checking whether summarization can be reached from a non-main or daemon thread,
   or from an `atexit`/finalizer path.
3. **Not the cause, but adjacent:** `mcp/telemetry.py:34` holds a *module-level*
   `ThreadPoolExecutor` (`_UMAMI_POOL`). Module-level pools are a classic source of this exact
   error. It is on the MCP surface rather than the CLI path, so it is unlikely to be implicated
   here — but it is the same anti-pattern and worth auditing.

#### Relationship to G1

Both G1 (silent wedge) and G2 are executor-lifecycle failures, and it is tempting to call them
one bug. **The evidence does not support that yet:** G1 showed a live pid at 0 % CPU with no
exception, G2 raised at submit time and exited 1. They may share a root cause in executor
management or may be independent. Worth investigating together; not worth assuming.

#### Operational verdict

**Safe to continue.** The failure is neither load-driven nor state-corrupting, and 12 of 15
episodes landed clean. Expect it to recur occasionally; `skip_existing` makes every retry free.
Smaller windows (15) already bound the loss per occurrence.

### G3 — progress signals are episode-granular, so "no progress" is unreadable — `[OPEN, medium]`

**Found 2026-08-13** when a client-side stall detector fired a **false positive** on a healthy
Hard Fork run.

The detector's rule was *"`catalog_episode_count` flat for 30 min while `status=running`"*. The
rule was satisfied and the inference was wrong: the corpus counter only increments when an
**episode completes**, and Hard Fork's median episode is 79 min (max 113). One long episode
plus transcription and enrichment trivially exceeds a 30-minute gap while everything is fine.

**What actually distinguished this from the G1 wedge was host CPU** — 15.5 % sustained over an
hour (working) versus 2.5 % (wedged). That is a machine-level metric that knows nothing about
episodes, and having to reach for it is the same "correlate three unrelated systems" problem
G1 exposed.

| | G1 (real wedge) | G3 (false positive) |
| --- | --- | --- |
| `status` | `running` | `running` |
| corpus growth | flat 4 h 15 m | flat ~30 min |
| **prod CPU** | **2.5 %** | **15.5 %** |
| outcome | cancelled by hand | succeeded normally |

**The generalisable point:** with progress measured only at episode completion, *any*
threshold is wrong. Too tight and long-form feeds cry wolf; too loose and a real wedge runs for
an hour before anyone notices. The signal is lumpy at exactly the granularity that matters.

**Fixes, in order of correctness:**

1. **Sub-episode progress watermark** (RFC-117 §4.2 / ADR-150 D7). "Episode 9 of 10,
   transcribing, last movement 40 s ago" would never have fired here and would have caught G1
   within minutes. This is the actual answer.
2. **Feed-aware thresholds** — now cheap, since §5h of the onboarding doc measured median and
   max episode duration for every candidate feed. A threshold of ~2× a feed's median is
   defensible where a flat 30 min is not.
3. **Interim mitigation applied 2026-08-13:** threshold raised to ~45 min and prod CPU is now
   reported *inside* the warning, so an alert states whether it is idle-stuck or busy-slow
   rather than leaving an operator to correlate by hand. This is a workaround, not a fix — CPU
   is a proxy for a signal the job should be emitting itself.

---

## G-notes. Operational observations (2026-08-12/13)

Smaller learnings from driving ~320 episodes through the operator API. Not defects; things the
next person will otherwise rediscover.

- **`GET /api/jobs/subprocess-log` and `/api/jobs/{id}/log` exist.** I initially tried to read a
  failed job's log through `/api/corpus/text-file`, which rejects `.log` ("Only .txt, .md,
  .vtt, .srt, and .json are allowed"), and concluded logs were unreachable. They are not —
  there are two purpose-built endpoints. **Use those for job diagnosis.**
- **Handle an empty API response explicitly in any watcher.** A poll where both the status and
  corpus calls failed produced `status=` and `corpus=` empty. Treating empty as "unchanged"
  would have read as healthy — the same silence-as-health failure as G1. An explicit
  `no-response` state surfaced it instead; the next poll recovered, confirming a transient blip.
- **Per-feed window yield varies enormously and is stable per feed.** From 15-episode windows:
  The Journal returned +8, +7, +7, +7 across four consecutive windows, while NVIDIA, Hard Fork
  and No Priors regularly returned +13/+14. This is B1's discovery lag interacting with a
  corpus whose episodes are not a contiguous newest-N block. Practical consequence: **plan
  windows by measured yield per feed, not by deficit** — The Journal needs roughly double the
  windows of NVIDIA for the same gain.
- **The corpus feeds endpoint lags the catalog** (C4). Compute offsets from
  `catalog_episode_count` plus the per-feed table only after a run settles, or offsets will be
  planned against stale counts.

---

## D. LiteLLM prod gateway (Option-A / ADR-142 follow-ups)

### D1 (= F7) — podcast-prod key had no budget cap — `[DONE]`

Set `max_budget=$25` via `/key/update` (verified).

**Residual code TODO — still open:** `deploy-litellm.yml:213` `/key/generate` should mint
the key *with* a budget so a fresh box gets one automatically, rather than relying on a
manual `/key/update` after the fact.

### D2 (= F8) — orphan gateway key — `[OPEN, hygiene]`

`proj-podcast-prod=b404b602` on the prod gateway is unused. Delete via `/key/delete` once
settled.

### D3 (= F9) — ops-card reads prod spend from the wrong gateway — `[OPEN, medium]`

`server/routes/llm_gateway.py` reads prod LLM spend `{box="prod"}` scraped **from
homelab**. Now that prod has its own gateway, point the card at the local gateway's
metrics.

### D4 (= F10) — `litellm_api_base` override is not durable — `[OPEN, medium]`

The override lives **only** in the box's `viewer_operator.yaml`. It survives deploys, but a
DR or volume rebuild loses it and silently reverts to the homelab gateway.

**Fix:** document in the DR runbook, or drive it from a deploy-managed env
(`config.py:2097`).

### D5 (= F11) — provisioning idempotency gap — `[OPEN, medium]`

The prod gateway never had the app key until this session; the app "worked" only by way of
homelab. Add a `deploy-prod` post-check asserting the pipeline's key authenticates against
the **configured** `litellm_api_base`.

### D6 — every budget signal measures spend; none measures headroom — `[OPEN, high]`

**Found 2026-08-13** when pass-3 hard-stopped on exhausted upstream credit.

Every cost signal in the estate is **cumulative outflow**:

| Metric | Measures |
| --- | --- |
| `litellm_key_spend_usd` | spent |
| `litellm_key_max_budget_usd` | a *configured* ceiling, not a live balance |
| `litellm_key_budget_burn_ratio` | spent ÷ configured ceiling |
| `openrouter_vertical_usd` | spent, by vertical |
| `podcast_pipeline_run_cost_usd_total` | spent, modelled |

**Nothing exports remaining credit at the upstream provider.** The consequence is that the
first notice of exhaustion is a **hard job failure in production**. There is no threshold to
alert on, no burn-down to watch, and no way to answer "can this batch finish?" before starting
it.

Note that `litellm_key_budget_burn_ratio` *looks* like a headroom metric and is not — it is
headroom against a **locally configured cap**, which is exactly why it read a comfortable
21.7 % at the moment the pipeline could not make a single call.

**This is the same defect class as G1, in the money domain.** G1 was invisible because every
detection signal was exception-triggered and nothing was absence-triggered. This is invisible
because every cost signal is presence-triggered (what was spent) and nothing measures the
absence (what remains). Both produce the same operator experience: the system is fine, the
system is fine, the system is dead.

**Proposed fix:**

1. Export `upstream_credit_remaining_usd{provider=...}` from whatever collector already talks
   to the provider billing API — the `openrouter-spend.sh` collector is the obvious host, since
   it already authenticates there.
2. Alert on **projected exhaustion**, not on a fixed floor: `remaining / burn_rate < 24h`.
   A fixed floor is wrong when one vertical can consume the balance at an unrelated rate.
3. Surface it on the ops card next to the spend figure D3 already covers. Spend without
   headroom is half a picture.

**Related risk (unverified):** verticals appear to share one prepaid account —
`pi` had consumed $72.75 against `podcast`'s $1.38 at the time of failure. If that is one
balance, any workload can starve any other, and a per-vertical spend cap does not prevent it.
Worth confirming before relying on per-vertical budgets as isolation.

---

## H. Data retention and reprocessability

### H1 — the audio archive is built, documented, and switched off — `[OPEN, high]`

**Found 2026-08-13** while checking prod disk headroom ahead of the next ingestion batch.

`docs/recipes/prod-audio-archive.md` (#1199) is a complete runbook: Hetzner Storage Box over
rclone/SFTP, `archive pull` for laptop recovery, and a reprocess flow that reads audio from the
archive instead of re-fetching feeds. `infra/terraform/storage_box.tf` implements the
provisioning. The feature is finished.

**It has never been enabled.** Verified:

| Where | State |
| --- | --- |
| `config/profiles/cloud_balanced.yaml` | `audio_storage_backend` **not set** |
| box `viewer_operator.yaml` | **not set** |
| `infra/terraform/storage_box.tf:12` | `count = var.audio_storage_box_type != "" ? 1 : 0` — defaults to empty, so **provisions nothing** |

Combined with the recipe's own statement — *"Prod stores no audio by default: episodes are
transcribed, then the media is discarded"* — the consequence is:

> **The source audio for every episode in the corpus is gone.** At the time of writing that is
> ~454 episodes, none of them reprocessable from archive.

**Why that matters more than it sounds.** Reprocessing is a first-class activity here —
ADR-149 governs a whole reprocess methodology, and the corpus has already moved v2.1→v2.5.
Every future model upgrade (better ASR, new diarization, a revised GI/KG schema) currently
requires **re-downloading audio from the live feed**, which the recipe itself says fails when:

- an episode rolls off the publisher's feed window (already observed — The Seen and the Unseen
  serves 117 items against 454 published), or
- a dynamic-ad feed re-encodes the file, so the re-downloaded audio is not the audio that
  produced the existing transcript.

So the corpus is quietly accumulating episodes that can never be re-derived, only re-scraped
if lucky.

**Cost of fixing:** a `bx11` Storage Box is 1 TB at roughly €3.20/month. The work is one
`tofu apply`, one GitHub secret (`PROD_RCLONE_STORAGEBOX_PASS`), and three profile lines. It is
among the cheapest items on this entire list.

**Recommended timing — before the next batch, not after.** The planned expansion adds ~25 shows
at ~10 episodes each (~250 episodes). Enabling the archive first makes those reprocessable;
enabling it after means another 250 one-shot episodes. The already-lost 454 cannot be
recovered either way, which is precisely why the next 250 should not join them.

### H1a — the local audio cache is written into an ephemeral container layer

**Verified 2026-08-13**, resolving the open question above. The answer is more specific than
the recipe's "media is discarded", and it makes the interim fix much cheaper.

Caching is **on** and always has been — it is the `remote` archive that is off:

| Setting | Value | Source |
| --- | --- | --- |
| `audio_cache_enabled` | `True` (default) | `config.py:3901` |
| `audio_cache_in_corpus` | not set → `False` | absent from profile |
| `DEFAULT_AUDIO_CACHE_DIR` | `.cache/audio` — **relative** | `config_constants.py:108` |
| `PODCAST_SCRAPER_WORK_DIR` | `/app` | `compose/docker-compose.stack.yml` |
| mounted volume | `corpus_data:/app/output` — **the only one** | same file |

So the cache resolves to **`/app/.cache/audio`**, which is inside the container filesystem and
**not** under the mounted corpus volume. Prod runs every job as `docker compose run --rm`.

**Each job therefore downloads audio, caches it, and destroys the cache on exit.** The audio is
not "discarded after transcription" so much as *cached into a directory that ceases to exist*.
Nothing is recoverable retroactively; the ~454 episodes are confirmed lost.

**Two fixes, and the cheap one needs no infrastructure at all:**

1. **`audio_cache_in_corpus: true`** — one line. The cache moves to
   `<corpus>/.podcast_scraper/audio-cache`, which IS on the mounted volume, so it survives the
   container. Costs prod disk: ~50 MB/episode against 96.9 GB free, so the planned +250
   episodes fit comfortably (~12 GB). **Zero cost, zero new infrastructure, available now.**
2. **`audio_storage_backend: remote`** (H1) — the durable answer, offloads to a ~€3.20/month
   Storage Box and does not consume prod disk.

They are not mutually exclusive and (1) is a strictly better default than the status quo even
if (2) is never enabled. **The status quo — caching enabled, writing to a path that is deleted
on every run — is the worst of the three options**: it pays the disk-write cost of caching and
gets none of the benefit.

---

## E. Deploy

### E1 (= F5) — `deploy-all-prod.yml` unvalidated — `[OPEN, low]`

The one-trigger orchestrator has never been dispatched. Needs
`secrets.DEPLOY_ORCHESTRATOR_PAT` plus one live run. Until then, deploy via the three
individual workflows.

---

## F. Quality QA

### F1qa (= F14) — no semantic-correctness audit — `[OPEN, optional]`

Quality has been assessed on structure and samples (strong), but there is no
insight-by-insight grounding check against the transcript.

**If a grounding gate is wanted:** spot-check N insights per episode, or use the
insight-node confidence scores.

---

## Status summary

15 items → **13 distinct** (F2+F12 merged, F4+F15 merged). B2 was raised 2026-08-12 and
**withdrawn the same session** as a measurement error — it is not open work.

> ### Implementation status — branch `fix/pipeline-resilience-supervision` (2026-08-12)
>
> Anchored by [#1620](https://github.com/chipi/podcast_scraper/issues/1620); designed in
> [RFC-117](../rfc/RFC-117-pipeline-supervision-and-absence-detection.md) and
> [ADR-150](../adr/ADR-150-supervision-bounds-and-absence-detection.md).
>
> | Item | State |
> | --- | --- |
> | **G1** silent wedge | **Fixed on branch** — loop bounded by main-thread liveness + wall-clock budget, evaluated unconditionally before queue state |
> | **G2** executor race | **Fixed on branch** — both submits guarded; explicit executor lifecycle so abandoning does not `shutdown(wait=True)` on the stuck future |
> | **C1 / F1** skip tallies as failed | **Fixed on branch** — records `status="skipped"`, exception-safe |
> | **A1** Deepgram labelled Whisper | **Partly fixed on branch** — provider-neutral `transcription_provider` / `transcription_model` added to the run manifest; the `transcript_source` enum rename is deliberately deferred (it is a migration touching argparse, config, evaluation and every on-disk metadata file) |
> | `timeout_context` decorative | **Documented honestly on branch** — cannot interrupt; demoted to a detection signal, deadline log raised to ERROR, timer made daemon |
> | **D1** budget cap | **Done** — plus the per-run cap raised 5.0 → 10.0 on the box (see cost analysis) |
> | **D2** orphan gateway key | **Verified closeable** — `$0.00000088` spend, `/key/delete` is safe |
> | **C2** `reindex_recommended` | **Not patchable as scoped.** The mtime scan (`search/index_source_mtime.py:28-63`) is already correctly limited to metadata + GI/KG/transcript and never sees run summaries. The false positive comes from episode metadata being **rewritten with identical content**, so a real fix needs a content fingerprint stored at index time and compared as a set. That is an index-schema change and a decision, not a patch. |
> | **A2**, **D3**, **D4**, **D5**, **E1**, **F1qa** | Untouched |
>
> **None of the branch code has been executed.** The authoring environment had no venv,
> no pytest and no mkdocs — `ast.parse` plus a 526-link relative-link check is the entire
> extent of local verification. Everything needs CI before it is trusted.

| State | Count | Items |
| --- | --- | --- |
| `[DONE]` | 1 | D1 (residual code TODO still open) |
| `[INFO]` | 2 | C3, part of B1 |
| `[WITHDRAWN]` | 1 | B2 |
| `[OPEN]` | 13 | A1, A2, B1(docs), C1, C2, **C4**, D2, D3, D4, D5, **D6**, E1, F1qa, **G1**, **G2** |

**Highest-value OPEN:** **G1** (jobs wedge silently and report healthy — an unattended batch
can lose hours with no signal), **D6** (no headroom metric anywhere — the first notice of
credit exhaustion is a production failure), **C4** (per-feed counts under-report, which
silently breaks any "top up each feed to N" automation), then D3 / D4 / D5 (gateway
durability), A1 (label bug), C1 (skip mislabel breaks EXIT gates).

> **The recurring shape.** G1 and D6 are the same defect in different domains. Every *detection*
> signal is exception-triggered, so a process that stops working invisibly looks healthy. Every
> *cost* signal is spend-triggered, so an account about to run dry looks healthy. In both cases
> the estate measures what happened and never measures what remains. That is the generalisable
> lesson of 2026-08-12/13, and it is worth applying as a review question to any new signal:
> **does this tell me a thing occurred, or does it tell me how much room is left?**

**D2 can be closed** — verified unused, see the cost analysis below.

### Found during the 2026-08-12 homelab session, not in the original F1–F15

| Item | Where | Nature |
| --- | --- | --- |
| **B1 evidence** | Rollout log | B1's under-count did **not** reproduce 5/5 when discovery ran per-feed; suggests the fix target and that the pending WSJ/NVIDIA +4 investigation can close |
| **Cost blindness** | Rollout log | `prod:4001` unreachable from `tag:homelab-host` — cost cannot be measured from the box that now runs ingestion. Companion to D3/D4 |
| **Enricher artifacts** | Rollout log | Only `topic_cooccurrence_corpus` appears under `/api/corpus/enrichments`; the enrichers the corpus growth is meant to feed are unaccounted for |
| **ACL grant** | Rollout log | `tag:homelab-host → tag:prod:443` added and applied; README corrected |
| **B2 (withdrawn)** | §B | Kept as a record of a disproved claim, not as work |

---

## Rollout log — Step-2 volume batch (+10/feed → ~250)

Run from the **homelab** against the prod operator API. Add-only; `skip_existing=true`
throughout, so every entry is idempotent and re-runnable.

### Infrastructure change made to enable this batch

**2026-08-12 — tailnet ACL: `tag:homelab-host` → `tag:prod:443`.**

Homelab previously held a single grant to prod, `:8099` (the ADR-145 delivery-worker outbox
seam). Every operator-API call on `:443` was dropped by the tailnet packet filter — all
ports timed out rather than refusing, including `:22`, the documented signature of "no
incoming grant" (`policy.hujson:194`).

Diagnosis evidence: `:8099` answered (HTTP 404 — TCP connects) while `:443`, `:80`, `:22`
and `:8000` all returned `curl (28) Operation timed out`, from `100.87.33.61` to
`100.124.111.115`.

Change: `"dst": ["tag:prod:8099"]` → `["tag:prod:443,8099"]`, committed as `fd337a17` and
pushed directly to `main` (branch protection bypassed, at operator direction), which
triggered `.github/workflows/tailscale-acl.yml` in `apply` mode (run `31565516573`,
success). Verified afterwards by the live path, not the workflow status:
`corpus_stats_http=200`.

Rationale for making it durable rather than running from an admin device: the operator
confirmed homelab will drive ingestion on a standing basis over the coming months.

Also corrected in the same commit: `tailscale/README.md` documented the pre-ADR-128
`tofu apply` flow, which has applied nothing since 2026-07-28. Following it leads an agent
to edit the policy, run `tofu apply`, and see no change on the live tailnet.

### Feed results

Baseline at batch start: **180 episodes / 9 feeds.**

Handover feed IDs resolved to real show names via `GET /api/corpus/feeds` (2026-08-12):
`Acast-6478` = **Unhedged**, `Simplecast-l2i9` = **Hard Fork**, `Flightcast` = **Latent
Space**, `NPR` = **Planet Money**.

| # | Show (handover ID) | Offset | Job ID | Status | Corpus | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | The Journal. (WSJ-Journal) | 23 | `5758bed6-0f06-4032-9fba-ba503bdee8e1` | succeeded | 180 → 180 | +0 |
| 2 | NVIDIA AI Podcast (NVIDIA-AI) | 17 | `cb45fa31-33d8-4b77-905b-da9e5d24d851` | succeeded | 180 → 190 | **+10** |
| 3 | Invest Like the Best | 19 | `e9a3ae29-0cde-46c0-9869-4af20dc0e603` | succeeded | 190 → 200 | **+10** |
| 4 | Unhedged (Acast-6478) | 21 | `5d20b36c-a51a-4607-a3b1-4d2770213e79` | succeeded | 200 → 210 | **+10** |
| 5 | The Daily | 25 | `5f1b34a0-5bd3-4705-b06f-76c41ce6750f` | succeeded | 210 → 220 | **+10** |
| 6 | Hard Fork (Simplecast-l2i9) | 19 | `15412156-0ec3-400b-86b8-a2f17b55d150` | succeeded | 220 → 230 | **+10** |
| 7 | Latent Space (Flightcast) | 0 | `f363523d-069b-4697-8587-8d9c9b9b1077` | succeeded | 230 → 230 | +0 |

**BATCH COMPLETE — 180 → 230 (+50). 7/7 succeeded, 0 failed, 0 stale, 0 cancelled.**

### Final per-feed state (verified via `GET /api/corpus/feeds`)

| Show | Before | After | Δ |
| --- | --- | --- | --- |
| No Priors | 27 | 27 | — (not in this batch) |
| Planet Money | 28 | 28 | — (not in this batch) |
| The Journal. | 26 | 26 | +0 (batch-2 already landed) |
| NVIDIA AI Podcast | 16 | 26 | **+10** |
| Invest Like the Best | 15 | 25 | **+10** |
| Unhedged | 17 | 27 | **+10** |
| The Daily | 21 | 31 | **+10** |
| Hard Fork | 17 | 27 | **+10** |
| Latent Space | 13 | 13 | +0 |
| **Total** | **180** | **230** | **+50** |

Arithmetic checks out against `catalog_episode_count` at both ends.

**Expected landing zone: ~235–245**, not 250 — baseline 180 plus six productive feeds at
+8/+10 each (see B1). WSJ contributes ~0 because its batch-2 had already landed before this
session picked up.

### Observations from this batch

- **WSJ +0 is correct, not a fault.** Batch-2 for WSJ completed before handoff; the corpus
  was already 180 when this session started, and `skip_existing` skipped all ten. This is
  also the datapoint showing C1's tally mislabel doesn't reach job-level status.
- **The B1 under-count did not reproduce once — 5 for 5.** This is the most actionable finding
  of the batch. B1 records NVIDIA at +4 of 5 in batch-1, and predicts feeds landing +8/+9
  rather than +10. In this batch **every productive feed delivered a full +10**: NVIDIA,
  Invest Like the Best, Unhedged, The Daily, Hard Fork.

  **Hypothesis (not proven):** batch-1 ran discovery **once up front for all feeds**, so the
  corpus-guid set went stale as earlier feeds added episodes — later feeds in the same batch
  then computed windows against an out-of-date set and short-added. This batch triggered each
  feed **separately, after the previous one finished**, so discovery was effectively fresh
  every time.

  If that holds, it has two consequences worth acting on:
  1. B1's proposed fix ("re-read corpus guids right before each feed") targets the **real
     cause**, and is worth building.
  2. The short adds in batch-1 were never a feed-specific property — so no per-feed
     investigation is needed for the WSJ/NVIDIA +4 under-counts listed as pending work.
     **That pending item can likely be closed.**

  **Not proven:** this is 5 observations from a batch whose sequencing differed from batch-1
  in exactly the suspected variable, but nobody has read the discovery code to confirm the
  mechanism. Someone with SSH should verify before closing the item.

- **Latent Space +0 was predicted and is correct.** It ran at `episode_offset=0` — the only
  feed not given a real offset — which targets the *newest* 10 episodes. It already held 13,
  so the whole window was already present and `skip_existing` skipped it. The broken
  feed-guid mapping B1 flags did not need to be invoked to explain this. **To actually grow
  Latent Space, it needs a non-zero offset** (its back-catalog), not a retry at 0.
- **Possible over-add in the pre-handoff WSJ run — unresolved.** The handover digest has
  NPR ending at 169 with WSJ then in-flight; the corpus measured 180 before this session's
  WSJ job, implying the original run went 169 → 180 = **+11**, above its own
  `max_episodes=10`. Two readings, not distinguishable without SSH: the digest's
  intermediate numbers are approximate, or something added beyond the cap. Noted because it
  runs *opposite* to the B1 under-count, which is the direction everything else drifts.
  Not treated as blocking.

### Pass-3 — RSS discovery, and the Latent Space cap (2026-08-12)

Target: ~500 episodes. Discovery run **from the homelab without SSH**, by fetching each RSS
directly and counting `<item>` elements — this is a viable substitute for `discover10.py` and
needs no box access.

| Feed | In corpus | RSS items available |
| --- | --- | --- |
| The Daily | 31 | 2944 |
| Invest Like the Best | 25 | 592 |
| Planet Money | 28 | 355 |
| Unhedged | 27 | 330 |
| NVIDIA AI Podcast | 26 | 306 |
| The Journal. | 26 | 300 |
| Hard Fork | 27 | 209 |
| No Priors | 27 | 173 |
| Latent Space | 13 | 219 |

**Every feed has ample back-catalog.** The smallest (No Priors, 173) still supports more than
triple the target.

> **Correction:** the first run of this table reported Latent Space at **2** items and
> concluded the feed was dead. That was a measurement error — see the withdrawn **B2**. The
> count above (219) is correct. Counting must use `grep -o "<item>" | wc -l`; `grep -c`
> counts lines and undercounts minified feeds catastrophically.

**Target: ~500 across all 9 feeds**, i.e. **55 per feed**. Invest Like the Best was already
launched at +36 (to 61) before the correction landed and is left to overshoot slightly —
harmless.

| Feed | Now | Target | Deficit | Offset |
| --- | --- | --- | --- | --- |
| Invest Like the Best | 25 | 61 | +36 | 25 (running) |
| Latent Space | 13 | 55 | +42 | 13 |
| The Journal. | 26 | 55 | +29 | 26 |
| NVIDIA AI Podcast | 26 | 55 | +29 | 26 |
| No Priors | 27 | 55 | +28 | 27 |
| Unhedged | 27 | 55 | +28 | 27 |
| Hard Fork | 27 | 55 | +28 | 27 |
| Planet Money | 28 | 55 | +27 | 28 |
| The Daily | 31 | 55 | +24 | 31 |
| **Total** | **230** | | **+271** | **= 501** |

Run as **one job per feed** covering its full deficit, rather than 27 separate 10-windows —
fewer jobs to supervise, and `skip_existing` keeps a retry cheap if one fails.

**Throughput — plan for ~18 hours, not an afternoon.** Invest Like the Best added roughly 15
episodes in its first hour, i.e. **~4 min/episode**, putting the full +271 push on the order of
**18 hours**. The +50 batch took ~3 hours; this one is 5.4× the volume. Rough estimate,
extrapolated from one feed's partial progress — expect variance, since episode length drives
transcription time and The Daily (~20 min/episode) should run far faster than Invest Like the
Best (~90 min/episode).

**Operational note:** a job of this size outlives a 1-hour monitor. Use a persistent watcher,
or the job will silently outrun its supervision and look like it stalled.

### Quality assessment after the +50 batch (2026-08-12)

Requested check on whether quality degraded. **No degradation detected.**

Verified:

- **Coverage is complete:** `total_episodes: 230, with_gi: 230, with_kg: 230, with_both: 230,
  with_neither: 0`.
- **Index agrees:** `episode_title: 230`, `summary_short: 230`, 24 961 vectors.
- **Per-episode density sits inside the batch-1 baseline** — insights 17.6/ep (baseline 6–31),
  KG nodes 22.7/ep (baseline ~20–29), quotes 26/ep.
- **Spot-check** of a newly-ingested The Daily episode: thesis-level `summary_title`,
  substantive bullets carrying named entities and hard figures, and
  `bridge_partition: {gi_only: 0, kg_only: 13, both: 13, total: 26}` — every GI node has a KG
  counterpart.
- **Unplanned benefit:** the batch extended the corpus *backward* — a new month appeared
  (2025-11, 6 episodes) and Feb–May thickened. Useful for `temporal_velocity`.

NOT verified:

- **The new 50 cannot be isolated** from corpus-wide averages — no per-episode stats endpoint
  and no pre-batch index snapshot. In-band averages are evidence, not proof.
- **No semantic grounding check** (F1qa / F14 still open).
- **Only one corpus-level enricher artifact exists:** `topic_cooccurrence_corpus` v1.1.0.
  `topic_perspectives`, disagreement, `guest_coappearance`, `temporal_velocity` and
  `topic_similarity` do **not** appear under `/api/corpus/enrichments`. They may be
  per-episode sidecars rather than corpus artifacts — undetermined from this endpoint.
  **Worth resolving**, since the entire expansion rationale is feeding those enrichers.
- **Cost is unobtainable from the homelab.** `prod:4001` returns `000` (dropped — the ACL
  grants `tag:prod:4001` to `autogroup:admin` only, not `tag:homelab-host`) and
  `/api/corpus/runs` is `404`. Credentials do not help; the packets never arrive. Fixing it
  means adding `4001` to the homelab grant, exactly as `443` was added above.

### Cost analysis (2026-08-12) — the $25 cap does not protect the real bill

Obtained after the `tag:prod:4001` grant landed. In the end the numbers came not from the
gateway API but from **VictoriaMetrics via the homelab start page** (`/vm/api/v1/query`),
which already scrapes both the pipeline and the gateway — no gateway master key needed.

| Metric | Value |
| --- | --- |
| `podcast_pipeline_run_cost_usd_total` | **$45.40** (counter created ~17 h before reading) |
| `litellm_key_spend_usd{key_alias="podcast-prod"}` | $3.156 |
| `litellm_key_max_budget_usd` | $25 |
| `litellm_key_budget_burn_ratio` | 0.126 |
| `openrouter_vertical_usd{vertical="podcast",window="total"}` | $1.379 |

**Cash exposure is small; the large number is modeled cost.** The $45.40 is what the pipeline
*models* the work as costing at list price. Per the operator (2026-08-12), **transcription runs
on a separate Deepgram free allowance**, so that ~$42 is not billed. Actual cash spend is the
**$3.156 LiteLLM key against its $25 cap — 12.6 % burn**, which is the figure that matters and
is comfortably within budget.

> **Correction.** This section originally concluded the $25 cap "does not protect the real
> bill" and projected $46–81 of unplanned spend on the remaining push, and the batch was
> paused on that basis. **That was wrong** — it assumed `podcast_pipeline_run_cost_usd_total`
> represented cash outlay. It does not, given the free Deepgram allowance. The lesson: a
> pipeline's *modeled* cost metric and its *billed* cost are different quantities, and the gap
> between them is whatever sits on free tiers or committed capacity. Check the billing
> arrangement before escalating on a cost metric.

**Per-episode modeled cost: $0.17 – $0.30 all-in**, of which roughly $0.012–0.02 is LLM (the
part that is actually billed). The spread is real uncertainty — the counter was created ~17 h
before the reading and the episode count inside that window is somewhere between 152 and 262.

**Implication for the +271 push:** roughly **$3–5 in real LLM spend**, taking the LiteLLM key
from 12.6 % to an estimated ~25–33 % of its $25 cap. No budget risk.

**Residual worth knowing (not a blocker):** a free allowance still has a **quota**. 271 more
episodes will consume some of it, and nothing in this telemetry shows how much headroom
remains — the Deepgram-side limit is not scraped. Worth surfacing on the ops card if
transcription volume keeps growing, since exhausting the allowance converts ~93 % of modeled
cost into real cost overnight.

**Measurement caveats:**

- `increase(podcast_pipeline_run_cost_usd_total[1h])` and `[2h]` return **0** while a job is
  actively running. The counter increments at run **completion**, so in-flight work is
  invisible and the $45.40 excludes the job running at the time of reading.
- The counter is a `_total` and will reset on api restart; treat it as "since last restart",
  not lifetime.

**Side finding — D2 is verified and safe to close.** The orphan key `proj-podcast-prod` shows
spend `$0.00000088` and burn ratio `0.0000000353`. It is genuinely unused; `/key/delete` is
safe.

**Side finding — ADR-142 routing is probably correct but not proven.**
`litellm_key_spend_usd` carries only `box="prod"` series for podcast keys, with no homelab
series. But `box` label values are `dgx / mini / prod`, so the absence of a `mini` podcast
series may mean the exporter does not cover mini's keys rather than that homelab spend is
zero. Suggestive, not conclusive — D5's proposed deploy post-check is still worth building.

### Pass-3 halt — upstream LLM credit exhausted (2026-08-13 02:12Z)

```
job_id:       8645ecd0-9a96-4235-9f81-13cc70ed65c6   (The Journal, offset 48)
status:       failed        exit_code: 1
duration:     25 seconds    corpus unchanged at 434
error_reason: podcast_scraper.exceptions.ProviderRuntimeError:
              [LiteLLMProvider/SpeakerDetection] OpenAI speaker detection failed: unknown:
              no budget/credit left on this key — this is NOT retryable, so the run is
              hard-stopping (the resilience fuse for money/access, same as the call-count fuse)
```

**This is NOT the LiteLLM virtual-key cap.** Measured at the time of failure:

| Metric | Value |
| --- | --- |
| `litellm_key_spend_usd{key_alias="podcast-prod"}` | **$5.42** |
| `litellm_key_max_budget_usd` | $25 |
| `litellm_key_budget_burn_ratio` | **0.217** (21.7 %) |
| `podcast_pipeline_run_cost_usd_total` | $91.05 (modelled) |

The virtual key had **79 % of its budget remaining**. The message is the **upstream
provider's** credit relayed through the gateway. Every LiteLLM stage in `cloud_balanced` uses
the alias `podcast-flash-0731` (`litellm_summary_model`, `litellm_speaker_model`,
`litellm_insight_model` — profile lines 78, 257, 258), so whatever that alias maps to upstream
is what ran dry.

**Hypothesis, not verified — a shared-account bystander risk.** OpenRouter spend by vertical
at the time:

| vertical | total |
| --- | --- |
| `pi` | **$72.75** |
| `gateway` | $11.53 |
| `opencode` | $6.35 |
| `podcast` | $1.38 (looks stale) |

If those verticals share one prepaid OpenRouter account, the podcast pipeline can be starved
by an unrelated workload. **Unverified:** no credit/balance metric exists in VictoriaMetrics,
and the prod gateway admin API needs a master key. This is inference from spend, not a reading
of the balance.

**Follow-up worth opening:** export an upstream **credit/balance remaining** metric. Every
budget signal we have today measures *spend*, and none measures *headroom* — so the first
notice of exhaustion is a hard job failure. That is the same absence-vs-presence blind spot
G1 exposed, in the money domain rather than the progress domain.

**The fuse behaved correctly, and this is worth recording as a positive.** The run hard-stopped
in **25 seconds** with a precise, correctly-classified-as-non-retryable error, no wedge, no
silence, no orphaned thread, and no partial state — corpus verified at `434/434` with GI and KG,
`with_neither: 0`. Contrast G1's 4 h 15 m of silence on the same day. When the error taxonomy
is right and the stop is deliberate, the machinery works.

### Offsets used

The handover's **fixed** offsets, not live-derived ones — `discover10.py` needs SSH, which
this session does not have. `skip_existing` absorbs the drift; per B1 this is exactly why
some feeds land +8/+9 rather than +10.

### Stop criteria in force

- `status=failed` → stop, report `job_id` + `error_reason`.
- A `+0` from a feed that should produce a delta → stop and report. That indicates a
  wholly stale offset window rather than a pipeline fault, and it's worth a decision rather
  than burning ~30 min per no-op job.
- `+8` / `+9`, or a thin Flightcast, are **not** anomalies (B1). Keep going.
