# HANDOVER — get ONE episode repaired, then 32

**Date:** 2026-08-22 · **Prod sha:** `sha-e68920b` · **Epic:** #1657 · **Tracking:** #1757

---

## The goal, and only the goal

Repair **one** episode — `substack:post:178618026` ("Netflix's Engineering Culture", 59.6 min,
~$0.26 of Deepgram). When that completes end-to-end, repair the remaining 31.

**It has never completed.** Four attempts, all failed for different reasons. Do not get pulled into
fixing adjacent things; every one of those was real and none of them got an episode repaired.

---

## Current production state

`sha-e68920b` is deployed to all three surfaces (control plane, player, operator), verified
container-by-container on the box. No pipeline containers running as of 20:22Z.

Three defects were found and fixed **and verified in production** today:

| what | commit | verified how |
|---|---|---|
| Corpus-wide audio sweep ran at the start of every pipeline run — a 1-episode repair walked all 678 episodes, one rclone round trip each, ~16 min of silence | `673c135d` | run reached the feed loop in ~15s instead of stalling |
| `timeout ... docker compose run` killed the docker CLIENT, not the container — the step reported failure at 29 min and the container ran on for another 14 | `41b52760` | not yet exercised live; verified against a stubbed docker only |
| App logs never reached stdout: OTEL attaches a handler to the ROOT logger at import, so `apply_log_level` took its `else` branch and never added a console handler | `e68920b2` | probe in prod: `HAS CONSOLE HANDLER: True`, probe line visible with `[run= trace=]` |

---

## THE BLOCKER — the repair is a no-op

Last real attempt (run `32585855644`, 31 min, **$0 spent, no ASR call**):

```
16:51:04-12  enrich-edges: episodes=0        <- x9, one per feed. EVERY feed selected ZERO episodes
16:51:15     WARNING reprocess_existing_only is set without reprocess_source; matched on-disk
             episodes will be skipped under skip_existing (no-op). Pair it with
             reprocess_source=whisper_transcription to actually re-diarize (#876).
16:51:17     -> 17:15:16  vector reindex, 678 scanned / 226 reindexed / 23,362 vectors  (24 min)
17:15:16     -> 17:25     topic clustering (killed here, still computing)
17:19:47     exit 124 (remote timeout) — container survived, killed manually at 17:25
```

**Why 0 episodes is not yet explained.** Every line that would explain it goes through the logger,
and the logger was silent at that time. That is now fixed, so **a fresh run should say why.**

Two candidate causes, neither confirmed:

1. `reprocess-prod.yml` passes `--skip-existing` unconditionally but only passes
   `--reprocess-source` when `selection=reprocess_source`. In `episode_ids_worklist` mode the
   matched episode may be skipped. **Counter-evidence:** the metadata-level skip
   (`metadata_generation.py:4222`) is permissive — it allows overwrite when `generate_summaries`
   is on, which `cloud_balanced` sets. So this may not be the real skip.
2. The work-list never matched. `_reprocess_existing_episodes` (`workflow/stages/scraping.py:230`)
   matches `wanted_ids` against on-disk `guid` or `episode_id`. If it matched, it logs
   `reprocess work-list: restricting this run to N of M`; if not, `none of the N listed
   episode(s) are in this feed's corpus`. **Neither line was visible** — because of the logging
   bug. One of them will appear now.

---

## Second blocker — 34 min of corpus-wide work on every run

`finalize_multi_feed_batch` (`workflow/corpus_operations.py:612`) forces
`skip_auto_vector_index: False` and rebuilds the **whole-corpus** vector index, then
`build_topic_clusters_for_corpus`. Measured: **24 min + 10 min**, on a run that touched zero
episodes. Same shape as the audio sweep, at the other end of the run.

**Do not fix this to unblock the repair.** Work around it with `timeout_minutes=90`. Fixing it is
a follow-up.

---

## Observability — USE THIS FIRST

VictoriaLogs is reachable **directly from the dev machine**. No SSH, no workflow needed.

```bash
curl -s "http://homelab:9428/select/logsql/query" \
  --data-urlencode 'query=_time:60m {app="podcast"} | limit 50'
```

Alloy scrapes container logs into it (`infra/observability/operator.alloy:47` explicitly includes
`compose-pipeline-llm[-_].*` as `surface="pipeline"`).

**CAVEAT, unresolved:** a query for `{app="podcast", surface="pipeline"}` over 90m returned
**0 rows**, while `{app="podcast"}` returns data (viewer/web streams). Cause unknown — wrong label
value, wrong time window, or that stream genuinely empty because the app was not writing to stdout
until `e68920b2`. **Verify this before relying on it.** Start by listing what streams exist:

```bash
curl -s "http://homelab:9428/select/logsql/query" \
  --data-urlencode 'query=_time:24h {app="podcast"} | limit 200'   # inspect the _stream values
```

Also available: VictoriaMetrics + VictoriaTraces (ADR-117/119), Grafana, GlitchTip/Sentry.

---

## Tooling built today (all on main)

| workflow | approval? | use |
|---|---|---|
| `stop-prod-pipeline.yml` `dry_run=true` | **NO** | list containers + images. Fast, free. Use constantly. |
| `stop-prod-pipeline.yml` `dump_stacks_first=true` `dry_run=false` | **NO** | SIGABRT each pipeline container (prints a full Python stack of every thread), then kill. This is how the audio-sweep hang was root-caused. |
| `inspect-prod-corpus.yml` `checks=none` `reprocess_log=latest` | yes | read `/srv/podcast-scraper/logs/*.log` from the box |
| `inspect-prod-corpus.yml` `checks=startup_trace` | yes | times import/config/feed-resolution in the prod image + dumps root-logger handlers |
| `inspect-prod-corpus.yml` `verify_recent_runs=true` | yes | POSITIVE assertion `attempts>=1, completed==attempts` — the audit reads `attempts==0` as healthy, so this is the only honest post-repair check |
| `sweep-prod-audio.yml` | yes | the audio sweep, now on-demand. `dry_run=true` deletes nothing |
| `reprocess-prod.yml` | yes | the repair. `diagnose_hang_seconds>0` arms faulthandler via sitecustomize |

⚠️ `inspect-prod-corpus`, `reprocess-prod`, `sweep-prod-audio` and the backups all share
`concurrency: prod-corpus`. **A read queues behind a running repair** — that is a design bug
(reading a log is not a mutation) and it is why live tailing via workflow does not work. Use
VictoriaLogs instead.

---

## DO THIS NEXT

1. `stop-prod-pipeline.yml` `dry_run=true` — confirm nothing is running (a sweep dry-run,
   run `32596796657`, may still hold the concurrency group; kill it, it is not needed).
2. Dispatch the repair and **let it finish**:
   ```
   reprocess-prod.yml
     confirm=PROD_REPROCESS
     selection=episode_ids_worklist
     episode_ids=substack:post:178618026
     use_transcript_cache=false
     cost_cap_usd=2
     timeout_minutes=90          <- MUST exceed 34 min of finalise + the episode's own work
     diagnose_hang_seconds=0
   ```
3. Watch it in VictoriaLogs, not by polling run status.
4. Lines that answer the question:
   ```
   Starting multi-feed podcast scrape ... feeds=9
   Feed start: rss=...
   reprocess work-list: restricting this run to 1 of N on-disk episodes   <- matched
     ...or...
   reprocess work-list: none of the 1 listed episode(s) are in this feed's corpus  <- did NOT match
   selection: 1 of N episodes · H audio-hours · est. $X
   ```
5. If it repairs: verify with `inspect-prod-corpus verify_recent_runs=true`, then do the other 31.

**Cost is bounded**: `cost_cap_usd=2`, one episode ≈ $0.26. The selection gate prices the set
before any download or provider call, so a wrong selection refuses rather than spends.

---

## Traps that cost hours today — do not repeat

- **`timeout` around `docker compose run` signals the docker CLI, not the container.** Killing the
  Actions job leaves the container running. Always confirm with `stop-prod-pipeline dry_run=true`.
- **Corpus episode COUNT cannot change on a reprocess.** Measuring it proves nothing.
- **A missing log line was not evidence of anything** while #1807 was live. It is now.
- **`make lint` is not the lint job.** The job runs six things: `format-check`, `lint`,
  `lint-markdown`, `check-test-policy`, `check-prod-secret-staging`, `actionlint`. Two pushes went
  red on gates that were never run locally.
- **`python-app.yml` gates Stack test, which gates the image publish.** A commit that triggers no
  python-app run publishes no image.
- Any new prod workflow that creates a container **must** stage tmpfs secrets and join
  `docker-compose.secrets.yml` — `make check-prod-secret-staging` enforces it.

---

## Open issues filed today

- **#1807** app logs never reached stdout — FIXED in `e68920b2`, verified in prod
- **#1808** audio sweep cost never measured; nightlies were paying it too
- **#1809** Deepgram retries billed per attempt but priced once — the ledger undercounts
- **#1810** nothing stops two pipeline runs sharing one corpus
- **#1811** reprocessing silently drops enrichment — 127 episodes already lost it

Unfiled and unexplained: the reindex reported `episodes_reindexed=226` on a run that changed
nothing.
