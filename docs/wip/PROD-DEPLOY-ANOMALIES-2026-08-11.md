# Prod deploy + incremental rollout — anomalies & homework (2026-08-11)

Running log of **unusual things** seen while deploying all 3 prod planes (control / operator /
player) and starting the incremental-processing rollout. Each entry: what happened, whether it
blocked, the follow-up.

Session context: deployed engine `sha-371e925` (main HEAD `371e9256`, PR #1547
`fix/long-term-fixes` merged, CI green). Validation is observability-first (GlitchTip /
VictoriaLogs / tailnet API); SSH is last-resort (operator staged the key).

---

## Validation results — end state (all green)

| Check | Result |
| --- | --- |
| 3 planes deployed on `sha-371e925` | control `31462318117` · operator `31463210839` · player `31463411280` — all GREEN |
| Control-plane `/api/health` (tailnet) | 200, `code_version 2.7.0.dev0`, `artifacts_api:true`, `search_api:true` |
| Player `<player-domain>/api/health` | 200 (0.24s) |
| Operator `<operator-domain>/api/health` | 200 (0.16s) |
| Corpus | **107** episodes / 9 feeds (after Step 0; The Daily 12→13) |
| Search index (LanceDB, real) | **107** episodes / **13,397** vectors; `/api/search` returns hits |
| `/api/index/stats` (after D2 touch) | 107 / 13,397, `reindex_recommended:false` |
| Step 0 (one ep, cloud_balanced) | **DONE, GO** — Deepgram + LiteLLM/DeepSeek verified (B4) |
| VictoriaLogs (`{app="podcast"}` 1h) | 13,852 lines ingesting; 0 error/Exception/Traceback |
| GlitchTip project 1 (pipeline+api) | 0 new issues (the 2 in project 6 = <other-project>-staging, unrelated) |

Prod mutations this session (all operator-approved / benign): index incremental upsert (B1), two
mtime `touch`es on the index dir to unstick the stats cache (B3/B4), and the Step-0 pipeline run
(B4, +1 episode, +$0.132).

**Step 1 (skip-existing crux) = FAIL / NO-GO (B5).** Re-running the same feed reprocessed the
already-present episode (+$0.13 wasted) instead of skipping; catalog stayed 107 (no dup). Root cause:
skip is guid-keyed but run-dir-scoped, not corpus-wide (D7). **HOLD volume until D7 lands.** Single
genuinely-new adds still work (Step 0).

---

## A. Deploy-time anomalies

### A1 — Local worktree was 17 commits behind `origin/main` (non-issue, noted)

- `main...origin/main [behind 17]` at session start. Local HEAD `ee99b9ae`; remote `371e9256`.
- Cause: the other agent's `fix/long-term-fixes` (PR #1547) merged to remote main after this
  worktree's last sync. Strictly behind (not diverged) → all local commits are ancestors.
- Impact: none on deploy (GHA deploys from remote main + GHCR image). Flagged only so nobody
  deploys from the stale local tree by hand.
- **Homework:** `git pull` this worktree to `371e9256` before any local build/validation work.

### A2 — `gh` token lacks `read:packages` scope

- Could not list GHCR image tags via `gh api /users/<org>/packages/...` (403 "need read:packages").
- Worked around with `docker manifest inspect ghcr.io/<org>/podcast-scraper-stack-api:sha-371e925`
  → "IMAGE EXISTS". Verification succeeded via a different path; not blocking.
- **Homework:** add `read:packages` to the local `gh` token if GHCR introspection is wanted from
  the laptop (optional; the deploy workflow validates the manifest itself pre-SSH).

### A3 — Control-plane deploy: two NON-FATAL o11y emit failures — RESOLVED (transient)

Run `31462318117` (deploy-prod.yml, `sha-371e925`, 2m7s, deploy itself GREEN). Two annotations:

- `! Sentry release boundary failed (non-fatal)`
- `! deploy-event emit failed (non-fatal)` — "Emit deploy event to VictoriaLogs"

Deploy health was fine (`/api/health OK` after 5s; post-deploy smoke on 6 surfaces ✓). The
operator + player deploys' VictoriaLogs emit steps were ✓, and post-deploy VictoriaLogs shows
**13,852** `{app="podcast"}` lines in 1h with 0 errors → the log path is healthy, the control-plane
emit failure was **transient**. Sentry release marker still worth a glance (see D3).

### A4 — Operator plane clean

Run `31463210839` (deploy-operator.yml, 1m6s) GREEN, no anomaly annotations. Pinned to the
control-plane engine sha (no drift).

### A5 — Player plane clean

Run `31463411280` (deploy-player.yml, 1m55s) GREEN, no anomaly annotations. Pinned to the operator
stack's running engine sha (`sha-371e925`, no drift).

---

## B. Rollout-time anomalies

### B1 — Post-deploy index "reads 94" → actually TWO issues; RESOLVED

Initial read after all 3 planes deployed: `GET /api/index/stats` = **94** for episode_*,
`total_vectors 12294`, `last_updated 2026-08-10T16:36:19Z`; but `GET /api/corpus/stats`
`catalog_episode_count` = **106**. Investigation split this into two distinct facts:

1. **The real LanceDB index WAS short at 94** (12 Planet Money eps missing) — the handover landmine
   was real. `discover_metadata_files` (deployed) correctly yields **106**, so the fixed discovery
   works; the on-disk table just hadn't been rebuilt since the pre-fix drop.
2. **The `/api/index/stats` endpoint ALSO mis-reported** — see B3; it kept showing 94 even after
   the table reached 106.

**Remediation applied (operator approved):** ONE incremental upsert
`POST /api/index/rebuild?path=/app/output&rebuild=false` (WITH the operator key — the route IS
gated; see correction below). It ran ~06:14→06:23, wrote new Lance versions (aux v36 / insights
v19 / segments v22 @ 06:23:04), bringing the table to **106 episodes / 13,328 vectors** (verified by
direct pyarrow query: aux episode_title/description/summary_short = 106 each; insights 106 distinct
eps; segments 105 — one ep has no transcript segments). NOT `rebuild=true` (full-rebuild = mimalloc
SIGSEGV under torch). Search verified functional. **Index RESOLVED: 106, search works.**

**CORRECTION to my earlier claim:** I first said `/api/index/rebuild` was ungated ("tailnet
privacy is the gate", app.py:248-249). WRONG — that was the router-mount deps; the
`OperatorWriteGuard` *middleware* gates by path, and `/api/index/rebuild` IS in `_OPERATOR_BASES`
(app_operator_guard.py). So the upsert needs a valid operator key; the first POST without it → 403.

### B2 — Saved operator API key on the laptop was STALE (rotated) — RESOLVED

- Key file `~/podcast_operator_api_key.txt` was 110 chars, `PROD…` prefix.
- Verified against prod: `GET /api/ops/summary` and `GET /api/jobs` WITH `X-Operator-Key: <file>`
  both returned **403 `Admin access required`** → did NOT match the server's `APP_OPERATOR_API_KEY`
  (`_valid_key`, app_operator_guard.py:82).
- **Cause:** rotated by the "Recreate operator api (apply rotated APP_OPERATOR_API_KEY)" workflow
  (`330711492`, 2026-08-10); the laptop copy predated it.
- **RESOLVED:** the current key is a file-mounted secret (`.env` has none). Read it from
  `/run/secrets/app_operator_api_key` inside `compose-api-1` (64-char, `6a6f…` — a different, newer
  key). Refreshed the laptop file (0600); re-verified `ops/summary → 200`, `jobs → 200`.
- **Homework:** whenever the key is rotated again, refresh the laptop file — it drifts silently.

### B3 — `/api/index/stats` served STALE counts after an in-place upsert (REAL code bug)

- After the upsert brought the table to 106, `/api/index/stats` still reported **94 / 12294** with a
  frozen `last_updated 2026-08-10T16:36:19Z`, while a direct LanceDB query showed **106 / 13328**
  and the tables' latest versions were written 06:23:04. So the endpoint was wrong, not the index.
- **Root cause:** `read_lance_index_stats` (search/lance_index_stats.py) memoizes in `perf_cache`
  keyed on **`perf_cache.lance_mtime(lance_dir)`** — the top-level `lance_index/` dir mtime. LanceDB
  upserts write into per-**table** subdirectories and do NOT bump the parent dir's mtime (it stayed
  Aug 10 16:36). So the cache key never changed → the endpoint kept returning the 94 it computed at
  api start. `_spawn_rebuild_thread` calls `invalidate_newest_index_source_mtime_cache` (a DIFFERENT
  cache), never `clear_index_stats_cache()`. `stats.last_updated` derives from that same stale mtime.
- **Operational fix applied:** `touch /srv/podcast-scraper/corpus/search/lance_index` (mtime-only,
  no data change) → cache key changed → stats now correctly reads **106 / 13328**,
  `reindex_recommended:false`, `last_updated 2026-08-11T06:29:36Z`. This also PROVED the diagnosis.
- **Impact was cosmetic/monitoring only** — search always served the real 106; only stats + the
  viewer index card + `reindex_recommended` were stale. Genuine defect — code fix in D2.

### B4 — Step 0 executed (The Daily, cloud_balanced) — GO, with notes

Triggered `POST /api/jobs?feed=<The Daily>&max_episodes=1&episode_order=newest&skip_existing=true`
(job `5d249ecb`, ~14 min, `succeeded` exit 0). Added "Why Adults Are Getting Cancer at…".

- Corpus **106 → 107** (The Daily 12→13); NOT 108 → no duplication. Outcome `{ok:1,failed:0,skipped:0}`.
- **Providers verified from the episode `config_snapshot` (on-disk, authoritative):** transcription
  = **Deepgram `nova-3`** (9 speakers, `transcribe_time=9.7s` — cloud, not local Whisper); LLM =
  **LiteLLM → `podcast-flash-0731` → `openrouter/deepseek/deepseek-v4-flash`** with app-level
  RFC-106 failover to native DeepSeek (infra/litellm/config.yaml). Matches cloud_balanced exactly.
  (Note: `content.whisper_model=nova-3` is a legacy MISNOMER — the value is the Deepgram model, and
  `content.transcript_source=whisper_transcription` is likewise a stale label; the real provider
  fields under `ml_providers.transcription` say deepgram. Confusing naming — cleanup candidate.)
- Index reindexed to **107 / 13,397 vectors** (`episodes_reindexed=107, errors=[]`).
- Cost **+$0.132** for one episode (0.8345→0.9664) — **~2× the handover's ~$0.05 estimate**;
  by_stage shows non-zero transcription + cleaning + GI + KG. Worth understanding why per-episode is
  higher than expected before scaling (Step 2 `$5/run` soft cap assumes cheaper eps).
- 0 new errors in GlitchTip podcast project 1. (2 issues in project **6 = <other-project>-staging**, a
  DIFFERENT app, unrelated.)
- **D2 bug RECURS after the pipeline reindex too:** post-run `/api/index/stats` showed stale 106 +
  `reindex_recommended:True` until a manual `touch` of the lance dir. So D2 affects the normal
  processing path, not just manual upserts — bumps its priority.
- **Watch:** `enrich-edges: HAS_EPISODE=1 MENTIONS=0 SPOKEN_BY=0` for the new episode despite 9
  diarized speakers — the handover's Step-0 EXIT expected SPOKEN_BY edges. Possible GI-edge
  derivation gap for incrementally-added episodes; verify before Step 1/2.

### B5 — Step 1 (skip-existing crux) FAILED — reprocess, no dup; NO-GO for scaling

Re-ran the SAME feed (The Daily) with `skip_existing=true` (job `bea9f8bd`, succeeded exit 0).
**It did NOT skip** — it re-ran the full pipeline on the already-present episode:

- Fresh Deepgram transcript + asr/cleaned/gi/kg/media all re-derived under a NEW run dir
  `run_bea9f8bd_…` (transcript written 07:11). Cost **+$0.130** wasted (0.9664→1.0967, run_count 12).
- Corpus stayed **107** (The Daily still 13) → **no duplicate episode** (episode_id dedup +
  newest-run-wins protect the catalog COUNT). But the reprocess is real spend + a second run dir for
  the same episode (the Step-0 `run_5d249ecb` is now orphaned/superseded).
- **Verdict: NO-GO** per the handover — hold volume; do not scale to Step 2 until fixed.

**Root cause (deployed code, episode_processor.py:321-346):** the GUID fix (#1) landed —
`skip_idx = run_index.resolve_ondisk_idx_for_episode(episode, effective_output_dir)` keys on the
STABLE guid, and skip triggers on `os.path.exists(final_out_path)`. BUT both the guid lookup and
`final_out_path` are scoped to `effective_output_dir`, which under `--single-feed-uses-corpus-layout`
is the FRESH run dir (empty at check time). The episode lives in a PRIOR run dir, so the check
can't see it → falls back to `episode.idx` → path absent → reprocess. **The skip is guid-keyed but
run-dir-scoped, not corpus-wide** — the same latent "only look at one run" assumption that
`3c01787d` fixed for index discovery, still present in the skip path. → homework D7.

---

## Phase 1 investigation conclusions (2026-08-11) — RE-SCOPED

Ran every homework item to ground before committing to implement. Outcome: scope SHRANK a lot.

| ID | Verdict | Needs image rebuild? | Effort |
| --- | --- | --- | --- |
| **D7** skip corpus-wide | **REAL — the only scaling blocker.** cfg.output_dir=corpus root is in scope; `_scan_corpus_metadata_index` already globs `feeds/*/run_*/metadata` corpus-wide. Fix = resolve + existence-check against corpus root (via entry's real path) in corpus-layout mode, not the fresh run dir. 3 call sites (episode_processor.py:325/2766/2877 + `_check_existing_transcript` glob). | YES (pipeline) | Moderate + Tier-2 test |
| **D2** stats cache | **REAL, small.** `perf_cache.lance_mtime` = `getmtime(lance_dir)` (top dir only); in-place upserts don't bump it. Fix = clear_index_stats_cache() after build / walk-newest / os.utime. Also hits the pipeline reindex path. | YES (api) | Small + test |
| **D1** unified deploy workflow | Operator ask, valid. **Pure GHA workflow YAML — NO image rebuild, ships on merge.** | **NO** | Moderate (YAML) |
| **D6** whisper_* misnomer | Baked into a schema enum `Literal[...,"whisper_transcription"]`; correct data already in `config_snapshot.ml_providers`. Document now; full rename is a later schema change. | (only if renamed) | Low (doc) / Large (rename) |
| **D3** deploy-event/Sentry | **NON-ISSUE.** `sha-371e925` release IS in GlitchTip (05:53:55); VictoriaLogs healthy (13.8k lines/1h). Only the cosmetic deploy-boundary annotation sub-step is flaky. | — | Drop |
| **D4** SPOKEN_BY/MENTIONS=0 | **FALSE ALARM.** Episode gi.json has 14 SPOKEN_BY edges + 2 Person nodes; speaker attribution ran (named=3, no alarm). MENTIONS=0 corpus-wide (247/247) — not implemented, pre-existing. The enrich-edges "0" was the corpus-linkage delta. | — | Drop |
| **D5** cost 2× | **EXPLAINED, not a bug.** `metrics.json llm_transcription_cost_usd=0.1203` on 1577s (~26min) of Deepgram nova-3 ≈ correct pricing. The $0.05 estimate was wrong. speaker_detection cost=0 because free self-intro heuristics resolved names (no LLM call). | — | Doc: update planning to ~$0.13/ep |
| **D8** full reindex per add | **NEW.** `vector_index_seconds=635` — every incremental add does a full ~10.6min corpus reindex (no incremental checkpoint). Real scaling cost; larger change. | (future) | Defer — investigate separately |

**Net implementation scope for THIS round:** **D7** (blocker) + **D2** (rides the rebuild) require the
image rebuild+deploy. **D1** ships independently (workflow only). D6 = doc note. D3/D4/D5 = closed.
D8 = deferred.

---

## Implementation status (2026-08-11) — branch `feat/incremental-processing-and-o11y`

Repro-first (each bug reproduced RED, then fixed to GREEN). Not committed/pushed yet.

| Item | Status | Code | Guardrail tests (new) |
| --- | --- | --- | --- |
| **D8** indexer incremental skip | ✅ DONE | `search/two_tier_indexer.py` (fingerprint skip + `_finalize_index_build`), `search/indexer.py` (wire `episodes_skipped_unchanged`) | `test_two_tier_indexer_incremental.py` (5), composition test |
| **D7** skip-existing corpus-wide | ✅ DONE | `workflow/run_index.py` (`episode_metadata_rel_in_corpus`, `existing_transcript_path_in_corpus`), `workflow/episode_processor.py` (both `_check_existing_transcript` + the ASR path) | `test_skip_existing_across_runs.py` (4, incl. ASR path) |
| **D2** index-stats freshness | ✅ DONE | `search/two_tier_indexer.py` (`os.utime` the index dir on a changed build — crosses the pipeline-subprocess boundary) | `test_index_stats_freshness.py` (1) |
| **D9** observability | ✅ DONE | `providers/ml/embedding_loader.py` (`show_progress_bar=False` — kills the "Batches" flood), `providers/deepgram/deepgram_provider.py` (announce provider+model like Whisper) | `test_deepgram_provider.py::test_transcribe_logs_provider_and_model` |
| **Composition** (Step 0+1 codified) | ✅ DONE | — | `test_multi_run_corpus_incremental_index.py` (union+newest-wins → index count; unchanged reindex = 0 embeds) |
| **D1** deploy-all-prod | ✅ FILE ADDED (actionlint-clean) | `.github/workflows/deploy-all-prod.yml` — one trigger, control→operator→player, stop-on-failure; keeps per-plane approval. Needs `secrets.DEPLOY_ORCHESTRATOR_PAT` + a live dispatch to validate (GHA can't be tested locally). | — |
| **D6** `whisper_*` misnomer | ⏸ DEFERRED (rationale) | Field VALUE already = the actual provider's model; provider is now unambiguous in logs (D9) + `config_snapshot.ml_providers`. A true rename touches a schema `Literal` enum + many readers — a separate, higher-risk change, not rushed here. | — |
| D3 / D4 / D5 | CLOSED | non-issues / cost doc-corrected (~$0.13/ep) | — |

**Verification:** affected suites green — search unit/integration 450+, workflow unit 800, deepgram
36, the 5 new guardrail files red→green; flake8 clean on all changed source+tests; actionlint clean
on the new workflow. NOT run yet: full `make ci-fast` (final pre-push gate).

---

## D8 RCA (2026-08-11) — search reindex is O(corpus), no incremental skip [BLOCKER for scaling]

Every incremental add re-embeds the ENTIRE corpus. Step-0 metrics: `episodes_scanned=107,
episodes_skipped_unchanged=0, episodes_reindexed=107, vector_index_seconds=635` (~10.6 min for ONE
new episode).

**Confirmed root cause:**
- `build_two_tier_index` (two_tier_indexer.py:295) loops `discover_metadata_files(out)` and calls
  `_embed(text)` for every doc of every episode (lines 338/352/365). **No fingerprint gate, no
  unchanged-skip.**
- `index_corpus` sets `episodes_reindexed = tt.episodes` (all) and NEVER sets
  `episodes_skipped_unchanged` → it is a DEAD metric (always 0), left from the retired FAISS indexer.
- `episode_fingerprints.json` does NOT exist on the box — the skip mechanism is not active.
- Upsert is idempotent for STORAGE (merge on id) but NOT for COMPUTE: it re-embeds all N first,
  then merges. Embedding is the 635s.
- Complexity is O(N) per add → 107 eps = 10.6 min; ~500 eps ≈ ~50 min per single-episode add.

**Fix direction:** add a per-episode unchanged-skip to `build_two_tier_index`: hash each episode's
index-relevant content (transcript chunks + insights + gi/kg), keyed by the STABLE
`index_fingerprint_scope_key(feed_id, episode_id)` (already exists in corpus_scope.py), persist to
`episode_fingerprints.json`, and skip embed+upsert for episodes whose hash is unchanged. Makes an
incremental add O(new episodes). Moderate-large change in the indexer hot loop + tests. **This is a
hard prerequisite for Step 2 volume, alongside D7.**

## D9 RCA (2026-08-11) — pipeline is observable only AFTER the run, not during

The session hit real "did Deepgram even run?" confusion. Root causes:
- **No stage→provider logging.** The pipeline log mentions deepgram/litellm/nova-3 **0 times** (grep
  count 0, any level). Provider truth lives ONLY in the episode `config_snapshot` metadata, written
  at the END of the run. You cannot tell from logs which provider is running a stage live.
- **Log signal-to-noise is terrible.** 13,889 lines dominated by docker-pull progress + `Batches:
  100%` embedding bars + `Loading weights`. Real stage lines are buried; the per-job byte cap can
  truncate real errors (the code comments on this).
- **log-tail API friction.** `GET /jobs/{id}/log-tail` enforces `max_bytes>=4096` (I passed 1200 →
  422). Field is `text`. Even used correctly it returns the noisy raw tail, not a stage summary.
- **Cost/spend not live-queryable.** LiteLLM SpendLogs was empty at query time (spend-push batch
  delay) and `LITELLM_MASTER_KEY` is a file-mounted secret (not in `.env`), so the documented
  `gateway_spend.py` path didn't work live. The per-run truth is `metrics.json` in the run dir.
- **Misnamed fields (D6)** point at "Whisper" when Deepgram ran (`whisper_model=nova-3`,
  `transcript_source=whisper_transcription`) — actively misleading.

**Fix direction (D9):** (a) emit one structured INFO line per stage naming provider+model+timing
(`transcribe: provider=deepgram model=nova-3 speech_s=1577 cost=$0.12`; `llm summarize:
provider=litellm model=podcast-flash-0731→openrouter/deepseek-v4-flash`); (b) route tqdm/progress
bars to non-tty / DEBUG so the log-tail shows stages, not embedding spam; (c) surface a live
stage/cost signal (metrics.json tail or a `/api/jobs/{id}/progress` summarizing current stage +
running cost); (d) fix the D6 field names so provider identity is unambiguous.

---

## C. Known landmines carried in from the handover (watch for these)

- **Index may read 94 not 106** if the pre-fix image was live when the index was last built → ONE
  incremental upsert (`rebuild=false`), NEVER full-rebuild (mimalloc SIGSEGV under torch). (Hit +
  resolved this session — B1/B3.)
- Run container commands as **`podcast`, never root** (root-owned index files → api "no_index").
- **skip-existing is idx-keyed, not GUID** → silent duplicate reprocessing if a feed published
  between runs. Step 1 is the go/no-go test for this. **Not yet run.**
- Rollback is **SSH-only** today (no DELETE API) — the one hard prod-SSH dependency.

---

## D. Homework — proposed follow-up work

### D1 — One unified "deploy all prod planes" workflow (operator ask, 2026-08-11)

Today deploying the full app is **3 separate `workflow_dispatch` runs** (deploy-prod →
deploy-operator → deploy-player), each with its OWN typed confirm AND its OWN `prod`
required-reviewer gate → **3 triggers + 3 approval clicks**. Painful and error-prone (easy to
forget a plane → engine drift).

**Want:** a single orchestrator workflow (e.g. `deploy-all-prod.yml`) that runs the three planes
**sequentially in the correct order** (control → operator → player), with **one trigger + ideally
one approval**, and a **validation gate after each plane** before proceeding to the next.

Design notes / constraints:

- Order is load-bearing: control plane FIRST (owns/writes the corpus + defines the engine sha the
  public planes pin to), then operator, then player.
- **Stop-on-failure:** if control fails, do NOT roll operator/player onto a bad engine.
- Per-plane validation between steps (reuse `scripts/ops/post_deploy_smoke.sh` / the external
  health probes each plane already runs; add the index-freshness check after control).
- Approval model: a single `environment: prod` gate on the orchestrator job (one click) is the
  goal — confirm that satisfies the "one human checkpoint" intent vs. per-plane gates. Reuse
  `workflow_call` / a `needs` chain of the three existing workflows rather than duplicating the
  deploy logic — keep one source of truth.
- All three pin the SAME sha so "roll all three to one sha" (PROD_RUNBOOK coupling rule) holds by
  construction.

### D2 — Fix the index-stats cache invalidation after in-place upserts (from B3)

`/api/index/stats` (and the viewer index card + `reindex_recommended`) go stale after an
incremental `POST /api/index/rebuild?rebuild=false`, because the perf_cache key
(`perf_cache.lance_mtime` = top-level `lance_index/` dir mtime) doesn't change when LanceDB upserts
write into table subdirectories. Fix options (pick one, add a regression test):

- Have `_spawn_rebuild_thread` (routes/index_rebuild.py) call `clear_index_stats_cache()` in its
  `finally` (it already calls `invalidate_newest_index_source_mtime_cache` — add this sibling). Most
  direct + local.
- OR make `perf_cache.lance_mtime` stat the NEWEST file recursively under the dir (not just the top
  dir). More robust but touches a shared helper.
- OR `os.utime(lance_dir)` at the end of a successful build so the dir mtime tracks table writes.
- Regression test: build index → upsert a new episode → assert `/api/index/stats` count increments
  WITHOUT an api restart or manual touch.
- Land it as its own small fix on `main` (independent of the naming-arc branch).

### D3 — Verify the deploy-event → VictoriaLogs / Sentry release path (from A3)

VictoriaLogs ingestion is confirmed healthy (13,852 lines/1h), so the control-plane
`deploy-event emit failed` was transient. Still worth confirming the Sentry/GlitchTip **release
marker** actually lands for `sha-371e925` (the "Sentry release boundary failed" annotation). If a
real gap, fix at cause (don't suppress).

### D4 — `enrich-edges` produced no SPOKEN_BY / MENTIONS for the Step-0 episode (from B4)

Despite Deepgram diarization finding 9 speakers, the new episode's GI edges show
`HAS_EPISODE=1 MENTIONS=0 SPOKEN_BY=0`. Investigate whether incrementally-added episodes get their
speaker/mention edges derived (compare to a batch-run episode's gi.json). If a real gap, it under-
populates the KG for every incremental add. Verify before scaling (Step 1/2).

### D5 — Per-episode cost is ~2× the planning estimate (from B4)

Step 0 cost +$0.132 for one episode vs the handover's ~$0.05–0.06. Break down `by_stage`
(transcription vs cleaning vs GI vs KG) for a single episode and confirm it's expected, because the
Step 2 `$5/run` soft cap + "≤5 eps/feed" batching math assumes the cheaper number.

### D6 — Rename the legacy `whisper_*` transcription fields (from B4)

`content.whisper_model` and `content.transcript_source=whisper_transcription` are stale misnomers —
they carried a Deepgram `nova-3` value this run and nearly triggered a false "wrong profile" alarm.
Rename to provider-neutral fields (or set them from the actual provider) so logs/metadata don't
imply Whisper when Deepgram ran.

### D7 — Make skip-existing corpus-wide, not run-dir-scoped (from B5) — BLOCKS scaling

`skip_existing` is guid-keyed (good) but its existence check
(`resolve_ondisk_idx_for_episode(episode, effective_output_dir)` + `os.path.exists(final_out_path)`,
episode_processor.py:321-346) only looks in the CURRENT run's `effective_output_dir`. Under
`--single-feed-uses-corpus-layout` that's a fresh empty run dir, so an episode already present in a
PRIOR run dir is not detected → full reprocess (Deepgram + LLM spend) on every incremental re-run,
though episode_id dedup prevents a duplicate in the catalog COUNT.

Fix: resolve the episode's existence by guid across ALL run dirs for the feed (reuse the
`corpus_scope` union / newest-run-wins discovery that `3c01787d` built for the index), not just the
current run dir. Add a Tier-2 matrix row: add an episode, re-run same feed with `skip_existing`,
assert 0 transcription cost + episode skipped + no new run dir with content.

**Until D7 lands: HOLD volume.** Each incremental re-add reprocesses already-present episodes
(wasted spend). Single genuinely-new episodes still add correctly (Step 0 worked). Also clean up the
orphaned duplicate run dir for the Step-0/1 Daily episode (`run_5d249ecb` superseded by
`run_bea9f8bd`).

---

## E. Next actual step (rollout, deferred)

Deploy + post-deploy validation are DONE and green. The real incremental-processing rollout has NOT
started. Resume point (needs a pipeline run + the now-valid operator key):

- **Step 0** — one episode, one feed, full end-to-end via `cloud_balanced`; diff the whole chain
  against baseline (episode count +1, outcome `ok`, index/clusters/cost move, 0 errors).
- **Step 1 (the crux)** — re-run the same feed with skip-existing; PASS = count stays flat + episode
  skipped (idx-skip holds), FAIL = duplicate → hold volume until GUID-keyed skip (fix #1) lands.

The full step-by-step + toolkit lived in `docs/wip/INCREMENTAL-PROCESSING-VALIDATION-ROLLOUT.md`,
removed 2026-09-03 (operator network identifiers); recoverable from git history.
