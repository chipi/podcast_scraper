# Incremental episode processing — validation & rollout playbook

**Status:** Handover + plan for the NEXT session. The corpus/index/deploy fixes are committed to
`main` (see "What's on `main`" below); the next agent picks this up AFTER the deploy lands.
Companion to `docs/guides/PROD_VALIDATION_QUICKREF.md` (the 2-min prod-change checks) — this doc is
the *processing-run* plan: how to add episodes to the live prod corpus cautiously, and exactly which
tool validates each step.

---

## STATUS — 2026-08-10 (READ THIS FIRST)

**Where we are:** all pre-req fixes are on `main`, **not yet deployed to prod**. Step 0 (one Planet
Money episode via `cloud_balanced`) already succeeded earlier — the live corpus is now **106
episodes** (was 105). Nothing below has run since. Next move is the deploy, then Step 1.

**What's on `main` (awaiting the next image rebuild + deploy):**

- `3c01787d` — **index every episode across all runs** (the 94→106 fix). Corpus discovery used to
  keep only the latest `run_*` per feed, so an incremental add (a new run dir holding only the new
  episode) silently dropped the feed's prior run — prod index had fallen to **94 of 106** (all 12
  Planet Money batch episodes). New central rule in `search/corpus_scope.py`: **union across ALL
  runs, one winner per `(feed_id, episode_id)`, newest run wins** (newest = run-folder timestamp;
  file-mtime fallback only for timestamp-less `run_append_*`). `server/corpus_catalog.py` converged
  onto the same rule so the library + index can no longer diverge. Unit + integration tested green.
- `33d0606e` — incremental adds integrate into the corpus (finalize targets the corpus root; index
  and topic-clusters recomputed for single-feed adds).
- `d0b340a8` — diagnosability + deploy (traceback→stderr on pipeline failure; `deploy.sh
  --force-recreate`; `scripts/ops/incremental_step_validate.sh`).

**INDEX VERIFICATION — do this FIRST, right after the deploy:**

1. `GET $B/api/index/stats?$Q` → `episode_title`/`description`/`summary_short` should each read
   **106** (not 94); `reindex_recommended:false`.
2. Deeper read-only proof (in the `compose-api-1` container **as the `podcast` user, never root**):
   open LanceDB at `/app/output/search/lance_index`, collect distinct `episode_id`, diff against
   distinct `episode.episode_id` across `/app/output/**/*.metadata.json`. Expect **0 missing / 106
   indexed**.
3. If the deployed index still reads 94 (the pre-fix image was live when it was built): run ONE
   plain **incremental** upsert — `POST $B/api/index/rebuild?$Q&rebuild=false $OK` — which upserts
   the 12 back in. **Do NOT full-rebuild the prod index** (`rebuild=true`) — crash-prone mimalloc
   SIGSEGV under torch; incremental upsert only. Run container commands as `podcast`, not root, or
   the index files become root-owned and the api can't read them (→ "no_index", search down).

**Coordination:** another agent is landing a batch of issue fixes in a separate worktree; the
operator combines into one PR and rebases on top of this `main`, then runs full `make ci`. That
issue work is out of THIS plan's scope — it rides the same image but does not change the rollout
sequence below.

**Resume point:** after the deploy → re-capture baseline (now 106) → **Step 1: the skip-existing
crux** (§3). Step 0 is done.

---

**Premise (original):** the fixes are merged, image rebuilt, deployed, and post-deploy validated
(QUICKREF + MCP tools green). Then: "execute the plan." Written so that happens end-to-end
**without discovering a missing key or a missing tool mid-run.**

---

## 0. The one finding that shapes everything

Skip-existing is keyed on `episode.idx` (a **run-local sequential position**), not the RSS
GUID (`scraping.py:323-326`; `episode_processor.py:337`). When a feed publishes between runs,
every existing episode's `idx` shifts → `--skip-existing`/`--append` stop matching on-disk
files → **silent duplicate reprocessing** (wasted cloud spend + a duplicate episode in the
corpus). Not corruption (the `episode_id` check prevents wrong-file overwrite), but "messy."
→ **Fix #1 (GUID-keyed skip) is the prerequisite for any *volume*.** Until it lands, the plan
below is deliberately tiny and empirically tests whether idx-skip holds on *this* corpus before
scaling.

---

## 1. Access you must have ready BEFORE executing (anticipate; don't discover mid-run)

| Need | What it unlocks | Have it? |
|---|---|---|
| **Operator API key** (`X-Operator-Key` header) or an admin session cookie | Trigger runs + edit feeds: `POST /api/jobs`, `PUT /api/feeds`, `/api/ops/*`. Gated by `app_operator_guard.py:33-47`. | **MUST be provided to the agent up front.** Read-only corpus/index GETs worked over the tailnet un-authed this session, but *triggering* a run does not. |
| Tailnet (Tailscale up) | Reach `https://prod-podcast.tail6d0ed4.ts.net` + `http://homelab:{8428,9428,10428}` | Yes (agent has tailnet) |
| `~/.ssh/homelab_mini` | GlitchTip DB / Umami / anything on homelab not exposed as HTTP | Yes (agent has it) |
| **`~/.ssh/podcast_prod_operator`** (prod box) | **Rollback only** — delete a `run_*` dir. There is NO HTTP DELETE for corpus artifacts. Key is transient (operator re-adds). | **Confirm available before any increment, OR land the rollback API (gap #1).** This is the ONLY hard SSH dependency in the whole plan. |

**Base + auth for every mutating call below:**
```sh
B=https://prod-podcast.tail6d0ed4.ts.net ; Q=path=/app/output
OK='-H X-Operator-Key:<OPERATOR_KEY>'      # only for POST/PUT (jobs, feeds, ops)
```

---

## 2. The toolkit — every signal, its endpoint/query, and that it needs NO prod SSH

Trigger + watch (operator-gated, tailnet):
- Trigger: `POST $B/api/jobs?$Q $OK`  (`routes/jobs.py:81`)
- Scope to one feed first: `GET/PUT $B/api/feeds?$Q`  (`routes/feeds.py:72-134`) — trim `feeds.spec.yaml` to the single target feed, run, then **restore the full list** (or scheduled sweeps see only the trimmed set).
- Watch: `GET $B/api/jobs` / `GET $B/api/jobs/{id}` → `status ∈ {queued,running,succeeded,failed,cancelled,stale}`, `error_reason`, `log_relpath` (`schemas.py:1310`). Progress text: `GET $B/api/jobs/{id}/log-tail`.

Post-run validation (read; worked un-authed over tailnet this session):
- **Cost:** `GET $B/api/corpus/documents/manifest?$Q` → `.cost_rollup` (`total_cost_usd`, `by_stage`, `run_count`) — auto-written every run (`corpus_operations.py:255-318`). *(Not rendered in the viewer — curl it.)*
- **Episode count / no-dup:** `GET $B/api/corpus/feeds?$Q` → per-feed `episode_count`, or `GET $B/api/corpus/stats?$Q` → `catalog_episode_count`. Count must rise by **exactly N**; a jump of 2N on a re-run = the idx-skip duplication bug.
- **Per-episode outcomes / quality:** `GET $B/api/corpus/runs/summary?$Q` → `episode_outcomes{ok,failed,skipped}` (`corpus_metrics.py:60-140`).
- **Index fresh:** `GET $B/api/index/stats?$Q` → `reindex_recommended:false`, `stats.doc_type_counts`, `last_updated`.
- **Topic clusters fresh:** `GET $B/api/corpus/topic-clusters?$Q` → clusters > 0 (auto-recomputed in finalize).
- **GI/KG coverage:** `GET $B/api/corpus/coverage?$Q`.

Observability (homelab; tailnet, no prod SSH):
- **Errors (pipeline project):** GlitchTip — `ssh -i ~/.ssh/homelab_mini homelab '/usr/local/bin/docker exec glitchtip-postgres-1 psql -U glitchtip -d glitchtip -tAc "select project_id,count(*),max(first_seen) from issue_events_issue where first_seen > now() - make_interval(hours=>2) group by 1"'` (project **1 = podcast/pipeline+api**; pipeline has its own DSN via `PODCAST_SENTRY_DSN_PIPELINE`, `sentry_init.py:33-38`).
- **Traces:** OTEL is **ON** in prod for the pipeline (`docker-compose.vps-prod.yml:44-52` → `homelab:10428`). `curl http://homelab:10428/select/jaeger/api/services` → expect `podcast-pipeline`.
- **Logs:** `curl -sG http://homelab:9428/select/logsql/query --data-urlencode 'query={app="podcast"} _time:1h (error OR Exception OR Traceback)'`. ⚠ **Verify pipeline-container logs actually ship** (gap #3) — the run's own stdout is always available via the `log-tail` API regardless.
- **Correlate everything by run_id:** `podcast_obs` — `GET $B/api/ops/summary` (`routes/ops.py:27`) for a 24h glance, or the `podcast_obs correlate --run-id <id>` path (`podcast_obs/aggregate.py:223-262`) which pulls `llm_cost` + logs + errors + trace for one run. **Get `<run_id>` from the job's log-tail** (`[run=<id> trace=<id>]` on every line) — until gap #2 lands, `job_id ≠ run_id`.

---

## 3. Rollout — baby steps (do NOT commit to "process 100")

### Baseline (prod `sha-5226a6c`, captured 2026-08-09, pre-processing) — diff every run against this
- **105 episodes** / 9 feeds (12·12·12·12·12·12·10·12·11)
- Index 13,645 vectors: transcript 3591 · insight 2598 · quote 3695 · kg_topic 989 · kg_entity 1547 · episode_title/description/summary_short 105 each · summary 910. `reindex_recommended:false`, `last_updated 2026-08-08T21:56:48Z`
- Topic clusters 27 · `cost_rollup.total_cost_usd 1.142609` · run_count 18
- **Re-capture this baseline right after the deploy** (the fixes-PR image changes it) before Step 0.

### Step 0 — one episode, one feed, full end-to-end
`PUT /api/feeds` → single target feed. `POST /api/jobs` (`cloud_balanced` is the deployed
operator profile). Watch to `succeeded` (poll `GET /api/jobs/{id}` + `GET /api/jobs/{id}/log-tail`).
Then check the WHOLE chain against baseline.

**EXIT = GO only if ALL of these hold (any miss → NO-GO → roll back the run, diagnose):**
- Episodes **105 → 106**, and the **target feed +1** (`GET /api/corpus/feeds`). A jump to **107 /
  +2 = the duplication bug → NO-GO.**
- `GET /api/corpus/runs/summary` → the new episode's outcome is **`ok`** (not `failed`/`skipped`).
- Index (`GET /api/index/stats`): `episode_title`/`description`/`summary_short` each **105 → 106**;
  transcript/insight/quote/kg counts rose; `reindex_recommended:false`; `last_updated` newer.
- Topic clusters (`GET /api/corpus/topic-clusters`) recomputed, **>0** (not error).
- Cost (`GET /api/corpus/documents/manifest` → `.cost_rollup`): `total_cost_usd` rose by **~$0.05–0.06**;
  `by_stage` shows transcription + llm for 1 ep; **not $0** (`cost_appears_uninstrumented`) and **not** a large jump.
- Edges: new episode's `gi.json` has `SPOKEN_BY`/`HAS_EPISODE`. If fix #2 (auto `enrich-edges`)
  didn't land, this is **stale → run `enrich-edges` manually** and re-check (SSH until that fix).
- Errors: **0** new issues in GlitchTip project 1 (pipeline) since the run; `log-tail` has no
  traceback and `error_reason` is null; a `podcast-pipeline` service reports at `homelab:10428`.
- **Restore the full `feeds.spec.yaml`** (`PUT /api/feeds`) — don't leave the spec trimmed.

### Step 1 — the skip-existing reliability test (THE crux, ~$0.06). Run it before ANY scaling.
Re-run the **same** feed, same window, with skip-existing/append.

**EXIT verdict:**
- **PASS → GO to Step 2:** episodes stay **106** (target feed unchanged), `log-tail` shows the
  episode skipped, ~$0 transcription in `by_stage`. → idx-skip holds on *this* corpus.
- **FAIL → NO-GO, hold volume:** episodes **106 → 107** (duplicate) or transcription re-ran →
  idx-drift is real here → **roll back the dup** (DELETE API, or SSH until it lands) and treat
  **fix #1 (GUID skip) as a hard prerequisite** before any further adds.

This single ~6-cent test is the go/no-go for the whole strategy — nothing scales until it passes.

### Step 2 — scale by the Step-1 result
- **Skip works:** feed-by-feed, **≤5 new episodes per feed**, one feed at a time (fewer
  corpus-level index/cluster recomputes than round-robin; cleaner isolation + rollback). Full
  §2 validation between each increment. Keep increments ≤10 (the `$5/run` soft cap is a
  cumulative, **no-rollback** abort — a small batch makes a mid-run abort a clean resume).
- **Skip fragile:** hold volume until fix #1 lands; in the meantime only Step-0-style
  single-episode adds with an explicit, hand-checked window.

---

## 4. Rollback (the ONLY prod-SSH dependency today)

If an increment is bad: `ssh -i ~/.ssh/podcast_prod_operator deploy@prod-podcast.tail6d0ed4.ts.net`
→ `rm -rf /srv/podcast-scraper/corpus/feeds/<slug>/run_<the_new_run>` → then **full reindex**
`curl -X POST $B/api/index/rebuild?$Q&rebuild=true $OK` (the index upserts with **no orphan
sweep** — deleting files alone leaves stale vectors; `two_tier_indexer.py:196-221`). Manifest
cost-rollup self-corrects; topic-clusters recompute on the rebuild.
→ **This is why prod SSH must be pre-confirmed, or gap #1 (rollback API) landed first.**

---

## 5. Observability/API coverage — COVERED vs GAPS (fix before executing)

**Covered, no SSH:** trigger (jobs+feeds API), watch (jobs + log-tail), cost (manifest +
podcast_obs `llm_cost`), episode/outcome counts (feeds/stats/runs-summary), index/clusters/
coverage freshness, errors (GlitchTip pipeline project + log-tail `error_reason`), traces (OTEL
→ homelab:10428).

**Gaps — decide/fix while the PR is in flight (mapped to the earlier fix brief where relevant):**
1. **Rollback is SSH-only** — no `DELETE /api/corpus/runs/{run_id}`/`episodes/{id}` that also
   re-aggregates the manifest + drops index vectors. *This is the single thing that can strand
   an execution if the operator is away.* Pair with fix-brief #3 (rollback-safe reindex).
   **Highest priority to close for a hands-off run.**
2. **`job_id` ≠ pipeline `run_id`** — `build_pipeline_argv` (`server/jobs.py:191-227`) doesn't
   pass `--run-id <job_id>`, so pivoting Jobs-API → obs (`podcast_obs correlate`) needs scraping
   the run_id from log-tail. One-line fix (pass `--run-id job_id` at enqueue) → single join key.
3. **Pipeline-container logs likely NOT shipped to VictoriaLogs** (PLAUSIBLE, unverified) —
   `infra/observability/operator.alloy:22-25` only matches `compose-api-1`/`compose-viewer-1`,
   not the ephemeral `pipeline-llm` job container. **Verify live on the first real run:**
   `curl -sG http://homelab:9428/select/logsql/query --data-urlencode 'query={app="podcast",surface="pipeline"}'`.
   If empty, either add an Alloy match for the job container (homelab infra repo) or rely on the
   `log-tail` API (which always works). Not a blocker, but decide the story before scaling.
4. **Cost not UI-surfaced** — `cost_rollup` is in `/api/corpus/documents/manifest` JSON but no
   dashboard card renders it. Low-effort card would make cost review click-not-curl.
5. **No cost/episode/GI-KG series in VictoriaMetrics** — `pipeline_run_prometheus.py` emits only
   timing histograms + a job-status counter. Fine for now (cost is on the manifest); note it if
   PromQL cost dashboards are wanted later.
6. **`artifacts.py` + `cil_queries.py` still use latest-run-only** (`latest_feed_run_allowed_relpaths`
   in `search/corpus_scope.py`, deliberately NOT changed by `3c01787d`) for artifact/CIL *serving* —
   a DIFFERENT surface from the index. Carries the same latent "an incremental add hides the older
   run" assumption, but left as-is because rewriting artifact serving was out of scope for the index
   fix. **DECISION for the operator:** converge these onto the same union+dedup rule too, or keep
   artifact/CIL serving on latest-run.

---

## 6. Do we need the SSH key? — the straight answer

**No — for the whole trigger→validate loop.** It runs on the operator API (needs the operator
**key**, provide it) + homelab obs (have it). **Yes — only for rollback** (delete a run dir),
until a rollback API (gap #1) lands. So before "execute the plan": (a) hand the agent the
`X-Operator-Key`, (b) confirm `podcast_prod_operator` SSH is reachable *or* accept that a bad
increment waits for rollback, ideally (c) land gaps #1–#3 in the fixes PR so the run is fully
hands-off and self-correcting.
