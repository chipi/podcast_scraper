# Handover 2026-09-29 — deployed, corpus repair running, what is left before the beta

Supersedes the 2026-09-27 handover (same file, renamed). The deploy that document waited for has
happened; everything below is the state after it.

**The goal today is a trustworthy corpus for the first beta customer.** Everything here serves that.
Times are **local (UTC+2)** unless marked `Z`; the prod box logs in UTC.

Measured on prod 2026-09-29. Anything unmeasured is in §9.

---

## 1. Where things are (12:18 local)

| | state |
| --- | --- |
| prod image | `sha-61a1450` on **all 11** podcast-scraper containers — read back with `docker ps`, not inferred from the deploy result |
| pipeline job image | `pipeline-llm:sha-61a1450` — resolved from `/srv/podcast-scraper/.env` `PODCAST_IMAGE_TAG`, the file the job launcher passes as `--env-file` |
| deploy run | `36529805736` (deploy-all-prod, `image_sha=sha-61a1450` explicit) — 7/7 jobs green |
| main CI @ `61a14508d` | 33 success / 7 skipped / 0 failed, incl. `stack-test`, `test-e2e`, `coverage-unified` |
| KG repair queue | **running** — 23 episodes repaired, 0 failed; 1 running + 19 queued |
| `.viewer/jobs.paused` | absent |
| nightly | still disabled — last step in the chain |
| code fixes | committed straight to main on top of `61a14508d` (#50, relabel tool, #2188); **pushed together with B2, then ONE player deploy** — see §6 |

## 2. The KG repair (ADR-156 / #2164) — IN FLIGHT

### 2.1 The population is 86, not 135

Measured by the **newest** `.kg.json` per episode (2002 unique episodes):

```text
provider:NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4   1367
provider:podcast-flash-0731                        549
topic_labels                                        49   <- fabricated from summary bullets
provider:extraction_failed                          37   <- empty KG
```

Earlier figures — 97 / 38 / 135 / "830 sentence-shaped nodes" — counted **files**, including
superseded artifacts in old run dirs. They were wrong. Most of those episodes had already been
re-derived by later runs.

### 2.2 How it is running

- **24 runs, one per feed**, each scoped by `feed=` **and** a work-list. The API accepts one feed per
  job, so 85 scattered episodes need 24 runs. `feed=` is kept deliberately: if a work-list is ever
  dropped again (it was on 2026-09-28, turning an approved 1 into 50), the damage caps at one feed
  instead of 2002 episodes. The operator chose this over the ~3× faster single-run shape.
- All runs are **pre-queued in the prod API's own registry**. It promotes them one at a time. Nothing
  on a workstation drives it; a Claude restart does not stop it.
- Queue concurrency is `max_concurrent_pipeline_jobs = 1`. It is runtime-editable via the corpus
  operator YAML, but concurrent runs on one corpus are **unproven** — the operator chose to keep 1.

**ETA ~20:00 local.** Measured ~7.7 min/episode wall-clock, dominated by run 3 (18 long substack
episodes, 2h20m). The DGX is not the limiter: GPU util 84–96%, `vllm:num_requests_running=2`,
`vllm:num_requests_waiting=0`. The pipeline only ever sends two requests.

### 2.3 Gates, both passed

| gate | episode | before → after |
| --- | --- | --- |
| A (fabricated) | `urn:bbc:podcast:m002vc9m` | `topic_labels` → NVFP4; 8 sentence-fragment topics → 10 real subjects; 0 → 14 entities |
| B (empty) | `699659cb4c238f5dca217595` | `extraction_failed` → NVFP4; 0 → 10 topics, 15 entities |

Gate B proves the 37 `extraction_failed` episodes are repairable: the original failure was
transient (vLLM unreachable at the time), not structural.

### 2.4 Four episodes the API cannot express — do these AFTER the queue drains

Their ids are RFC-4151 `tag:` URIs, which contain a comma, in both `episode_id` and `guid`:

```text
tag:soundcloud,2010:tracks/2135080263                                  rss_feeds.soundcloud.com_c56bfb45
tag:soundcloud,2010:tracks/2306582951                                  rss_feeds.soundcloud.com_c56bfb45
tag:soundcloud,2010:tracks/2293062902                                  rss_feeds.soundcloud.com_9f6bf928
tag:blogger.com,1999:blog-1793063735579568706.post-7556576044575364143 rss_feeds.feedburner.com_2aa371eb
```

The deployed API splits `reprocess_episode_ids` on commas, so these became fragments that matched
nothing. Run `3de5efd7` derived nothing and **reported `succeeded`**; runs 9 and 18 were cancelled
before they could do the same. Fixed in code (§6, #50) but **not deployed**.

**Route:** the CLI with a work-list **file** (one id per line — commas are fine there), same image,
same `prod_dgx_full`, same feed scoping — 3 small runs. Wait for the queue to empty first: a manual
`docker compose run` alongside an API job is the concurrent-run risk the operator declined.

Do **not** use a whole-feed reprocess instead: it re-derives 61 episodes to repair 4, churning 57
healthy artifacts (extraction is sampled, so their labels would shift) and adding them all to the
index debt in §4.

## 3. The relabel mop-up (#2097 / task #44) — gate QUEUED

### 3.1 The generator was gone; it is now committed

`/tmp/step3_all.py` defined which episodes still needed `relabel_only`. `/tmp` on prod was cleared
and it went with it. Rebuilt from committed predicates as **`scripts/audit/relabel_worklist.py`**:
the serving view is `search.corpus_scope.dedupe_metadata_paths_newest_run_per_episode` (the same rule
the migrations select through), and the rules are `kg.speaker_coherence`'s universal checks, called
one by one so each violation is attributable.

### 3.2 The population is 40, not ~109

```text
not_collapsed_one_spkr    26   every attributed quote on one person
speakers_actually_spoke   13   roster names someone absent from the words
one_quote_one_speaker      6   one quote attributed to two people
```

The **canonical class B is gone**: zero episodes hit `no_show_as_speaker`, `no_anonymous_speakers`,
`roles_are_known` or `spoken_by_targets_exist`. The runbook's `['Host']` / org / show-name rosters
were cleared by the earlier batch.

**The 26 are attribution faults, not diarization — measured.** Every one has 2–20 distinct
`speaker_label` values in its own `*.segments.json`, yet all its quotes landed on one person. So
`relabel_only` (which re-runs GI and recomputes SPOKEN_BY) is the right repair and **no GPU is
needed**. This was first asserted the other way, confidently, before measuring; that would have sent
26 cheap repairs to the GPU path.

None of the 40 ids contains a comma — the whole batch can go through the API.

Artifacts on the prod corpus root: `.viewer/jobs/_relabel_worklist.txt` (40 ids) and
`.viewer/jobs/_relabel_classified.json` (per-episode rules, feed, roster).

### 3.3 The gate

Job **`893a1dd8-dfe1-46e7-8931-fba1aa30cadb`**, queue position 20 — runs after the KG queue drains.
`relabel_only`, feed `rss_meduza.io_152318f0` (10 episodes on disk, so a dropped work-list caps at
10), episode `6e67ad5f1458e952ed1d426296b1` — deliberately an attribution-collapse case, to test the
claim in §3.2.

**Acceptance:** re-run `relabel_worklist.py`; that episode must drop out. This is also the first
real timing for `relabel_only` — none exists yet, so the remaining 39 have no ETA until it lands.

## 4. The search index — one FULL rebuild, last

The incremental reindex **refuses to prune on a legitimate shrink**. On gate A:

```text
two-tier index: REFUSING to prune tier=insight — emitted 39 row(s) but 123 are indexed, would delete 84
two-tier index: REFUSING to prune tier=aux     — emitted 118 but 286 indexed, would delete 168
```

That guard (#2158) treats "fewer rows than before" as a partial artifact. Removing fabricated topics
*is* fewer rows, so every repaired episode leaves stale rows derived from the fabricated artifacts,
and search keeps serving them. **A full rebuild is the only thing that clears them.**

`reindex-prod.yml mode=rebuild with_clusters=true` — the same operation as #2097's own step 2, so it
runs **once**, after every repair. Needs operator approval.

Each run also re-pays ~4 min of corpus-wide work regardless of size (measured: gate A result
06:34:23Z → next job 06:38:26Z; gate B 07:00:33Z → 07:04:20Z): `enrich-edges` walks all 2002
episodes and the reindex covers the corpus delta (`changed: 56` to repair one episode).

## 5. Remaining order — do not reorder

1. KG queue drains (~20:00).
2. The 4 comma-id episodes via CLI work-list file (§2.4).
3. Relabel gate lands → verify (§3.3) → queue the other 39.
4. Re-enrich the 10 from #2097 step 3.
5. **Full search rebuild** — approval needed (§4).
6. Re-enable nightly, re-trigger the skipped ingest — **last**.

**Open decision to re-check before spending GPU:** the operator approved the **51 GPU
re-diarizations** from #2097 step 9. That approval predates today's finding that the 26 collapse
cases are attribution, not diarization. Those 51 were not re-measured. Measure their segments before
dispatching (§9).

## 6. Code fixed this session — on main, NOT yet deployed

### #50 — work-list ids containing commas; a zero-match repair reported green

- `server/jobs.py` `normalize_reprocess_episode_ids` — no separator at all; each element is one id,
  a bare `str` is one id (never split, never iterated as characters).
- `server/routes/jobs.py` — `reprocess_episode_ids` is now a **repeated** query param
  (`?reprocess_episode_ids=a&reprocess_episode_ids=b`). The old `str` type also hid a second bug:
  with repeated params FastAPI kept only the **last** value and silently dropped the rest.
- `workflow/worklist_report.py` — `matched_nothing` predicate.
- `cli.py` — a work-list that matched nothing now **exits 1** on both paths. The single-feed path —
  which every API job takes — had **never called `log_worklist_outcome()`**: the outcome line appeared
  0 times across all three repair logs today. Partial-unmatched stays a WARNING + exit 0, exactly as
  #1855 decided.

Tests: `tests/unit/podcast_scraper/cli/test_worklist_zero_match_fails_the_run.py` (new) and
`tests/integration/server/test_jobs_reprocess_worklist_scoping.py` (the old
`test_a_comma_separated_string_is_accepted` inverted). Each new test was run and **failed** before
the fix. After: `254 passed` across those and every neighbouring suite (`test_cli.py`,
`test_multi_feed_budget_halt.py`, `test_selection_gate_end_to_end.py`,
`test_reprocess_worklist_merge.py`, `test_reprocess_episode_ids.py`). flake8 / black / isort / mypy
clean on the touched files.

### #2188 — the prod obs MCP was blind; it now reads URLs the way every other container does

Prod telemetry itself **flows completely** — measured against the stores. The prod obs MCP could
not see it: run inside `player-obs-1`, its own `summary` returned **2 of 13** sources live, with
metrics, logs, traces and errors all failing while a direct query to each backend returned data.

**Cause, measured.** The loader has two paths. `from_env` derives backend URLs from the settings the
platform already renders for its other containers; the YAML path read YAML literals only. Prod sets
`PODCAST_OBS_CONFIG`, so it always took the YAML path, and that YAML had no metrics / traces URL and
pointed logs and grafana at `homelab:9428` / `homelab:3000`. The tailnet ACL **drops** prod ->
`homelab:9428` and `:3000` while allowing `:8428` / `:10428` (tested from a laptop, the prod host and
inside the container). GlitchTip listens on the homelab loopback only.

**Fix — obs now gets its URLs like every other container, rendered at deploy time:**

- `src/podcast_obs/config.py` — one `_platform_read_urls()` used by both paths; a YAML target that
  omits a URL takes the rendered one. A literal still wins, so local configs are unchanged.
- `config/observability.prod.yaml` — **no backend URLs** (a literal would win and re-break it); fills
  the Sentry org and projects; uses a dedicated GlitchTip token.
- `.github/workflows/deploy-player.yml` — builds `PODCAST_OBS_{VICTORIALOGS,VICTORIAMETRICS,GRAFANA,
  SENTRY}_URL` from `infra/observability/vps-observability.endpoints.env` + the tailnet suffix, exactly
  as the Alloy endpoints deploy builds its write URLs, and fails loud on a malformed value. Traces reuse
  the `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` the player api already exports to.
  Deliberately **not** `PODCAST_{LOGS,METRICS}_PUSH_URL`: the app's `dev_push.py` treats those as
  "push directly from this process".
- Grafana / GlitchTip nodes are declared in the deploy step itself: the shared node file is a `*.env`,
  and the pre-commit bare-`.env` guard blocks editing it, so it stays exactly as Alloy uses it.
- `compose/docker-compose.player-public.yml` — declares the vars on `obs` (required: compose passes a
  container only what it names).

**Secrets:** `PODCAST_OBS_SENTRY_TOKEN` reuses the existing `podcast-obs-dev` GlitchTip token (scopes
decoded read-only); `PODCAST_OBS_GRAFANA_TOKEN` is a new Viewer service account (`podcast-obs-reader`).

**Verified before deploy:** the new loader + YAML run inside `player-obs-1` with the deploy-shaped env
returned metrics, logs, traces, errors and alerts all OK — `summary` live **2 -> 7**. The remaining 6
are by decision: 5 need the operator key (B2), `deploys` needs a PAT (declined). The workflow's own
rendering step has not run yet; it first executes on the next player deploy.

Tests: `tests/unit/podcast_obs/test_config.py` — the loader tests fail on the deployed code; seam
tests pin that every needed var is rendered by the deploy and declared on the service.

### Also uncommitted

`scripts/audit/relabel_worklist.py` (§3.1). Two `package-lock.json` changes predate this session and
are not part of any of this.

### Local ML stack vs Homebrew FFmpeg 9 (commit `dddb14619`)

Homebrew moved this Mac to **FFmpeg 9.0.2** (libavutil.61) on 2026-09-29 11:46. `torchcodec` <=0.15
only has cores up to FFmpeg 8, so every macOS `import sentence_transformers` failed ("Could not load
libtorchcodec") and the pre-commit hook could not collect tests. `pyproject.toml` now pins
`torchcodec>=0.16,<0.17` on macOS (the first release with an FFmpeg 9 core) and keeps `<0.15` on Linux,
where CI and prod bring their own FFmpeg and the July teardown crash lives.

**Only this worktree's `.venv` was upgraded.** Every other worktree's venv on this Mac is still broken
the same way until it reinstalls: `.venv/bin/pip install -e ".[dev,ml,llm,search]"`.

## 7. Parked — discuss after the chain (operator's instruction)

These were local tasks in the previous session and may not have survived its restart, so they are
recorded here:

- **Let the index prune a legitimate shrink.** The guard infers "partial" from row counts, which
  cannot tell "the artifact legitimately shrank" from "the build died". If manifests recorded the
  expected per-stage output counts, the indexer could allow a shrink that matches and refuse one that
  does not — and repairs would stop forcing full rebuilds (§4). Direction only; #2158's guard has not
  been read.
- **A work-list size ceiling.** A request-level cap that fails closed would bound blast radius by
  count instead of by feed, so one run could carry all ids safely — removing today's speed-vs-safety
  trade (§2.2).
- **Scope the post-run `enrich-edges`** to the run's episodes instead of all 2002. RFC-118's
  `corpus_delta` already knows what changed; check whether the corpus-wide sweep is load-bearing
  (cross-episode resolution) before narrowing it.
- **A prod MCP path for agents.** The MCP wired into Claude clients is the LOCAL server
  (`local-dev-observability`, `make serve-obs`, `observability.local.yaml` → `localhost:8000`). No
  client is wired to the prod obs MCP, so no agent can verify production through it.

## 8. Operational notes worth not rediscovering

- **SSH**: `ssh -o BatchMode=yes root@prod-podcast` works as of today. The previous handover's note
  that bare `ssh prod-podcast` fails is **out of date**.
- **Operator key inside `compose-api-1`**: `cat /run/secrets/app_operator_api_key`.
- **Cancel route** is `POST /api/jobs/{job_id}/cancel`. `POST /api/jobs/cancel` is a 405.
- **A cancel that returns `status: running` may still have worked.** Check the registry row
  (`.viewer/jobs.jsonl`) and whether a log was written. Pipeline containers are `--rm`, so
  `docker ps -a` shows nothing once one exits — read the log, not the container.
- **The post-run tail is not a hang.** A job container stays up ~4 min after `result: episodes=` —
  that is `enrich-edges` plus `search.reindex`. It holds the queue slot while it runs.
- **Read the process environment, not config text.** `base.alloy` reads
  `coalesce(sys.env("LOGS_WRITE_URL"), "http://localhost:9428/…")`; the literal is a dead fallback.
  `/proc/<pid>/environ` of the running process is the truth.
- **Metrics are readable directly** at `https://vm.<tailnet>.ts.net/api/v1/query` — DCGM
  (`DCGM_FI_DEV_GPU_UTIL`) and vLLM (`vllm:num_requests_running` / `_waiting`,
  `vllm:generation_tokens_total`). `DCGM_FI_DEV_POWER_USAGE` reads ~25 W at 96% util on GB10 — not
  trustworthy.
- **Count episodes, not files.** Multiple run dirs per episode inflate any `rglob("*.kg.json")` count;
  take the newest artifact per `episode_id`.
- **lean-ctx can latch onto another worktree's root** and refuse shell reads here. Native `Read`
  still works.
- **zsh does not word-split** an unquoted `$VAR` — use an array: `F=(a b c); cmd "${F[@]}"`.

## 9. NOT verified — do not treat as settled

- **Whether `relabel_only` actually fixes attribution collapse.** Segment evidence says it should; no
  relabel has run yet. The gate (§3.3) is the test.
- **`one_quote_one_speaker` (6 episodes)** has never been tested against a relabel.
- **The 51 GPU re-diarizations** (#2097 step 9) have not been re-measured against today's finding.
- **The GI artifacts of repaired episodes.** Only KG artifacts were opened; logs report `GI:1(+0f)`
  each time, but no Insight counts were checked.
- **`relabel_only` duration** — unknown, so nothing after ~20:00 has an ETA.
- **Char-offset misalignment**: the corpus-wide `enrich-edges` pass warns
  `N quote(s) have char_start not aligned with the transcript` (105, 67, 63, 57, 51…) on episodes
  these repairs did not touch. Not investigated.
- **Obs token staging** — whether `SENTRY_AUTH_TOKEN`, `PODCAST_OBS_GRAFANA_TOKEN`,
  `PODCAST_OBS_GITHUB_TOKEN` reach the obs service. Unchecked.
- **Who removed `jobs.paused`** — unknown. Its absence is what let today's queue run; nothing
  automated contends (no crontab, no systemd timer, no scheduler env, no schedule-event runs).
- **Carried from the 2026-09-27 handover, not re-verified this session:** the unexplained
  partial-emission mechanism (#2158), the ~107 not-on-any-roster persons (#2096), the 87
  newline-less transcripts (#2098 / #2099 / #2100), and the WEB-tier enrichment passes.
