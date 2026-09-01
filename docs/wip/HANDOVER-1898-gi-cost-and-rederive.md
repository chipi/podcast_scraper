# Handover — PR #1898 (GI cost, rederive_only, silent-failure sweep)

**Date:** 2026-09-01
**Branch:** `feat/gi-cost-and-rederive-reprocess`
**PR:** <https://github.com/chipi/podcast_scraper/pull/1898>
**Merged:** `e8c6f35e1` (squash of 5 commits onto `a624e7735`), 2026-09-01. Branch deleted.

Everything here came out of watching the 100-episode DGX batch and then re-mining its
logs. **The batch ran `sha-a624e77`; none of these fixes were live for it.** Every number
below is a baseline for the *next* run to be measured against, not a result.

The recurring shape across all of it: **things that succeed while doing the wrong thing.**
Exit 0, WARNING-only, counters that go nowhere. Almost nothing surfaced as an error.

> **Where the numbers come from.** The batch analysis lives in
> [`FAILURE-MODES-2026-08-31-dgx-100-batch.md`](FAILURE-MODES-2026-08-31-dgx-100-batch.md),
> added by this same PR — that is the source for 83/100, the 79.5 median, 95→49, the
> token ceilings and the DGX serving-ratio analysis. Read it for the evidence behind any
> figure here.
> **Warning:** that doc uses its own A/B/C letter scheme which does **not** match §2's
> below (its A3 is cost billing; §2's A3 is the relabel gates). Cite by name, not letter.
> Figures below marked *(session-only)* were measured while working and have **no
> preserved raw data** — treat them as indicative, not reproducible.

---

## 1. Where it stands

| item | state |
|---|---|
| Local `make ci-fast` | green |
| PR checks | 26 pass / 8 skipping / 0 fail — all green before merge |
| CodeQL alert #549 | dismissed false positive + ledger row |
| Merge | **DONE** — squash-merged as `e8c6f35e1` |
| main @ `e8c6f35e1` | Python application -> Stack test in flight; **must be green before deploy** |
| Deploy | **NOT started. Operator-gated.** |

Stack test **does not run on this PR** — it is `workflow_run`-chained off "Python
application" and restricted to `branches: [main, release/2.4, release/2.5, release/2.6]`. The ~24-minute integration
gate fires only *after* the squash lands. Post-merge is where the risk sits.

---

## 2. Everything that was fixed

Twenty-one items (A1–A4, B1–B4, C1–C2, D1–D5, E1–E5, F1–F2). Grouped by what they broke,
not by file.

### A. Reprocessing was impossible

**A1 — `rederive_only` did nothing, and exited 0.**
The stage coerces `transcribe_missing=false`, and the only other exit from
`process_episode_download` was the `transcribe_missing and temp_dir` gate — so no
`ProcessingJob` was ever queued. The run reported success having done nothing.

**A2 — a second no-op hid behind the first.**
The stage also coerces `skip_existing=true`, so without `--reprocess-existing-only` the
work list is built from the *live feed* and every on-disk episode is then skipped
(`Episodes to process: 0 of 0`). Fixing A1 alone still produced nothing.

Verified end-to-end on a real corpus: `transcribe_sec=0.0`, `download_media_sec=0.0`,
`gi_sec=165.0`, insights 64 → 35, content hash changed. Took **three** attempts; the
first two never reached the new code. Unit tests alone were not evidence of wiring.

**A3 — `relabel_only` / `rediarize_only` were unusable on the default cloud profile.**
Two independent gates demanded a Deepgram key they never call, because those stages set
`transcribe_missing=true` purely as a routing trick to reach their interception point.

**A4 — renamed `enrich_only` → `rederive_only`.**
It collided with the corpus-level `enrich` command (topic clusters), so the name implied
rebuilding corpus enrichments when it re-derives one episode's LLM stages. Old name kept
as a deprecated alias, normalised before any logic sees it. Surfaced in CLI help, the MCP
`reenrich` docstring, and `POST /api/jobs?pipeline_stage=` (new, allowlisted — the value
reaches a subprocess argv).

### B. GI was burning GPU time on output that was thrown away

**B1 — the insight ceiling was multiplied by the chunk count.**
The cap is duration-scaled and the chunk count is *also* duration-derived, so duration was
counted twice: 52k chars → 100 effective, 200k → 1200. Production ran at a median of
**79.5** insights against a configured 50 (max 157), then `cleaned[:max_insights]`
discarded the surplus — generated, paid for, dropped. Validated on a real model:
**95 → 49**. Batch-wide GI output ceiling **637,500 → 265,640 tokens (−58%)**.
Per **ADR-135** the merged list is deliberately NOT trimmed; a test rejects that mutation.

**B2 — quote budget was truncating, not the model misbehaving.**
384 → 640 tokens/insight. Over 1345 calls: `completion_tokens p50=989 p75=1310 p90=1902
p99=3840 max=3840`, and **63 calls sat exactly on a ceiling** (1024×11, 1920×18, 2688×1,
3840×33) — i.e. truncated. The p99/max of 3840 is *censored by* the ceiling, so the true
requirement is higher than anything measurable from that window. B2 had been filed as
"Qwen JSON reliability" from 3 samples; across 91 the pipeline's own diagnostic says
**68 `DOCUMENT_ENDED_EARLY` vs 22 malformed**. Truncation.

**B3 — then capped at 5120, because B2's raise introduced an overflow.**
Max prompt 26,714 + 640×10 = 33,114 against a 32,768 window. A regression this branch
created and caught before shipping.

**B4 — prompt de-contradiction** across 14 files: the prompts asked for exhaustiveness and
a hard count simultaneously.

### C. Twenty counters wrote to nowhere

**C1.** `Metrics` is a dataclass, but `finish()` builds an explicit dict literal — so a
counter must be **both** a declared field **and** named there. Twenty were missing one or
both.

Why it was invisible: most are bumped via `_bump(pipeline_metrics, "name")` — a **string
key**, so `setattr` on an undeclared field silently wrote nowhere. Nothing ever errored.
This covered **every value-gate counter** (which is why #1895's drop rate had to be
reconstructed by grepping WARNING lines) and `gi_empty_extraction_count`, which was guarded
by `hasattr` against a field that did not exist — so it never incremented once.

`test_bumped_metrics_are_exported.py` AST-scans bump sites to enforce both bars.

**C2 — value-gate drop audit** (#1895 F2): the gate's *dropped* insights are now recorded
as a list, not a count. `gi.json` keeps only survivors, so without this there is no way to
compare raters on a finished episode — two raters can drop the same number while
disagreeing about which.

### D. Correctness bugs that silently produced wrong artifacts

**D1 — the oldest-run bug was still live in the flat layout.**
The shared dedupe bails without a `feeds/` prefix, so in flat `run_*/metadata/` layouts
first-wins picked the **OLDEST** run. That index drives the transcript glob, so
relabel/rediarize could rewrite a superseded transcript over the current one. Every dedupe
test used `feeds/`, which is exactly why it hid.

**D2 — bare-name scoping validated the KG payload with the GI validator.**
Failed on `GI artifact missing required key: 'model_version'` — a key KG does not have and
should not. The both-or-neither guard then correctly refused the write, the caller logged
"non-fatal, ids left as minted", and **10 episodes kept unscoped person ids.** Right guard,
wrong input, error message blaming the innocent layer.

**D3 — a fabricated circular-import justification.** An inline-import comment asserted a
cycle that does not exist; verified by hoisting and importing the package five ways.

**D4 — every repaired artifact carried fabricated lineage.**
`gi/repair.py` imported `..providers.gil_lineage` — **a module that does not exist** —
under a bare `except Exception`. So the resolver ALWAYS fell back to `cfg.summary_model`
and *never once* resolved real lineage: repaired artifacts were stamped with the summary
model or "unknown". Lineage is what tells two derivations apart, and it was invented.
(#1657 was itself about a fake lineage stamp, which makes this the same bug twice.)

**D5 — the run-timestamp regex matched none of prod's run directories.**
`_RUN_TS_RE` was anchored `^run_(\d{8}-\d{6})`, but prod's dirs are
`run_<uuid>_<YYYYMMDD-HHMMSS>_<hash>` — **all 397 of them**. Every one fell through to the
mtime fallback, so supersession ordering rested entirely on mtime, which the function's own
docstring calls unsafe for a real corpus. Now `^run_(?:.*_)?(\d{8}-\d{6})`.
`run_append_<hash>` still legitimately falls through (it carries no timestamp).

### E. Selection and timeouts

**E1 — `episode_selection=unprocessed`** (new; default `position` unchanged).
`episode_offset` counts feed *positions*, which shift as feeds publish, so a job asking
for 10 delivers fewer whenever the feed grew since last time. `skip_existing` caught the
overlap correctly, but *after* `max_episodes` had already been spent on it. The batch
finished at **83 of 100** (`FAILURE-MODES:565`).

*On the `8,8,8,8,7,7,7` figure often quoted alongside this:* it is a **mid-run snapshot**
of episodes reaching `kg` per feed job, taken while the batch was still running, and
`FAILURE-MODES:264` explicitly calls it a **floor** from a slice it could not prove was
the whole window. It is illustrative of the shape, not a breakdown of the final 83 — do
not try to reconcile it arithmetically.
`episode_offset` is deliberately not redefined: documented positional behaviour (#521)
with an E2E suite (`tests/e2e/test_episode_selection_e2e.py`).

**E2 — per-call chat timeout was the deadline itself** (#1894). Not a useful bound: one
stuck call consumes the entire budget, so the deadline alert could only fire after it had
already cost everything. Now `deadline × 1/3` with a 120s floor.

**E3 — context clamp** (#1893): learns each model's real window from the provider's error
text and re-fits subsequent requests, so the ladder recovers instead of burning a tier.
Reactive by design — the first oversized call to an unseen model still errors once.

**E4 — deepseek fallback could not answer** (#1892): `reasoning_effort: none` added to the
profile. Reasoning models spend `max_tokens` on `reasoning_content` before emitting any
content, so the fallback tier returned empty and the ladder fell through.

**E5 — thread join bound** was 120s/episode, calibrated for API Whisper; measured DGX asr
is p50 496s, p99 1168s. The warning fired on every healthy multi-feed run. Now derived
from `transcription_timeout` with the old constant as a floor.

### F. `gi-repair` targeting

**F1 — the ids-file path did not dedupe.** Every duplicate id was a **paid** re-derivation.
Ids are now normalised once (`dict.fromkeys`, stripped) before the corpus lookup, so the
same episode cannot be billed twice in one invocation.

**F2 — `--episode-ids` selection by identity.** Repair can now target specific episodes
rather than a scan, and *reports ids it could not find* instead of silently repairing
fewer than asked. Covered by `test_gi_repair_targeting.py` and `test_gi_repair_cli_args.py`.

---

## 3. Deploy

**Operator-gated at every step. Triggering is not approving.**

### 3.1 Merge and watch main

1. Approve + **squash**-merge #1898.
2. Watch `Python application` → then `Stack test` (~24 min, main-only).
   Red on main = fix forward in the same pass. Do not deploy over a red main.

### 3.2 Deploy

`deploy-all-prod.yml`, `workflow_dispatch`. Verified inputs:

| input | value |
|---|---|
| `confirm` | literal `DEPLOY_ALL` |
| `image_sha` | explicit `sha-<7>` from the Stack test run summary |
| `skip_control_plane` | `false` unless already on that sha |

**Do not leave `image_sha` empty.** The workflow's own comment records that empty is how
the three surfaces drifted 8 days apart.

**Rollback:** redeploy the prior sha through the same workflow. Recent good runs:
`33296195981` (2026-08-30), `33289033623`, `33280527077` — all `success`.

### 3.3 Smoke — `scripts/ops/post_deploy_smoke.sh`

```bash
# via make (preferred)
PROD_TAILNET_FQDN=<prod-magicdns-fqdn> make smoke-prod

# with an explicit corpus path
PROD_TAILNET_FQDN=<fqdn> SMOKE_CORPUS_PATH=/app/output make smoke-prod

# direct
bash scripts/ops/post_deploy_smoke.sh <tailnet-fqdn> --corpus-path /app/output
```

Six surfaces probed:

1. `GET /api/health` — status ok + core subsystem flags true
2. `GET /api/corpus/episodes` — at least one episode (Library)
3. `GET /api/corpus/digest` — structured 200 (rows may be empty)
4. `GET /api/artifacts` — at least one GI/KG/bridge file (Graph)
5. `GET /api/corpus/topic-clusters` — clusters when the index is built
6. `GET /api/search` — 200, non-5xx

Exit codes: `0` all green · `1` health subsystem false / status not ok **— and also a
corpus-identity mismatch** · `2` corpus surface returned unexpected 4xx/5xx **— and also
usage/argument errors** · `3` 2xx but malformed or empty where data was expected.

**That exit-1 overload matters:** if you use the identity assertions below and the served
corpus is the wrong one, it comes back as exit 1, which looks identical to a health
failure. Read the script's message, not just the code.

**Use the corpus-identity assertions.** A stale corpus passes every subsystem check — this
is what would have caught the #14 bad deploy that reported green while serving the old
corpus:

```bash
EXPECT_CORPUS_PRODUCED_AT=<manifest value> \
EXPECT_CORPUS_CODE_VERSION=<manifest value> \
  bash scripts/ops/post_deploy_smoke.sh <fqdn> --corpus-path /app/output
```

**Before believing any smoke result, confirm the running sha equals the merge commit.**
A green smoke against the old image is the failure mode this project has already had once.

---

## 4. The next 100-episode run

Same 10 feeds. It should use `episode_selection=unprocessed`, or it reproduces the
83-of-100 shortfall that E1 exists to fix.

### 4.0 OPEN — establish how the batch is actually launched

The 2026-08-31 batch's launch mechanism is **not recorded anywhere in the repo**, and the
right place to put `episode_selection` (§4.1) depends on it.

**What the evidence narrows it to.** `FAILURE-MODES:174` investigates feed job
`e259ae92`:

```
Episodes to process: 10 of 298   (episode_offset=10 applied)
  [1], [2]  already on disk -> skipped by --skip-existing
  [3]..[10] processed
```

So: **ten separate per-feed jobs**, each `max_episodes=10` with `episode_offset=10` and
`skip_existing`, and job ids are UUID-shaped (`e259ae92`) — which matches the API's
pipeline-job registry (`routes/jobs.py` types `job_id` as "Pipeline job id (UUID)").

**Where the evidence stops.** The API starts the pipeline *as a subprocess*, so an
API-launched job and a hand-run CLI invocation emit the identical log line. A UUID job id
is consistent with the API path but does not prove it.

**Resolve before the batch** — from the operator, shell history, or the API's job registry
— then record the answer here. Do **not** upgrade the inference above into a conclusion:
doing exactly that produced a bogus "hard blocker" in an earlier draft of this document.

### 4.1 How to set `episode_selection`, once §4.0 is answered

`episode_selection` is a Config field (`config.py:759`), so it is set by any of the normal
routes:

- CLI: `--episode-selection unprocessed`
- Profile / operator YAML: `episode_selection: unprocessed` (profile YAML maps onto Config
  fields — `profile_freeze.example.yaml` sets `max_episodes` the same way)

It is **not** a `POST /api/jobs` query param — that endpoint takes `path, feed,
skip_existing, append, max_episodes, episode_offset, episode_order, profile,
pipeline_stage`. If the batch is launched through that endpoint, set the value in the
profile instead. Adding it as a query param, to match `pipeline_stage`, is a reasonable
follow-up but nothing here depends on it.

### 4.2 What to check while it runs

Live, via the observability MCP:

| what | tool |
|---|---|
| batch progress / per-run state | `prod_recent_runs`, `prod_run_summary` |
| errors as they land | `prod_recent_errors`, `prod_recent_logs` |
| stage timings + `llm_cost` events | `obs_events` (`pipeline_stage`, `llm_cost`) |
| spend so far | `prod_cost_today`, `prod_usage` |
| ladder failovers / breaker state | `prod_resilience` |
| alerts | `prod_recent_alerts` |

**Watch these five specifically, because they are the fixes:**

1. **Insights per episode** — should sit near 50, not 80. If it is back at 80, B1 did not
   take effect (check the profile actually carries the new cap).
2. **`gi_value_gate_*` counters present in `metrics.json`** — if absent, C1 did not deploy.
   Their presence is the cheapest single proof the build is live.
3. **Duplicate-key warnings** — 15,084 before *(session-only total; `FAILURE-MODES` gives
   sampled counts and "90%+ of log volume", not a total)*. Near-zero expected. These come
   from the run-index dedupe path — **D1 and D5** are the fixes.
4. **`DOCUMENT_ENDED_EARLY` count** — 68 before. Near-zero expected; a residue means the
   640/insight budget is still short for some episodes.
5. **Episodes delivered** — should be 100/100, not 83. If it is short, the run used
   positional selection: check `episode_selection` actually reached the config (§4.1).

**DGX health:** verify via the `asr_provider_actual` field — **not** deepgram cost. The
fallback chain falls to `whisper` first, so cost alone cannot prove the DGX served the
request. Note `asr_provider_actual` is a **run-context field**
(`src/podcast_scraper/obs/events.py:102`), not an MCP tool: reach it through `obs_events`.

**Do not disturb the DGX mid-batch.** It is shared; never run `gpu-mode-swap.sh code`
(that is the operator's IDE vLLM, `coder-next`).

### 4.3 Verification table

| signal | before (`sha-a624e77`) | expected after | fix | source |
|---|---|---|---|---|
| duplicate-key warnings | 15,084 | ~0 | D1, D5 | session-only |
| insights / episode (median) | 79.5 (max 157) | ~50 | B1 | FAILURE-MODES:371 |
| GI output ceiling (batch) | 637,500 tok | 265,640 tok | B1 | FAILURE-MODES |
| quote calls exactly on a ceiling | 63 | ~0 | B2 | bundled_prompts.py:79 |
| `DOCUMENT_ENDED_EARLY` | 68 | ~0 | B2 | bundled_prompts.py:81 |
| unscoped person ids | 10 episodes | 0 | D2 | metadata_generation.py |
| episodes delivered | 83 / 100 | 100 / 100 | E1 | FAILURE-MODES:565 |
| value-gate counters in metrics.json | absent | present | C1 | metrics.py |

---

## 5. NOT done / NOT verified

**No CI tier calls a real LLM.** Stack test runs `airgapped_thin`; cloud-thin is
local-only (policy, #1055/#1058). So the list below is not "it hasn't merged yet" — it is
the set of changes whose correctness **no gate before prod can establish**, because they
depend on how a live model behaves. These are what to watch on the first real run.

### Cannot be caught by any pre-prod gate

- **The 95→49 result rests on ONE real episode.** The arithmetic is unit-tested; the
  effect on a live model has a sample size of one.
- **No quality or recall measurement** of the insight reduction. The provider discards by
  *order of emission*, not by value — nothing establishes the dropped ones were the weak
  ones. **This is the single largest unknown in the PR.**
- **The 5120 quote cap** guards a real context overflow that airgapped CI cannot reach, so
  it is untested against the condition it exists for.
- **E3's clamp parses provider error text.** A different vendor message format means it
  silently never learns and the call keeps failing. Only exercised against my fixtures.
- **E4 (`reasoning_effort: none`)** is a profile change entirely about a real vendor's
  behaviour. No gate covers it.
- **E2's 1/3-deadline** is proven *computed*, not proven *sufficient*. Too tight converts a
  slow call into a failed one.

(Everything else — `rederive_only` on a real corpus, D1/D5, D2, E1's logic, the metrics
export wiring — has real coverage; a gate catches those if they regress.)

### Deliberately not done

- **The 1200s deadline is unchanged**, though it fires on 41% of healthy episodes
  (provenance in `timeout_config.py:147`). Left for a deliberate signal-quality decision
  rather than raised to silence it.
- **Residual over-generation is serving-side.** The same Qwen model runs 1.08× on
  OpenRouter vs 1.08–2.52× on our DGX; NVFP4 quantisation is the prime suspect. → #1896.
- **Speaker naming untouched** — 144/281 voices resolve to a person; 42% of episodes have
  >30% of talk time unattributed *(session-only; see #1897 for the figures as filed)*.
  Needs an eval, not a patch.
- **DGX concurrency measured, not implemented**: 1× → 1.74× → 2.55× → 4.16× → 5.93× at
  concurrency 1/2/4/8/16, knee not reached *(session-only — **raw data not preserved**,
  and `FAILURE-MODES` line 312 still says the 4–7× expectation is "unmeasured". Re-measure
  before relying on these.)* The quote queue is strictly serial (`while queue:
  queue.pop()`), no parallelism knob. → #1896.
- **Prod's running sha not re-verified at handover** — the obs MCP in that session pointed
  at `localhost` (connection refused). Treat `sha-a624e77` as stale until re-checked.

---

## 6. Operational gotchas learned here

- **`test-unit` hung 14m10s on `apt-get install ffmpeg`** and hit `timeout-minutes: 15`.
  Downstream jobs skipped, no coverage artifacts uploaded, and `coverage-unified` failed
  with `UNIT= INT= E2E=` — an empty-artifact failure, **not** a coverage-threshold failure.
  Fix is a **full** workflow re-run; `--failed` alone reproduces it, since the producing
  jobs must actually execute. **Nine** jobs in `python-app.yml` do a bare `apt-get update
  && apt-get install ffmpeg` with no retry (viewer-e2e, test-unit, app-e2e,
  preload-ml-models, test-integration, test-integration-fast, test-e2e-fast, test-e2e,
  test-acceptance-fixtures). Standing fragility, untouched.
- **Inline `# codeql[...]` pragmas do not suppress.** Only the API dismissal clears the
  check. #549 is a re-issue of dismissed #503 — the refactor moved the same sink 288 → 291.
  Sanitiser chain: `routes/corpus_rollback` → `resolve_corpus_path_param`
  (normpath + resolve + `startswith(anchor + os.sep)`, raises 400 on escape).
- **`.venv-dev` cannot gate doc changes** — no mkdocs. Use
  `make docs PYTHON=.venv/bin/python`.
- **`make ci-fast` needs `PYTHON=.venv-dev/bin/python`** — under `.venv` the ML extras
  break the no-ML dedupe guards.

---

## 7. Issues

| issue | state |
|---|---|
| #1892 deepseek fallback empty content | closed by this PR |
| #1893 context length exceeded | closed by this PR |
| #1894 no transport timeout | closed by this PR |
| #1891 Qwen ignores insight count | open — capped; root cause serving-side |
| #1895 value-gate eval | open — all 3 blockers closed, now runnable |
| #1896 GI cost programme | open — DGX serving gap is item 1 |
| #1897 speaker naming | open — largest quality gap the batch exposed |

Local decisions (task list, not GH):

- Add `episode_selection` to `POST /api/jobs` to match `pipeline_stage`? — see §4.1.
  Optional consistency follow-up; the CLI and profile routes already work.
- Should the per-feed profile pin be allowlist-validated?
- Tune `GI_INSIGHT_TOKENS_EACH`, now measurable at 84.1 tok/kept-insight.
