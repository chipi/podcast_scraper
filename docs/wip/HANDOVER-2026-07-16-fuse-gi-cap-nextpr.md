# Handover — LLM call-fuse, harden fixes, GI duration-scaled cap (→ next PR)

**Date:** 2026-07-16
**Branch:** `feat/speaker-person-quality` (local only — **the PR #1190 remote branch was merged and
deleted**). The 7 commits below sit on top of the merged main and are the seed of the **next PR**.
**State:** committed, **NOT pushed** (operator will rebase onto the new main + open the next PR).

## The 7 unpushed commits (newest first)

| SHA | What |
| --- | --- |
| `a01377c3` | feat(gi #1191): duration-scaled insight ceiling + recalibrate LLM call-fuse |
| `36a1116a` | fix(resilience): wire the per-episode call-fuse where the single-episode storm runs |
| `751f5c5b` | docs: resolve harden doc-staleness (ADR-110/113 status, index Pending→legal, ADR-113 gap-closed) |
| `7a5d4054` | fix(resilience): install the LLM call-fuse in production + close harden test gaps (#5/#6/#7 tests, #3 CLI flag) |
| `896a4457` | harden(config): openai_api_base env-over-profile footgun fix + continuous profile-construct guards |
| `ea45a792` | test(ci): eliminate all 13 unit-test skips (config key_env, DGX profile keys, 3 test rewrites) |
| `34d7ef47` | fix(pipeline): summarization 9-tuple crash + breaker default-ON stale test |

Everything at `45324004` and below is already in main (merged PR #1190).

## Validated locally (green)

- Full unit suite last full run: **6810 passed, 0 skipped**, mypy clean, pre-commit green on every commit.
- GI suite after the cap change: 359 passed. Fuse suite: 10 passed (incl. cross-thread + per-episode wiring).
- `make docs` (mkdocs strict) exit 0.
- **NOT run since the last commits:** full `make ci` (integration + e2e + coverage + docker stack-test).
  Run it first thing next session.

## PENDING before the next PR is ready

1. **`git fetch origin && git rebase origin/main`** — main moved (PR #1190 + other agents' worktrees
   merged). Expect conflicts in the hot files below.
2. **Full `make ci` green** (never run against the fuse/cap changes).
3. Open the next PR; body must list `Closes #…` for anything it delivers (squash erases commit mentions).
4. **Decide staged-mode fuse headroom** (see caveats) and **add a per-episode `llm_calls` counter**
   to the metrics so the fuse numbers stop being modeled and start being measured.

### Likely rebase-conflict files (also edited by other worktrees)

`src/podcast_scraper/config.py`, `config_constants.py`, `gi/pipeline.py`,
`workflow/metadata_generation.py`, `workflow/orchestration.py`, `workflow/stages/summarization.py`,
`utils/llm_call_fuse.py`, the 3 DGX profile YAMLs, `docs/adr/index.md`, `docs/adr/ADR-110/ADR-113`.

## LLM call-fuse — how it now works (was: only installed in the eval harness)

- **Run fuse** = process-global (`_run_fuse_global` in `utils/llm_call_fuse.py`) so it is visible in the
  summarization/processing **ThreadPoolExecutor workers** (a ContextVar does not propagate into pool
  workers). Installed in `orchestration.run_pipeline` (main thread, before any stage). `_RunFuse` is
  lock-guarded. Concurrency-safe because runs are one-per-process (CLI once; server spawns each run as a
  **subprocess**; `service.py` multi-feed loop is **sequential**) — matches the existing `rss/downloader`
  process-wide-state precedent.
- **Per-episode fuse** installed in `metadata_generation.generate_episode_metadata` — where the
  ~3,500-call storm actually runs (bundled GI evidence → per-pair fallback). GI evidence has **no internal
  thread pool** (sequential-in-thread), so the thread-local `_episode_fuse` ContextVar is visible to every
  tick() during the storm; concurrent episodes on different workers keep independent fuses.

## Fuse + GI-cap numbers (calibrated 2026-07-16) and the model behind them

Measured on the prod-v3 18-ep corpus (median ~47 min, max ~1.6h; 25–37 insights/ep; ≤6 chunks). Per-episode
call count is **bounded by `gi_max_insights`, not duration** (chunks cap at 6, insights cap at the ceiling).

| Duration | insights | calls **bundled** (prod) | calls **staged** (default cfg) |
| --- | --- | --- | --- |
| 1h | 50 | 38 | 256 |
| 2h | 100 | 60 | 424 |
| 4h | 200 | 79 | 443 |

- **GI ceiling:** `GI_MAX_INSIGHTS_CEILING` 50 → **200**; `duration_scaled_max_insights(chars, base)` scales
  the configured `gi_max_insights` (the **base/floor**, default 50) **multiplicatively** — 1x ≤1h, +0.5x per
  30-min unit, 4x at 4h. So default 50 → 1h:50, 2h:100, 3h:150, 4h:200. Multiplicative so a small EXPLICIT
  cap is respected (base=3 stays ~3). Applied once in `gi.pipeline._resolve_insight_specs`. Default stays 50
  (registry parity: `test_the_config_default_is_not_a_trap`).
- **Fuse:** `llm_max_calls_per_episode` 500 → **1500**; `llm_max_calls_per_run` 8000 → **40000**. 1,500 clears
  the worst legit case (staged 4h ≈ 1,650… see caveat) and still trips below the 3,500 storm.

## Caveats / open decisions for next session

- **Staged-mode tightness:** the "staged" evidence numbers assume ~6 quote-candidates/insight (**estimated** —
  final quotes/insight measured ~2.4, candidates aren't logged). A staged 4h/200-insight episode ≈ 1,650 calls,
  slightly above the 1,500 per-episode fuse → could false-trip. **Prod/eval use bundled** (~200), so it's the
  naive-default path that's exposed. Fix by adding a per-episode `llm_calls` counter (the `_EpisodeFuse.calls`
  value is already there — just log it at episode end) so the next run **measures** the real staged multiplier.
- **Eval baselines are now stale for long episodes:** the dynamic cap changes GI behavior vs the flat-50
  baseline the bake-off measured. Re-baseline before trusting long-form eval numbers.
- **#1191** — the *proper* fix (route-and-tag, no pipeline truncation, cut at the UI) is filed as a GH issue.
  This duration-scaled cap is the **interim** stopgap. Design source: `docs/wip/GI_WHAT_TO_SURFACE.md`.

## Harden audit (all 10 findings resolved in these commits)

- **#1** prod fuse (above) · **#2** ADR-110 Accepted · **#3** `--preprocessing-silence-removal` CLI flag · **#4** ADR index `Pending`→legal
- **#5** pyannote MPS→CPU test · **#6** pricing capability-list test (Mistral-12× root cause)
- **#7** `emit_llm_cost_event(response=)` test · **#8** ADR-113 gap-closed · **#9** airgapped-grounder WIP entry
- **#10** KG-segments verified **false positive** (KG uses no segments in prod either).
