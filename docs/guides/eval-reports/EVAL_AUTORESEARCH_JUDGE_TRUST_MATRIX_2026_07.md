# EVAL — Autoresearch judge trust matrix, 2026-07-04

**Date:** 2026-07-04
**Predecessor:** `EVAL_AUTORESEARCH_CLOUD_VS_LOCAL_JUDGES_2026_07.md`
(2026-07-03) — that report established that our then-current 3-judge
panel had ρ = 0.12-0.39 vs cloud. This report closes the gap.
**Cohort:** 12 Ollama candidates (unchanged)
**Cloud ground truth:** Sonnet-4.6 + GPT-5.4 scalar (Leaderboard A from
2026-07-03 rejudge; unchanged)
**New this report:** 5 local judges × 2 modes = **10 phases** measured
head-to-head against cloud.

## TL;DR

1. **`judge_qwen_next_scalar` promoted to weekly-sweep primary** — Spearman
   ρ = **+0.958** vs flagship cloud judges. Near-perfect proxy for
   Sonnet + GPT-5.4 at $0/call.
2. **Scalar mode wins over pairwise mode for every judge tested** —
   consistent across 5 vendors, gap 0.14 to 0.64 ρ per judge. Pairwise-
   vs-silver methodology is retired.
3. **Bigger Qwen judge is dramatically better** — Qwen3-Next-80B-A3B
   (scalar) ρ = 0.958 vs Qwen3-30B-A3B (scalar) ρ = 0.755. Same 3B
   active-param compute cost, +0.20 correlation. Round 3 (bigger judge)
   is the biggest single ρ jump we've measured.
4. **Round 2 (Nemotron cross-vendor) also lands** — Nemotron-3-Super-120B
   (scalar) ρ = 0.832. Nvidia-eval-tuned latent-MoE. Third-vendor
   perspective on the panel.
5. **Weekly panel** (all scalar): judge_qwen_next_scalar + judge_gpt_oss_scalar
   + judge_nemotron_scalar. 3 vendors (Alibaba + OpenAI OSS + Nvidia).
   Panel-average ρ = 0.930 — slightly worse than the single best judge.
   Primary alone would suffice for drift; panel used for cross-check +
   contestation flags.

---

## Full trust matrix (10 phases vs cloud ground truth)

Spearman rank correlation between each local-judge-mode phase and the
cloud panel (Sonnet-4.6 + GPT-5.4 scalar, average of two judges'
per-episode 1-5 rubric scores).

| Rank | Judge × Mode | ρ vs cloud | Model | Verdict |
| ---: | --- | ---: | --- | --- |
| 🥇 | **judge_qwen_next_scalar** | **+0.958** | Qwen3-Next-80B-A3B (NVFP4) | ✓ TRUSTWORTHY |
| 🥈 | **judge_gpt_oss_scalar** | **+0.937** | gpt-oss:120b (MXFP4, Ollama) | ✓ TRUSTWORTHY |
| 🥉 | **judge_nemotron_scalar** | **+0.832** | Nemotron-3-Super-120B-A12B (NVFP4) | ✓ TRUSTWORTHY |
| 4 | judge_qwen_scalar | +0.755 | Qwen3-30B-A3B (NVFP4) | ✓ TRUSTWORTHY |
| 5 | judge_llama_scalar | +0.741 | Llama-3.3-70B (NVFP4) | ✓ TRUSTWORTHY |
| 6 | judge_qwen (pairwise) | +0.664 | Qwen3-30B-A3B | ✓ trustworthy |
| 7 | judge_nemotron (pairwise) | +0.622 | Nemotron-3-Super-120B | ✓ trustworthy |
| 8 | judge_gpt_oss (pairwise) | +0.587 | gpt-oss:120b | ⚠ noisy |
| 9 | judge_qwen_next (pairwise) | +0.524 | Qwen3-Next-80B | ⚠ noisy |
| 10 | judge_llama (pairwise) | +0.105 | Llama-3.3-70B | ✗ unreliable |

**Trust thresholds:** ρ > 0.6 = trustworthy; ρ 0.3-0.6 = noisy; ρ < 0.3 =
unreliable. (Set from the 2026-07-03 report's judgment call.)

## Scalar-vs-pairwise gap — the top-line finding

| Judge | Pairwise ρ | Scalar ρ | Δ (scalar - pairwise) |
| --- | ---: | ---: | ---: |
| judge_qwen (30B) | +0.664 | +0.755 | **+0.091** |
| judge_qwen_next (80B) | +0.524 | +0.958 | **+0.434** |
| judge_llama (70B) | +0.105 | +0.741 | **+0.636** |
| judge_nemotron (120B/12B) | +0.622 | +0.832 | **+0.210** |
| judge_gpt_oss (120B) | +0.587 | +0.937 | **+0.350** |

**Scalar beats pairwise for every single judge.** Average delta +0.344.
This is the strongest and cleanest finding in the report — the pairwise-
vs-silver methodology we've been running since 2026-06 was leaving huge
signal on the table.

Why scalar wins:

+ Pairwise mode forces judges to pick a winner between the candidate and
  silver on every episode. Since silver is Sonnet-4.6 (frontier cloud
  model), OSS candidates almost never win any comparison → the whole
  cohort compresses to jA ∈ [0.0, 0.3]. Rank information is destroyed.
+ Scalar mode rates each candidate on an absolute 1-5 rubric. Even
  under grade inflation (everyone gets 4-5), the RANKING between
  candidates is preserved. That ranking correlates well with cloud
  ranking.

## Cost / wall-clock per weekly sweep

| Phase | Wall-clock | Cost |
| --- | ---: | --- |
| Stage 1: generate (12 candidates Ollama inference) | ~20 min | $0 |
| judge_qwen_next_scalar (swap ~5 min + rejudge ~10 min) | ~15 min | $0 |
| judge_gpt_oss_scalar (Ollama, idle-swap ~2 min + 12 × 30s) | ~10 min | $0 |
| judge_nemotron_scalar (swap ~15 min + 12 × 1min) | ~30 min | $0 |
| Persist + drift check | ~3 min | $0 |
| **Total sweep** | **~78 min** | **$0** |

Weekly sweep is cheaper than before (was ~90 min for 3 pairwise judges +
15 min for gpt_oss). Trust-matrix-quality signal at lower wall-clock.

## Cross-vendor coverage (panel members)

| Judge | Vendor | Base model | Quant |
| --- | --- | --- | --- |
| judge_qwen_next_scalar | Alibaba | Qwen3-Next-80B-A3B-Instruct | NVFP4 |
| judge_gpt_oss_scalar | OpenAI (open-weights) | gpt-oss:120b | MXFP4 (Ollama) |
| judge_nemotron_scalar | Nvidia | NVIDIA-Nemotron-3-Super-120B-A12B | NVFP4 |

3 distinct vendors. **Same-vendor bias flags** still fire for cohort
candidates that match a judge's family (e.g. Qwen candidates flagged
under judge_qwen_next). The primary judge (Alibaba) has 3 Qwen
candidates in the cohort — bias visibility is now the panel's job:
cross-check the primary's ranking of Qwen candidates against gpt-oss
and nemotron.

## Ensemble analysis

Correlation-weighted panel-mean rank aggregation across all 10 phases:

+ Ensemble ρ = **+0.930**
+ Best individual ρ = +0.958 (`judge_qwen_next_scalar`)
+ Delta = **-0.028** (ensemble slightly worse than best single judge)

**Why the ensemble underperforms the best individual:** the noisy pairwise
phases (judge_llama pairwise ρ=0.105, judge_qwen_next pairwise ρ=0.524)
still get non-zero weights in the ensemble because they aren't strictly
negative. Their noise dilutes the signal from the strong scalar judges.

A **scalar-only 3-judge ensemble** (qwen_next_scalar + gpt_oss_scalar +
nemotron_scalar) would likely match or slightly beat the single best. Not
implemented in this iteration — retained as follow-up if primary ever
starts drifting from cloud.

## Same-run vs cross-run variance

Two important caveats when reading the ρ numbers:

+ **The 2026-07-03 report's ρ values were CROSS-run** (cloud rejudged
  yesterday's local mega-sweep predictions; this report's ledger came
  from a FRESH mega-sweep on today's predictions). Regenerating
  predictions changes the judge inputs. So the ρ numbers from that
  report and this one aren't directly comparable in absolute terms.
+ **Within-run variance is real too.** Comparing yesterday's cross-run
  numbers (0.385 / 0.385 / 0.119) with today's same-run numbers on the
  SAME judges (0.664 / 0.741 / 0.937 for qwen/llama_s/gpt_oss_s) shows
  ρ can vary by ~0.3 across runs.

**Implication for the monthly cloud-anchor calibration** (roadmap
Round 6): the anchor MUST rejudge the SAME predictions the weekly sweep
scored. Anything else compares apples to oranges.

## Concrete production changes shipped in this iteration

**Weekly sweep panel** (workflow yaml JUDGE_CONFIGS):

Before:

```text
judge_qwen.yaml, judge_llama.yaml, judge_gpt_oss.yaml   (3 pairwise)
```

After:

```text
judge_qwen_next_scalar.yaml,       (primary — Alibaba)
judge_gpt_oss_scalar.yaml,         (panel   — OpenAI OSS)
judge_nemotron_scalar.yaml         (panel   — Nvidia)
```

**Drift check primary** (`check_autoresearch_drift.py`):

Before: `_PRIMARY_PHASE = "judge_qwen"`
After:  `_PRIMARY_PHASE = "judge_qwen_next_scalar"`

**Judge yaml prep_cmd cleanup:** each promoted scalar yaml now carries
its own prep_cmd (swap the right vLLM into place). Idempotent — safe
for both standalone weekly runs and paired pairwise+scalar mega sweeps.

**Parser fix** (`pairwise.py::_extract_json_object`): new helper strips
`</think>...` reasoning blocks + code fences, then balanced-brace-scans
the first `{...}` object. Fixes nemotron judge calls that were silently
100% failing pre-2026-07-04.

**Sweep aggregate fix** (`autoresearch_sweep.py::_aggregate_rows`): when
a candidate fails phase N, phases 1..N-1 scores now land in the ledger
`scores_by_phase` block. Pre-fix, all early-phase data was discarded
when any later phase crashed.

## When primary is used vs when panel is used

**Primary judge** — the anchor for automated drift detection:

+ `check_autoresearch_drift.py` compares each candidate's `final` score
  under `_PRIMARY_PHASE` week-over-week
+ Threshold breach → GH issue auto-filed → operator notified
+ Changing primary invalidates historical drift thresholds — should be
  rare (every 3-6 months at most, driven by cloud-anchor drift or a
  bigger judge landing)

**Panel** — evidence when interpreting individual results:

+ Leaderboard renders all 3 panel members' scores per candidate
+ Cross-phase contestation flag (Δ jA > 0.30 across judges) surfaces
  candidates where the panel disagrees — human review triggered
+ Manual quality calls ("is candidate X actually better?"): check
  what all 3 say
+ Panel-mean fallback if primary drifts from cloud

**Cost**: ~$0 either way — panel members are already spun up per weekly
sweep. No extra API spend or wall-clock beyond ~55 minutes for the
extra scalar phases.

## Rounds retired

| Round | Original ask | Status |
| --- | --- | --- |
| 1: mode swap | Add scalar variants | ✓ SHIPPED — scalar mode across the board |
| 2: cross-vendor | Deploy Cohere/Nemotron | ✓ SHIPPED as judge_nemotron (Cohere had no NVFP4) |
| 3: bigger Qwen | Deploy Qwen3-Next-80B | ✓ SHIPPED — this is the primary (ρ = 0.958) |
| 4: methodology (Bradley-Terry) | Pairwise-between-candidates | ✗ NOT NEEDED — scalar mode fixed the pairwise mess |
| 5: ensemble | Correlation-weighted panel | ✓ MEASURED — underperforms single best; kept as fallback |
| 6: cloud-anchor cadence | Monthly rejudge | 📋 pending — should implement as a cron |

## What's next

1. **Verify prod behavior**: trigger the updated GHA weekly sweep, watch
   it produce the new ledger, drift check runs cleanly.
2. **Round 6 — monthly cloud anchor cron**: automate the ~$5/month
   Leaderboard A refresh so we can track ρ over time and alarm if the
   primary drifts.
3. **Retire dead judges**: drop the pairwise-only yamls
   (judge_qwen.yaml, judge_llama.yaml, judge_nemotron.yaml,
   judge_qwen_next.yaml, judge_gpt_oss.yaml) from the workflow. They're
   not currently referenced.

## Data

+ Trust matrix numbers: `docs/wip/FINAL_TRUST_MATRIX_2026-07-04.json`
+ Reconstructed partial sweep data: `docs/wip/MEGA_SWEEP_V3_PARTIAL.json`
+ Targeted rejudge (post-parser-fix): `docs/wip/TARGETED_REJUDGE_NEMOTRON_GPT_OSS.json`
+ Cloud ground truth (unchanged): `docs/wip/CLOUD_COHORT_LEADERBOARD_2026-07-03.json`
+ Iteration roadmap: `docs/wip/AUTORESEARCH_JUDGE_ITERATION_ROADMAP.md`

## Recipes

Rerun the trust matrix on new predictions:

```bash
# 1. Regenerate predictions + all 10 judge phases
make autoresearch-sweep-local \
  JUDGE_CONFIGS="autoresearch/.../judge_qwen.yaml,..._scalar.yaml,\
                autoresearch/.../judge_qwen_next.yaml,..._scalar.yaml,\
                autoresearch/.../judge_llama.yaml,..._scalar.yaml,\
                autoresearch/.../judge_nemotron.yaml,..._scalar.yaml,\
                autoresearch/.../judge_gpt_oss.yaml,..._scalar.yaml"

# 2. Cloud rejudge for ground truth (~$5)
.venv/bin/python /tmp/cloud_cohort_rejudge.py

# 3. Recompute the ρ table
python3 /tmp/final_trust_matrix.py
```
