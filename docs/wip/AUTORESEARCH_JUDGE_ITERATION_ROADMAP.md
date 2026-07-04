# Autoresearch judge iteration — roadmap to ρ > 0.6

**Goal**: raise the correlation between our free/local judge panel and
flagship cloud judges (Sonnet-4.6 + GPT-5.4) from the current
**ρ = 0.12 - 0.39** to **ρ > 0.6** (trustworthy threshold).

**Baseline** (from
`docs/guides/eval-reports/EVAL_AUTORESEARCH_CLOUD_VS_LOCAL_JUDGES_2026_07.md`):

| Judge | ρ vs cloud | Verdict |
| --- | ---: | --- |
| judge_qwen (Qwen3-30B-A3B NVFP4, pairwise) | +0.385 | ⚠ noisy |
| judge_llama (Llama-3.3-70B NVFP4, pairwise) | +0.385 | ⚠ noisy |
| judge_gpt_oss (gpt-oss:120b MXFP4, pairwise) | +0.119 | ✗ unreliable |

**Framework we have available** (built in the 2026-07-02 investigation):

- Sweep script accepts `--judge-configs` = comma-list of yamls
  (`autoresearch_sweep.py`).
- Judge yamls encode provider + prep_cmd (GPU swap) + mode (pairwise or
  scalar) + judge_families (bias-flag column).
- `gpu-mode-swap.sh` handles vLLM container swaps on DGX (sudo-free
  Ollama flush via API).
- Cloud rejudge script (`/tmp/cloud_cohort_rejudge.py`) rejudges any
  cohort via Sonnet + GPT-5.4 for ~$5/cohort.
- Spearman ρ script (`/tmp/compute_correlations.py`) reads a v2 ledger
  + Leaderboard A JSON and computes per-phase correlations.

**Cost model per round**: an experiment adds ~1 hr infra work + ~30-90
min sweep wall-clock + $0 (local) or $5 (if new cloud ground truth
needed). Deployable, testable, comparable in ~half a day per round.

---

## Round 1 — Free mode swap (in progress, this session)

**Change**: add the 3 scalar variants of the existing judges to the
sweep. Each judge runs both pairwise AND scalar back-to-back on the
same vLLM container (scalar yamls carry no prep_cmd — reuse the
container the paired pairwise yaml just brought up).

**Why we think it might help**: yesterday's 4-judge investigation
showed scalar-mode judge_mean clustered at 0.85-1.00 for everyone —
grade inflation. Pairwise mode compresses too (bottom half all at
jA=0.0-0.1). Neither mode alone discriminates well. But COMBINED —
`final = 0.7 × ROUGE + 0.3 × judge_mean` in scalar mode might track
cloud better than pairwise-final since scalar `judge_mean` at least
has some spread (0.65-0.89 across cohort).

**Cost**: $0 (uses existing infrastructure).
**Wall-clock**: sweep with 6 phases ≈ 100 min if run locally; also
extractable from yesterday's mega-sweep log (which had 8 phases
including these 3 scalars, before judge_gemma was dropped).

**Decision gate**:
- Any scalar ρ > 0.5 → promote to weekly workflow (add to
  `JUDGE_CONFIGS` env). Cost: ~14 min extra per weekly sweep.
- All scalar ρ ≤ 0.5 → move to Round 2. Scalars don't fix the
  correlation.

**How we'd act on it**: update
`.github/workflows/autoresearch-eval-nightly.yml`'s `JUDGE_CONFIGS`
env to include the promoted yamls. Timeout stays at 150 min (already
sized for 6 phases per yesterday's mega sweep).

---

## Round 2 — Cross-vendor cohere judge (Command R+)

**Change**: deploy `CohereForAI/c4ai-command-r-plus-08-2024` (or NVFP4
quant) as a 4th judge, `judge_cohere`. Add pairwise + scalar variants.

**Why we think it might help**: Command R+ 104B is explicitly designed
for RAG + evaluation use cases, trained on distinct Cohere-controlled
data. Fully cross-vendor to Alibaba (Qwen), Meta (Llama), OpenAI
(gpt-oss), Google (Gemma dropped), Anthropic (silver). Orthogonal
preference distribution to every existing judge — highest chance of
adding novel signal.

**Cost**: ~1 hr deploy (homelab compose + gpu-mode `judging o` mode +
model download ~60GB NVFP4) + $0 sweep + ~30 min extra per sweep
that includes it.

**Wall-clock**: full-cohort sweep with 8 phases (existing 3 pairwise + 3
scalar + cohere pairwise + cohere scalar) ≈ 130 min. Might need to
bump workflow timeout.

**Prereqs**:
- Verify NVFP4 quant is available on HF. If not, BF16 = ~200 GB → won't fit GB10.
- Add `judging o` mode to `gpu-mode-swap.sh` (same pattern as
  `judging g` we added + removed for Gemma).

**Decision gate**:
- ρ_cohere > 0.5 → add to weekly panel (4 judges).
- 0.4 ≤ ρ_cohere ≤ 0.5 → keep for occasional runs, don't add to weekly
  (marginal signal not worth the ~30 min per sweep).
- ρ_cohere < 0.4 → drop and move to Round 3.

---

## Round 3 — Bigger Qwen (Qwen3-Next-80B-A3B)

**Change**: swap `judge_qwen` served model from
`NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` (30B/3B active) to
`Qwen3-Next-80B-A3B-Instruct` NVFP4 (80B/3B active) if a NVFP4 quant
is available. Same active-param count, larger total knowledge base.

**Why we think it might help**: Qwen3-30B-A3B was released mid-2025;
Qwen3-Next-80B-A3B is the successor (Nov 2025) with bigger total
params and improved instruction-following. Same active-param count
means comparable inference latency (~3B active). Hypothesis: bigger
knowledge base = more discriminating.

**Cost**: model download (~50GB NVFP4) + swap tests. Same-vendor bias
still expected (Alibaba judging Alibaba candidates), so the bias
concern doesn't go away — but the ρ ceiling might improve.

**Wall-clock**: swap-in, no new mode needed (reuses judge-a compose).

**Decision gate**:
- ρ_qwen80 > ρ_qwen30 by ≥ 0.10 → adopt (bigger judge is worth the
  compute cost).
- Otherwise revert to 30B (faster).

---

## Round 4 — Methodology: pairwise BETWEEN CANDIDATES

**Change**: instead of `judge(silver, candidate)`, run
`judge(candidate_A, candidate_B)` head-to-head. Aggregate to
Bradley-Terry / Elo score. Cohort of N candidates → up to N×(N-1)
comparisons per judge (12 candidates = 132 comparisons per judge per
episode); Swiss-tournament pairing caps at ~40 per candidate.

**Why we think it might help**: **current pairwise is
silver-anchored** — every comparison is `(silver, candidate)`, judge
picks winner. Silver is Sonnet-4.6, so any judge with training-data
overlap with Anthropic will lean toward silver stylistically. Our
Qwen/Llama/gpt-oss judges DO have OpenAI+Anthropic-adjacent training,
so silver is biased-favored.

Removing silver from the comparison eliminates the silver-style
bias axis. Cohort candidates get ranked on their relative quality vs
each other, not vs Sonnet.

**Cost**: significant code changes to `pairwise.py` /
`autoresearch_track_a.py`. Estimated 1-2 days of work. Sweep cost
depends on pairing scheme (Swiss ~40 calls/candidate/judge; round-robin
~132).

**Decision gate**:
- If cohort ranking under Bradley-Terry has Spearman ρ > 0.6 vs cloud
  when using the same judge model → mode change fixes the judge.
- If not → the judge model itself is the bottleneck; go back to
  Rounds 2/3.

---

## Round 5 — Ensemble: correlation-weighted rank aggregation

**Change**: rather than picking one primary judge for drift-check,
aggregate ALL judge ranks weighted by their known correlation with
cloud. Formally:

```text
final_score(candidate) = Σ weight_j × score_j(candidate)
weight_j = max(0, ρ_j)  # negative correlations get zeroed out
```

**Why we think it might help**: even individually-noisy judges (ρ ≈ 0.3)
can combine into a strong signal if their noise is uncorrelated.
Wisdom-of-crowds effect.

**Cost**: pure Python (in `check_autoresearch_drift.py` or a new
aggregation script). ~1 day of work. Requires the cloud-anchor
calibration (Round 6 below) to know the weights.

**Decision gate**:
- Ensemble ρ > 0.6 → THIS becomes the drift-check primary. Individual
  judges kept for diagnostic purposes.
- Not moving the needle → problem is in individual judges, not
  aggregation. Go back to Rounds 2/3.

---

## Round 6 — Cloud-anchor calibration cadence

**Change**: monthly cloud rejudge of the current week's ledger,
tracking ρ per judge over time. Alert if ρ drops below its established
floor. Recommended in the 2026-07-03 cloud-vs-local report.

**Cost**: ~$5/month = $60/year. Zero infra work — reuse
`/tmp/cloud_cohort_rejudge.py`.

**Not gated by other rounds** — this is a monitoring investment we
should make regardless of which Round 2-5 experiments succeed. It's
the tripwire that catches judge-quality regressions we couldn't detect
otherwise.

---

## Rough execution order

1. **Round 1** (this session) — extract scalar data from yesterday's
   mega sweep log; compute ρ; decide.
2. **Round 6** (this week) — automate monthly cloud anchor as a cron.
3. **Round 2** (next session, if Round 1 doesn't hit) — deploy
   Command R+.
4. **Round 3** (if 1+2 don't hit) — try Qwen3-Next-80B.
5. **Round 4** (bigger scope, if 1-3 don't hit) — methodology change.
6. **Round 5** — always worth trying once we have ≥ 2 rounds of data
   to weight against.

## Success criterion

Weekly sweep drift-check primary phase (currently `judge_qwen`)
achieves **ρ ≥ 0.6** vs a fresh cloud anchor, and stays there across
3+ monthly re-anchor measurements.

## What "acceptable failure" looks like

If NONE of Rounds 1-5 pushes us above ρ = 0.5, the honest conclusion
is that our local judge panel is a **cheap first-line filter for
gross regressions** (catches Meta-tier drops), but any refined
candidate ranking or promotion decision needs cloud judges. Cost:
~$5/month for cloud anchor. Not free, but modest.

## Data + code pointers

- Cohort: `data/autoresearch_baselines/cohort.yaml`
- Ledger schema: `data/autoresearch_baselines/autoresearch-2026-W27.json` (v2)
- Cloud Leaderboard A: `docs/wip/CLOUD_COHORT_LEADERBOARD_2026-07-03.json`
- Cross-methodology report: `docs/guides/eval-reports/EVAL_AUTORESEARCH_CLOUD_VS_LOCAL_JUDGES_2026_07.md`
- Sweep script: `scripts/baselines/autoresearch_sweep.py`
- Cloud rejudge: `/tmp/cloud_cohort_rejudge.py`
- Correlation script: `/tmp/compute_correlations.py`
- GPU-swap script (DGX): `~/agentic-ai-homelab/infra/dgx/bin/gpu-mode-swap.sh`
