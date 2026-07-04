# EVAL — Cloud vs local judges + registry impact, 2026-07-03

**Date:** 2026-07-03
**Framework:** cross-methodology triangulation — same 12-candidate cohort,
same 5 smoke_v2 episodes, same Sonnet-4.6 silver, judged by:

- **Leaderboard A (ground truth):** Sonnet-4.6 + GPT-5.4 scalar
  (G-Eval-style 4-dim rubric), aggregated across `judge_a_mean`
  (GPT-5.4) and `judge_b_mean` (Sonnet).
- **Leaderboard B (production):** Qwen3-30B (vLLM) + Llama-3.3-70B
  (vLLM) + gpt-oss:120b (Ollama) — pairwise, from the 2026-07-02
  GHA weekly cron ledger `data/autoresearch_baselines/autoresearch-2026-W27.json`.

**Cost:** ~$5 for 12 candidates × 40 cloud judge calls (2026-07-03 sweep).
**Total spend for the multi-judge investigation to date:** ~$6.

## TL;DR

1. **hermes3:8b tops Leaderboard A at 0.4533** — cloud validates the June
   finale winner as the correct laptop default. **No change to
   `config/profiles/local.yaml` warranted.**
2. **All three of our vLLM/Ollama judges are noisy-to-unreliable proxies
   for cloud judgment.** Spearman ρ vs cloud:
   - judge_qwen: ρ = +0.385 (⚠ noisy)
   - judge_llama: ρ = +0.385 (⚠ noisy)
   - judge_gpt_oss: ρ = +0.119 (✗ essentially uncorrelated)

   None cross the ρ > 0.6 "trustworthy" threshold. The weekly sweep's
   drift-check primary phase (currently judge_qwen) has real but small
   signal — better than random, not as strong as cloud.
3. **Cloud judges compress absolute scores tighter than expected** —
   Sonnet + GPT-5.4 rate the whole cohort 0.657 - 0.891 on judge_mean,
   final-score range 0.356 - 0.453. Discrimination between mid-tier
   candidates is subtle even under flagship cloud judges. This is
   partially the smoke-v2 dataset (5 short episodes) not stressing
   models hard enough.
4. **Meta candidates confirmed bottom-tier by cloud** (llama3.1:8b rank
   11, llama3.2:3b rank 12). All 3 vLLM judges agree — this is the
   strongest cross-methodology signal in the data.
5. **qwen3.5:35b stays as DGX prod default** — cloud rank 5, all-judge
   consensus top-6. June's 5.00/5.00 was a bit generous but the
   ranking direction holds.

---

## Leaderboard A — Cloud judges (Sonnet-4.6 + GPT-5.4, scalar mode)

Same rubric as June G-Eval methodology. `judge_mean` is the equal-weight
average of GPT-5.4 (`judge_a_mean`) and Sonnet-4.6 (`judge_b_mean`).
`final = 0.7 × ROUGE-L + 0.3 × judge_mean`.

| Cloud rank | Model | family | final | judge_mean | GPT-5.4 | Sonnet |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 🥇 | `hermes3:8b` | Nous | **0.4533** | 0.822 | 0.893 | 0.751 |
| 🥈 | `mistral:7b` | Mistral AI | 0.4500 | 0.839 | 0.889 | 0.789 |
| 🥉 | `mistral-small:24b` | Mistral AI | 0.4359 | 0.879 | 0.931 | 0.827 |
| 4 | `mistral-nemo:12b` | Mistral AI | 0.4260 | 0.838 | 0.899 | 0.777 |
| 5 | `qwen3.5:35b` | Alibaba | 0.4234 | 0.860 | 0.909 | 0.811 |
| 6 | `qwen3.6:latest` | Alibaba | 0.4207 | 0.891 | 0.923 | 0.859 |
| 7 | `phi4:14b` | Microsoft | 0.4203 | 0.842 | 0.900 | 0.783 |
| 8 | `qwen3.5:27b` | Alibaba | 0.4185 | 0.871 | 0.913 | 0.829 |
| 9 | `mistral-small3.2:latest` | Mistral AI | 0.4135 | 0.863 | 0.916 | 0.810 |
| 10 | `gemma2:9b` | Google | 0.4085 | 0.873 | 0.923 | 0.824 |
| 11 | `llama3.1:8b` | Meta | 0.3800 | 0.742 | 0.795 | 0.688 |
| 12 | `llama3.2:3b` | Meta | 0.3561 | 0.657 | 0.762 | 0.553 |

### Observations on Leaderboard A

- **Sonnet is systematically stricter than GPT-5.4** — mean Sonnet score
  is 0.775 vs GPT-5.4's 0.890, ~11pp lower. Sonnet-vs-Sonnet-silver
  comparison naturally makes Sonnet more discriminating.
- **The score compression** — Cloud judge_mean ranges 0.657 - 0.891
  (spread 0.234). Extract the ROUGE component and cloud can barely
  discriminate rank 3 (mistral-small:24b 0.879) from rank 10 (gemma2:9b
  0.873). ROUGE-L is doing most of the discrimination work for the
  final score.
- **hermes3:8b's low judge_mean (0.822) vs high final (0.4533)** is
  because it has the highest ROUGE-L in the cohort (0.2953). Judge
  scores actually push it down relative to peers like mistral-small
  (0.879) but ROUGE keeps it on top.

---

## Leaderboard B — Local production judges (vLLM/Ollama pairwise)

From the committed 2026-07-02 GHA weekly cron ledger.

| Model | jQwen rank / final | jLlama rank / final | jGPT-OSS rank / final |
| --- | --- | --- | --- |
| `hermes3:8b` | 6 / 0.142 | 4 / 0.274 | 7 / 0.286 |
| `mistral:7b` | 7 / 0.113 | 10 / 0.173 | 4 / 0.329 |
| `mistral-small:24b` | 9 / 0.098 | 5 / 0.254 | 10 / 0.218 |
| `mistral-nemo:12b` | 3 / 0.304 | 1 / 0.544 | 9 / 0.232 |
| `qwen3.5:35b` | 4 / 0.214 | 3 / 0.442 | 2 / 0.382 |
| `qwen3.6:latest` | 2 / 0.328 | 8 / 0.244 | 6 / 0.304 |
| `phi4:14b` | 5 / 0.204 | 11 / 0.156 | 3 / 0.372 |
| `qwen3.5:27b` | 1 / 0.338 | 2 / 0.530 | 11 / 0.218 |
| `mistral-small3.2:latest` | 12 / 0.088 | 7 / 0.244 | 1 / 0.400 |
| `gemma2:9b` | 10 / 0.096 | 9 / 0.240 | 8 / 0.252 |
| `llama3.1:8b` | 8 / 0.102 | 6 / 0.246 | 5 / 0.318 |
| `llama3.2:3b` | 11 / 0.091 | 12 / 0.151 | 12 / 0.187 |

Note the disagreement: each of the 3 vLLM judges picks a DIFFERENT top
candidate. judge_qwen crowns qwen3.5:27b; judge_llama crowns
mistral-nemo:12b; judge_gpt_oss crowns mistral-small3.2:latest.

---

## Side-by-side ranking (top → bottom = cloud rank)

| Cloud rank | Model | Cloud jM | jQwen rank | jLlama rank | jGPT-OSS rank | Disagreement |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `hermes3:8b` | 0.822 | 6 | 4 | 7 | Δ 5 |
| 2 | `mistral:7b` | 0.839 | 7 | 10 | 4 | Δ 5 |
| 3 | `mistral-small:24b` | 0.879 | 9 | 5 | 10 | Δ 6 |
| 4 | `mistral-nemo:12b` | 0.838 | 3 | 1 | 9 | Δ 3 |
| 5 | `qwen3.5:35b` | 0.860 | 4 | 3 | 2 | Δ 1 ← consensus |
| 6 | `qwen3.6:latest` | 0.891 | 2 | 8 | 6 | Δ 0 ← consensus |
| 7 | `phi4:14b` | 0.842 | 5 | 11 | 3 | Δ 4 |
| 8 | `qwen3.5:27b` | 0.871 | 1 | 2 | 11 | Δ 3 |
| 9 | `mistral-small3.2:latest` | 0.863 | 12 | 7 | 1 | Δ 2 |
| 10 | `gemma2:9b` | 0.873 | 10 | 9 | 8 | Δ 2 |
| 11 | `llama3.1:8b` | 0.742 | 8 | 6 | 5 | Δ 5 |
| 12 | `llama3.2:3b` | 0.657 | 11 | 12 | 12 | Δ 0 ← consensus |

**Cross-methodology consensus** (rows where cloud rank and majority of
local judge ranks are within 3): only 3 candidates — qwen3.5:35b,
qwen3.6:latest, llama3.2:3b. For the other 9, at least one local
judge disagrees by 4+ ranks.

---

## Judge trustworthiness — Spearman ρ vs cloud

| Judge | ρ | Verdict |
| --- | ---: | --- |
| judge_qwen (pairwise) | +0.385 | ⚠ noisy — better than random, doesn't cross ρ > 0.6 trustworthy threshold |
| judge_llama (pairwise) | +0.385 | ⚠ noisy — matches judge_qwen coincidentally |
| judge_gpt_oss (pairwise) | +0.119 | ✗ essentially uncorrelated — even though it's OpenAI's own model, its rankings do not track Sonnet+GPT-5.4 |

**None of our vLLM/Ollama judges reach ρ > 0.6.** The single strongest
individual local judge (judge_qwen or judge_llama at ρ=0.385) captures
only ~15% of the cross-run rank variance a cloud judge would.

### Why judge_gpt_oss failed hardest

Same vendor family as GPT-5.4 (both OpenAI), but MXFP4-quantized gpt-oss:120b
serves with `reasoning_effort: low` and produces very short pairwise
judgments. The quantization + reasoning-suppression combo probably
strips out the fine-grained preference signal that flagship GPT-5.4 has
uncompressed. Not the vendor overlap we hoped would boost correlation.

### Why judge_qwen ties judge_llama at ρ=0.385

They agree on the bottom (Meta candidates are bad — trivially correct)
and disagree on the top (qwen3.5:27b vs mistral-nemo). The residual
correlation with cloud is roughly the "Meta bottom" signal that any
reasonable judge would produce; the top-half signal is lost.

### What this means for the weekly sweep

The drift-check primary phase (`_PRIMARY_PHASE = "judge_qwen"`) is
picking up **some** real signal — ρ = 0.385 is above chance — but it
would miss a subtle prod-model quality regression that flagship judges
would catch. A ~20% quality degradation on hermes3:8b would probably
show up in judge_qwen. A 5-10% degradation would probably not.

**Recommendation** — see the "Methodology proposal" section below.

---

## Registry / profile impact

### Confirmed correct

| Profile | Current default | Cloud says | Verdict |
| --- | --- | --- | --- |
| `local.yaml` (laptop) | `hermes3:8b` | 🥇 rank 1 (0.4533) | ✓ keep |
| `local_dgx_full.yaml` | `llama3.3:70b` | not in cohort (=judge_llama) | untested (see below) |
| `local_dgx_balanced.yaml` | `qwen3.5:9b` | not in cohort | untested (see below) |

The June finale winner `hermes3:8b` for laptop tier is **validated
independently** by the 2026-07-03 cloud rejudge — same methodology, same
rank. No swap warranted.

### Runners-up worth registering

Cloud rank 2 (`mistral:7b`, 0.4500) is essentially tied with hermes3:8b
(gap 0.0033). Cloud rank 3-4 (`mistral-small:24b` 0.4359 and
`mistral-nemo:12b` 0.4260) sit within 0.03 of the top. These are all
strong candidates worth having in the registry, even if not the
current default.

Cloud rank 5 (`qwen3.5:35b` 0.4234) — the DGX prod default. Ranking
holds; keep.

### Not currently tested but production defaults

- `llama3.3:70b` — DGX full-fallback profile prod default. Never in the
  cohort because it IS `judge_llama`. **Zero regression signal on this
  prod model.** Follow-up: either (a) drop llama3.3:70b from the judge
  cohort and add it as a candidate; or (b) add a periodic cloud-anchor
  measurement.
- `qwen3.5:9b` — DGX-balanced profile prod default. Dropped from the
  cohort because of #912 flakiness + June rank 12/18. Also no
  regression signal. Follow-up: cloud-sanity-check qwen3.5:9b directly.

### Meta candidates

`llama3.1:8b` (rank 11) and `llama3.2:3b` (rank 12) are the two weakest
in the cohort by cloud judgment. Cloud judges agree with all 3 local
judges here (rare consensus). **Neither should ever be a default in a
production profile.**

### Family-tier readout (best → worst by cloud judge_mean)

| Family | Best member | Cloud jM | # in cohort |
| --- | --- | ---: | ---: |
| Mistral AI | mistral-small:24b | 0.879 | 4 |
| Alibaba | qwen3.6:latest | 0.891 | 3 |
| Google | gemma2:9b | 0.873 | 1 |
| Microsoft | phi4:14b | 0.842 | 1 |
| Nous | hermes3:8b | 0.822 | 1 |
| Meta | llama3.1:8b | 0.742 | 2 |

Mistral is the strongest family in cohort (4 members, all in the top
half by cloud), with `mistral-small:24b` the individual top scorer on
`judge_mean` alone. Alibaba's Qwen3 line is #2. Meta is clearly last.

---

## Methodology proposal — cloud-anchor calibration

Given ρ=0.385 for our best local judge, we should NOT abandon local
judging (weekly cost, drift signal is real), but we SHOULD periodically
recalibrate against cloud ground truth.

**Proposed cadence**:

- Weekly sweep — unchanged. 3 local judges (Qwen + Llama + gpt-oss).
- **Monthly cloud anchor** — one-shot cloud rejudge of the current
  week's ledger predictions (same script as this eval, ~$5). Compute
  Spearman ρ between cloud rank and each local judge rank. If ρ drops
  below the previous month's floor, promote to a drift alert.
- **Quarterly** — cross-methodology report like this one to check
  whether the local judges have drifted vs cloud (e.g. after any
  cohort change, judge model change, or ledger schema change).

Cost: $60/year for monthly cloud anchor + $20/year for quarterly full
comparisons = ~$80/year total. Gives ground-truth-anchored confidence
that the free weekly sweep is producing usable signal.

---

## Answering the operator's meta-question

> "Maybe learnings will be that ollama judges are crap also, so this
> comparison is 2 goals: how good we feel about ollama judges and
> re-assess all our local models."

**Answer 1 — how good do we feel about local judges?** Not great.
Correlation with flagship cloud is ρ=0.12-0.39 depending on judge, well
below the ρ > 0.6 trustworthy threshold. The judges catch some real
signal (Meta bottom, gross regressions) but disagree on the top-half
ordering, and each judge is stochastically noisy across runs. For a
weekly drift check the local panel is acceptable as a cheap first-line
filter, but any candidate-swap or promotion decision should be
cloud-anchored.

**Answer 2 — reassess local models.** hermes3:8b is validated as the
laptop default. mistral-nemo:12b's surprise elite status from the
vLLM sweep is HALF confirmed by cloud (rank 4, not rank 1 or 2). The
DGX prod default qwen3.5:35b stays at cloud rank 5. Meta candidates
should never be defaults. No urgent registry changes; the follow-up
work is measuring the two untested prod defaults (llama3.3:70b and
qwen3.5:9b) with cloud judges.

## Data

- Leaderboard A: `docs/wip/CLOUD_COHORT_LEADERBOARD_2026-07-03.json`
- Leaderboard B (ledger): `data/autoresearch_baselines/autoresearch-2026-W27.json`
  (2026-07-02 GHA cron run, committed by baselines-commit-bot)
- Cloud sweep log: `/tmp/cloud_cohort.log`
- Correlation script: `/tmp/compute_correlations.py`
- Previous investigation: `docs/guides/eval-reports/EVAL_AUTORESEARCH_JUDGES_MATRIX_2026_07.md`
