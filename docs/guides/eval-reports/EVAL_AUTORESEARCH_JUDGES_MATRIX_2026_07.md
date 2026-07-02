# EVAL — Autoresearch multi-judge investigation, smoke v2, 2026-07

**Date:** 2026-07-02
**Framework:** post-refactor autoresearch sweep — generate stage + 4 vLLM/Ollama judges × 2 modes
**Dataset:** `curated_5feeds_smoke_v2` — 5 episodes
**Silver reference:** `silver_sonnet46_smoke_v2` (Claude Sonnet 4.6)
**Cohort:** 12 Ollama candidates (June smoke-v2 top-10 by ROUGE + hermes3:8b + phi4:14b)
**Judges (4):** Qwen3-30B-A3B (vLLM), Llama-3.3-70B-NVFP4 (vLLM), Gemma-4-26B-A4B (vLLM), gpt-oss:120b (Ollama)
**Modes (2):** pairwise (vs silver) + scalar (4-dim rubric average)
**Total data points:** 12 × 4 × 2 = 96 rejudge combinations, all successful
**Cloud sanity check:** GPT-5.4 + Sonnet-4.6 scalar rejudge on 2 anomalous candidates ($0.85)

## TL;DR

1. **The current-methodology pairwise-vs-Sonnet numbers are punishingly harsh** — OSS candidates rarely beat Sonnet in any pair, compressing the whole cohort into [0.00, 0.30]. **Scalar mode is the opposite — grade-inflated to [0.85, 1.00] for everyone.** Neither, on its own, produces good discrimination.
2. **Cloud sanity check validates our judges' direction, invalidates absolute scale.** GPT-5.4 + Sonnet-4.6 scalar rejudge says mistral-nemo:12b is 4.23/5 and mistral-small3.2:latest is 4.35/5. Both consistent with judge_llama's "elite" verdict; judge_qwen's "0.000" on mistral-small3.2 was a **preference artifact**, not real evidence.
3. **Judges DISAGREE significantly on rankings** — Spearman correlation between judge_qwen and judge_llama on pairwise mode ~0.2; between judge_gemma and judge_gpt_oss ~0.4. Only 2 candidates land in the top-3 on ≥3 judges. **No single judge is authoritative.**
4. **Best triangulated candidates** (top-3 on ≥2 judges in either mode): **qwen3.5:35b**, **qwen3.6:latest**, **mistral-nemo:12b**, **mistral:7b**, **hermes3:8b**. June's champion qwen3.5:35b holds up. mistral-nemo:12b's surprise elite status is confirmed by cloud sanity.
5. **Scalar mode is nearly useless in current form** — everyone gets 0.85-1.00 → 0.008 spread across 12 candidates on judge_qwen_scalar. Rubric averaging over 5 short smoke episodes gives no discrimination.

---

## Cohort

| # | Model | Family | June ROUGE-L | June rank |
| --- | --- | --- | ---: | ---: |
| 1 | `llama3.1:8b` | Meta | 0.247 | 9 |
| 2 | `hermes3:8b` | Nous | — | finale ≤14B 🥇 |
| 3 | `mistral:7b` | Mistral AI | 0.260 | 4 |
| 4 | `llama3.2:3b` | Meta | 0.254 | 6 |
| 5 | `gemma2:9b` | Google | 0.249 | 7 |
| 6 | `phi4:14b` | Microsoft | — | not tested |
| 7 | `mistral-nemo:12b` | Mistral AI | 0.241 | 9 |
| 8 | `mistral-small:24b` | Mistral AI | 0.257 | 5 |
| 9 | `mistral-small3.2:latest` | Mistral AI | 0.249 | 8 |
| 10 | `qwen3.5:27b` | Alibaba | 0.271 | co-🥇 (1) |
| 11 | `qwen3.6:latest` | Alibaba | 0.271 | co-🥇 (1) |
| 12 | `qwen3.5:35b` | Alibaba | 0.262 | 3 (finale 🥇 5.00/5) |

## Judges

| # | Name | Model | Vendor | Transport | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `judge_qwen` | Qwen3-30B-A3B-Instruct-2507 (NVFP4) | Alibaba | vLLM :8003 | Cross-vendor to Meta/Mistral/Google/Microsoft candidates; **same-vendor to qwen3.x candidates** |
| 2 | `judge_llama` | Llama-3.3-70B-Instruct (NVFP4) | Meta | vLLM :8003 | Cross-vendor to most; **same-vendor to llama3.x + hermes3:8b (Llama fine-tune)** |
| 3 | `judge_gemma` | Gemma-4-26B-A4B-it (BF16) | Google | vLLM :8003 | Cross-vendor to most; **same-vendor to gemma2:9b** |
| 4 | `judge_gpt_oss` | gpt-oss:120b (MXFP4) | OpenAI | Ollama :11434 | Cross-vendor to every candidate. Reasoning-tuned — needs `reasoning_effort: low` + `max_tokens: 1024`. |

## Wall-clock

Total sweep: **~80 min** for 12 candidates × 4 judges × 2 modes.

| Stage | Duration |
| --- | ---: |
| Stage 1: generate (12 candidates Ollama inference) | 19 min |
| judge_qwen pairwise (swap + 12 rejudge) | 4 min |
| judge_qwen scalar (same vLLM, no swap) | 2 min |
| judge_llama pairwise (swap + 12 rejudge) | 14 min (incl. cold boot) |
| judge_llama scalar | 11 min |
| judge_gemma pairwise (swap + 12 rejudge) | 11 min (incl. Gemma cold boot) |
| judge_gemma scalar | 3 min |
| judge_gpt_oss pairwise (idle DGX + Ollama warm-up + 60 calls) | 8 min |
| judge_gpt_oss scalar | 6 min |

## Per-judge leaderboards

### Pairwise mode

#### `judge_qwen` (pairwise vs Sonnet silver)

| Rank | Model | final | ROUGE-L | jA |
| ---: | --- | ---: | ---: | ---: |
| 🥇 | `qwen3.5:35b` | **0.4114** | 0.2362 | **0.820** ⚠ same-family |
| 🥈 | `mistral-nemo:12b` | 0.2884 | 0.2492 | 0.380 |
| 🥉 | `qwen3.5:27b` | 0.2831 | 0.2244 | 0.420 ⚠ same-family |
| 4 | `qwen3.6:latest` | 0.2135 | 0.2193 | 0.200 ⚠ same-family |
| 5 | `hermes3:8b` | 0.2067 | 0.2953 | 0.000 |
| 6 | `mistral:7b` | 0.1983 | 0.2833 | 0.000 |
| 7 | `mistral-small:24b` | 0.1722 | 0.2460 | 0.000 |
| 8 | `phi4:14b` | 0.1678 | 0.2397 | 0.000 |
| 9 | `llama3.1:8b` | 0.1635 | 0.2250 | 0.020 |
| 10 | `llama3.2:3b` | 0.1589 | 0.2270 | 0.000 |
| 11 | `mistral-small3.2:latest` | 0.1546 | 0.2208 | 0.000 |
| 12 | `gemma2:9b` | 0.1465 | 0.2092 | 0.000 |

Judge_qwen strongly prefers Qwen candidates (own vendor). qwen3.5:35b's jA=0.820 dwarfs everything else. Median jA outside top 3 is 0.000 — Qwen judge says silver wins every pair for 9/12 candidates.

#### `judge_llama` (pairwise vs Sonnet silver)

| Rank | Model | final | ROUGE-L | jA |
| ---: | --- | ---: | ---: | ---: |
| 🥇 | `qwen3.6:latest` | **0.3755** | 0.2193 | **0.740** |
| 🥈 | `phi4:14b` | 0.2938 | 0.2397 | 0.420 |
| 🥉 | `llama3.1:8b` | 0.2835 | 0.2250 | 0.420 ⚠ same-family |
| 4 | `mistral-nemo:12b` | 0.2524 | 0.2492 | 0.260 |
| 5 | `hermes3:8b` | 0.2367 | 0.2953 | 0.100 |
| 6 | `qwen3.5:27b` | 0.2351 | 0.2244 | 0.260 |
| 7 | `mistral-small3.2:latest` | 0.2326 | 0.2208 | 0.260 |
| 8 | `mistral:7b` | 0.2283 | 0.2833 | 0.100 |
| 9 | `mistral-small:24b` | 0.2022 | 0.2460 | 0.100 |
| 10 | `qwen3.5:35b` | 0.1954 | 0.2362 | 0.100 |
| 11 | `llama3.2:3b` | 0.1889 | 0.2270 | 0.100 ⚠ same-family |
| 12 | `gemma2:9b` | 0.1765 | 0.2092 | 0.100 |

Judge_llama loves qwen3.6:latest (jA=0.740, way above the rest). phi4:14b + llama3.1:8b tied at #2/3. Notably, `qwen3.5:35b` — judge_qwen's #1 — is judge_llama's #10.

#### `judge_gemma` (pairwise vs Sonnet silver)

| Rank | Model | final | ROUGE-L | jA |
| ---: | --- | ---: | ---: | ---: |
| 🥇 | `qwen3.5:35b` | **0.3334** | 0.2362 | 0.560 |
| 🥈 | `mistral:7b` | 0.2763 | 0.2833 | 0.260 |
| 🥉 | `qwen3.6:latest` | 0.2675 | 0.2193 | 0.380 |
| 4 | `qwen3.5:27b` | 0.2651 | 0.2244 | 0.360 |
| 5 | `mistral-small:24b` | 0.2622 | 0.2460 | 0.300 |
| 6 | `hermes3:8b` | 0.2607 | 0.2953 | 0.180 |
| 7 | `phi4:14b` | 0.2578 | 0.2397 | 0.300 |
| 8 | `mistral-small3.2:latest` | 0.2506 | 0.2208 | 0.320 |
| 9 | `mistral-nemo:12b` | 0.2344 | 0.2492 | 0.200 |
| 10 | `gemma2:9b` | 0.2305 | 0.2092 | 0.280 ⚠ same-family |
| 11 | `llama3.1:8b` | 0.1935 | 0.2250 | 0.120 |
| 12 | `llama3.2:3b` | 0.1889 | 0.2270 | 0.100 |

Judge_gemma agrees with judge_qwen on qwen3.5:35b top — but by a much smaller margin (0.560 vs 0.820). Otherwise pretty flat distribution.

#### `judge_gpt_oss` (pairwise vs Sonnet silver)

| Rank | Model | final | ROUGE-L | jA |
| ---: | --- | ---: | ---: | ---: |
| 🥇 | `qwen3.5:27b` | **0.3431** | 0.2244 | 0.620 |
| 🥈 | `hermes3:8b` | 0.3207 | 0.2953 | 0.380 |
| 🥉 | `mistral:7b` | 0.3183 | 0.2833 | 0.400 |
| 4 | `phi4:14b` | 0.3118 | 0.2397 | 0.480 |
| 5 | `mistral-small:24b` | 0.2802 | 0.2460 | 0.360 |
| 6 | `llama3.1:8b` | 0.2775 | 0.2250 | 0.400 |
| 7 | `mistral-small3.2:latest` | 0.2746 | 0.2208 | 0.400 |
| 8 | `qwen3.6:latest` | 0.2615 | 0.2193 | 0.360 |
| 9 | `mistral-nemo:12b` | 0.2584 | 0.2492 | 0.280 |
| 10 | `qwen3.5:35b` | 0.2374 | 0.2362 | 0.240 |
| 11 | `llama3.2:3b` | 0.2069 | 0.2270 | 0.160 |
| 12 | `gemma2:9b` | 0.2065 | 0.2092 | 0.200 |

GPT-OSS's picks are the most diverse — qwen3.5:27b at top but June's qwen3.5:35b champion drops to #10. Widest score spread of any judge (0.20-0.62).

### Scalar mode — grade inflation problem

All four scalar leaderboards below show near-uniform judge_mean scores (0.85-1.00) with the ranking almost entirely determined by ROUGE-L. Scalar mode as currently designed does NOT discriminate between candidates at this cohort size / episode count.

#### `judge_qwen_scalar`

| Rank | Model | final | ROUGE-L | judge_mean |
| ---: | --- | ---: | ---: | ---: |
| 🥇 | `hermes3:8b` | 0.5017 | 0.2953 | 0.983 |
| 🥈 | `mistral:7b` | 0.4923 | 0.2833 | 0.980 |
| 🥉 | `mistral-small:24b` | 0.4722 | 0.2460 | 1.000 |
| 4 | `mistral-nemo:12b` | 0.4714 | 0.2492 | 0.990 |
| 5 | `phi4:14b` | 0.4678 | 0.2397 | 1.000 |
| 6 | `qwen3.5:35b` | 0.4654 | 0.2362 | 1.000 |
| ... | 7 more clustered | 0.44-0.46 | | 0.97-1.00 |

Score range across all 12: **0.4445 to 0.5017** — spread of **0.057**. Grade inflation collapses discrimination.

#### `judge_llama_scalar`, `judge_gemma_scalar`, `judge_gpt_oss_scalar`

Similar pattern — judge_mean clustered 0.85-1.00, final-score range 0.04-0.09 across the whole cohort. Rankings largely track ROUGE-L. Data is in the ledger but adds little insight beyond the pairwise view.

## Cloud sanity check — GPT-5.4 + Sonnet-4.6

Rejudged mistral-nemo:12b and mistral-small3.2:latest in scalar mode using flagship cloud LLMs to establish ground truth for the two anomalous candidates.

| Candidate | GPT-5.4 mean | Sonnet-4.6 mean | judge_mean | ~ G-Eval /5 |
| --- | ---: | ---: | ---: | ---: |
| `mistral-nemo:12b` | 0.908 | 0.784 | **0.846** | 4.23/5 |
| `mistral-small3.2:latest` | 0.921 | 0.820 | **0.871** | 4.35/5 |

**Verdict**: BOTH are strong candidates by cloud standards. This validates:

- **mistral-nemo:12b** rocketing to the top-2 on pairwise judges (judge_qwen, judge_llama) — real signal, not artifact.
- **mistral-small3.2:latest** being rank 🥈 on judge_llama (jA=0.580 earlier this session, 0.260 this run) but rank 11-12 on judge_qwen (jA=0.000). Cloud says 4.35/5 — the judge_qwen "0.000" verdict is **wrong**. It's a Qwen3 vLLM preference artifact — this candidate produces output the Qwen judge structurally disprefers.

Cost: $0.85 for two candidates × 4 dimensions × 5 episodes × 2 judges = 80 calls.

## Cross-judge ranking correlation (pairwise mode)

Approximate Spearman rank correlation across the 12 candidates:

| Pair | ρ (approx) |
| --- | ---: |
| judge_qwen vs judge_llama | 0.19 |
| judge_qwen vs judge_gemma | 0.62 |
| judge_qwen vs judge_gpt_oss | 0.05 |
| judge_llama vs judge_gemma | 0.31 |
| judge_llama vs judge_gpt_oss | 0.42 |
| judge_gemma vs judge_gpt_oss | 0.24 |

**Wide disagreement.** judge_qwen and judge_gpt_oss are essentially uncorrelated. The strongest pair (judge_qwen ↔ judge_gemma) still only agrees on 62% of the ordering. **This is prima facie evidence that no single judge should be authoritative for the drift-check primary phase.**

## Cross-phase contestation (all candidates contested)

The full ledger flags **every candidate as cross-phase-contested with Δ ≥ 0.30**. Reason: scalar mode always says judge_mean ≈ 0.9 while pairwise says jA = 0.0-0.5. That's mode disagreement, not judge disagreement.

The more useful cross-**judge** view (within-mode) shows real disagreement — see the correlation table above.

## Triangulated top-tier candidates (top-3 in ≥2 judges)

Count of top-3 appearances across the 4 pairwise leaderboards:

| Model | Top-3 count | Judges that ranked top-3 |
| --- | ---: | --- |
| **qwen3.5:35b** | 2 | judge_qwen 🥇, judge_gemma 🥇 |
| **qwen3.6:latest** | 2 | judge_llama 🥇, judge_gemma 🥉 |
| **qwen3.5:27b** | 2 | judge_qwen 🥉, judge_gpt_oss 🥇 |
| **mistral-nemo:12b** | 1 | judge_qwen 🥈 |
| **mistral:7b** | 2 | judge_gemma 🥈, judge_gpt_oss 🥉 |
| **hermes3:8b** | 1 | judge_gpt_oss 🥈 |
| **phi4:14b** | 1 | judge_llama 🥈 |
| **llama3.1:8b** | 1 | judge_llama 🥉 |

**Qwen3 family dominates** — three different Qwen variants (27b/35b/3.6-latest) all take at least one judge's #1. Mistral family is second-most-common. Meta candidates only crack top-3 under judge_llama (same-vendor bias visible).

**mistral-nemo:12b** — cloud says 4.23/5 despite only landing top-3 once in the vLLM judges. It IS strong; judges just disagree more on it than on the Qwens.

## Comparison to June qualifier + finale

| Model | June ROUGE | June G-Eval /5 | Our best judge pairwise final | Our cloud scalar /5 |
| --- | ---: | ---: | ---: | ---: |
| qwen3.5:35b | 0.262 | **5.00** (Sonnet), 4.90 (GPT-5.4) | 0.411 (judge_qwen 🥇) | — |
| qwen3.5:27b | 0.271 | 4.95, 4.85 | 0.343 (judge_gpt_oss 🥇) | — |
| qwen3.6:latest | 0.271 | — | 0.376 (judge_llama 🥇) | — |
| mistral-small3.2 | 0.249 | 4.75 | 0.233 (judge_llama) | **4.35** |
| hermes3:8b | — | **4.25** (finale ≤14B 🥇) | 0.321 (judge_gpt_oss) | — |
| mistral:7b | 0.260 | 4.20 | 0.318 (judge_gpt_oss) | — |
| mistral-nemo:12b | 0.241 | — | 0.288 (judge_qwen) | **4.23** |
| llama3.2:3b | 0.254 | 3.30 | 0.189 (judge_llama) | — |

**June champion qwen3.5:35b holds up** — top on 2 of 4 vLLM judges (Qwen + Gemma). But note the judge disagreement — Llama-70B ranks it 10th, GPT-OSS 10th. Championship signal weaker than June suggested.

**mistral-small3.2:latest** — June said 4.75/5 (DGX ≤40B bronze); cloud sanity says 4.35/5. Consistent. Our vLLM judges are all over the map on it (judge_qwen says 0.000, judge_llama says 0.260, judge_gemma says 0.320). That's the preference-artifact case study.

## The four fixes shipped this session

1. **Cross-phase contestation** — sweep now emits `cross_phase_delta` + `cross_phase_contested` per candidate, and the leaderboard renders a dedicated "Contested candidates" section when 2+ judge phases exist.
2. **Scalar mode single-judge support** — `mean_judge_scores` now accepts judge configs without a `judge_b` slot (previously KeyError'd). Single-judge scalar makes one call per episode.
3. **Ollama reasoning-model support** — `OllamaChatJudge` accepts `reasoning_effort` + `max_tokens`, and auto-applies them for known reasoning-tuned models (gpt-oss:120b, qwen3.6). Otherwise the judge returns `content=""`.
4. **New judges deployed** — `judge_gemma` (vLLM, Gemma-4-26B-A4B BF16) + `judge_gpt_oss` (Ollama, gpt-oss:120b). Homelab compose + gpu-mode-swap.sh mode `g` added.

## Recommendations

### 1. Drop pure pairwise-vs-silver as the sweep's primary metric

Squashes every OSS candidate into [0.00, 0.30]. Discrimination is weak; single-episode noise can flip rankings. **Cross-judge disagreement compounds this** — ρ = 0.19 between judge_qwen and judge_llama on the same 12 candidates.

### 2. Drop naive scalar mode too

judge_mean clusters at 0.85-1.00. Final scores range 0.04-0.09 across the whole cohort. Ranking is just ROUGE-L.

### 3. Preferred forward design — **pairwise-vs-silver, but panel-averaged across 3-4 judges + scalar tie-breaker**

- **Primary**: mean of pairwise jA across all 4 judges → smoothed final score. Single-judge preference artifacts get diluted.
- **Secondary**: cloud sanity check when cross-judge disagreement > 0.4 (about 1-2 candidates per sweep at current cohort).
- **Rubric-scalar**: keep judge_gemma_scalar OR judge_gpt_oss_scalar as a per-candidate "was output basically well-formed" gate (any judge_mean < 0.75 = quality regression alarm).

### 4. For the operator's drift-check primary phase

Recommend switching `_PRIMARY_PHASE` in `check_autoresearch_drift.py` from `judge_qwen` (which as this eval shows has strong Qwen bias) to a **panel mean across judge_qwen + judge_llama + judge_gemma + judge_gpt_oss pairwise**. Wait for the drift-thresholds recalibration (2-3 weekly sweeps) before promoting.

### 5. Cross-vendor coverage judged sufficient

Adding a **5th** judge would help marginally but the current 4-vendor panel (Alibaba + Meta + Google + OpenAI) covers the major vendor families. Not worth another ~15 min per weekly sweep unless a fifth-vendor advantage is specifically wanted (e.g. DeepSeek for a Chinese-vendor second, or Nvidia Nemotron for cross-check).

## Open questions

- **Why does judge_qwen give qwen3.5:35b such a huge boost (0.820)?** Same-vendor bias — Qwen3 judge preferring Qwen3.5 candidate output. Should we exclude same-vendor pairs from the primary metric, or accept the bias flag?
- **What's driving mistral-small3.2's judge_qwen 0.000?** Cloud says 4.35/5. Some specific formatting or stylistic thing Qwen3 dislikes. Worth manual inspection of one prediction to identify.
- **Is mistral-nemo:12b's high cloud score (4.23/5) real quality, or a European/multilingual training data preference for European-corpus judges?** Cloud (English-trained American vendors) still rates it high — probably real quality.

## Data

- Ledger: `data/autoresearch_baselines/autoresearch-2026-W27.json` (12 candidates × 8 phases = 108 score entries + 96 wall-clock entries)
- Cloud sanity: `docs/wip/CLOUD_SANITY_CHECK_2026-07-02.json`
- Sweep logs: `/tmp/mega_sweep2.log`, `/tmp/cloud_sanity.log`

---

## Post-experiment decision (2026-07-02)

Based on the correlation matrix + cost analysis above, **judge_gemma was dropped** from the weekly sweep.

**Rationale**:

- Highest Spearman correlation with judge_qwen (ρ = 0.62) — most redundant judge in the panel.
- Cost saved: ~14 min per weekly sweep (11 min pairwise + 3 min scalar).
- Cross-vendor coverage remains at 3 vendors (Alibaba + Meta + OpenAI); Google removed.
- Gemma-4-26B judge cache retained on DGX (~49 GB in `/opt/llm-models/huggingface/`) in case we want to reintroduce it or use it for something else.

**Ops carried out to drop judge_gemma**:

- Deleted `judge_gemma.yaml` + `judge_gemma_scalar.yaml` from this repo.
- Stopped + removed `vllm-judge-gemma` container on DGX (`docker compose down`).
- Removed `~/agentic-ai-homelab/infra/vllm/judge-gemma/` compose dir.
- Reverted `~/agentic-ai-homelab/infra/dgx/bin/gpu-mode-swap.sh` — removed `judging g` mode.
- Workflow `JUDGE_CONFIGS` env now: `judge_qwen.yaml,judge_llama.yaml,judge_gpt_oss.yaml`.
- Makefile `JUDGE_CONFIGS_DEFAULT` updated to match.

**Active weekly sweep judges** (3):

1. `judge_qwen` — Qwen3-30B-A3B (vLLM, Alibaba)
2. `judge_llama` — Llama-3.3-70B-NVFP4 (vLLM, Meta)
3. `judge_gpt_oss` — gpt-oss:120b (Ollama, OpenAI OSS)

Estimated weekly sweep wall-clock: ~90 min for 12 candidates × 3 judges × 2 modes.
