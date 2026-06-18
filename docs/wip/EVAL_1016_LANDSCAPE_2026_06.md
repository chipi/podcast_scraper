# #1016 LLM Landscape Per-Stage Eval — 2026-06-16

**Status:** in progress (work doc; promoted to `docs/guides/eval-reports/` on completion)
**Branch:** `feat/928-reframe-llm-landscape-2026-06-16`
**Parent issue:** #1016 (Phase 2 of #928 reframe)

## What this eval answers

Per the #928 reframe: with the autoresearch vLLM stack live on `vllm:26.05-py3` (Qwen3-30B-A3B-Instruct-2507 hot path), what's the right model to pin per stage (summary / GI / KG)?

**Original Phase 1 cohort (7 candidates)** — picked June 2026 from the open-weight landscape:

1. **ollama_qwen35_35b** — prod incumbent, Ollama-served qwen3.5:35b
2. **vllm_qwen3_30b_a3b_2507** — Qwen3-30B-A3B-Instruct-2507 (current autoresearch default)
3. **vllm_qwen3_5_35b_a3b** — Qwen3.5-35B-A3B (heavier challenger)
4. **vllm_r1distill_32b** — DeepSeek-R1-Distill-Qwen-32B (#961 anti-think prompt + post-processor)
5. **vllm_magistral_small_2509** — Mistral Magistral Small 2509
6. **vllm_mistral_small_3_2_24b** — Mistral Small 3.2 24B Instruct
7. **gemini25_flash_lite** — Google Gemini 2.5 Flash Lite (cloud, included for cross-vendor reality check)

**Phase 2c-overnight expansion (9 NEW candidates)** — added 2026-06-17 after a fresh deep
audit of all major vendors' HF catalogs found multiple candidates Phase 1 missed
(see §Phase-1-landscape-miss-audit below). Adding them so the per-stage answer
reflects the actual current state-of-the-art, not the June-2026 snapshot:

8. **Mistral-Small-4-119B-NVFP4** (70.8 GB) — successor to Mistral Small 3.x
9. **Ministral-3-14B-Instruct-2512** (31.5 GB) — speed-king Mistral candidate
10. **Gemma-4-26B-A4B-it** (51.6 GB) — Google A4B MoE, direct Qwen3.5-A3B rival
11. **Llama-3.3-70B-Instruct-NVFP4** (RedHatAI, 42.7 GB) — Meta flagship dense at NVFP4
12. **DeepSeek-V2-Lite-Chat** (31.4 GB) — older non-reasoning DeepSeek MoE
13. **DeepSeek-R1-0528-Qwen3-8B** (16.4 GB) — newer R1 distill on Qwen3 base
14. **Kimi-Linear-48B-A3B-Instruct** (98.2 GB) — Moonshot A3B MoE + linear attention
15. **Moonlight-16B-A3B-Instruct** (31.9 GB) — Moonshot older small A3B MoE
16. **Nemotron-Super-49B-v1_5-FP8** (52 GB) — NVIDIA-tuned Llama-3.3 derivative
    (closely related to the DGX-Spark blog's reference baseline)

**Original Ollama + cloud rows** (Gemini, Ollama qwen3.5:35b apples) retained
for runtime comparison. **R1-Distill-32B dropped** from Phase 2c after Phase 2b
speed-disqualified it (204s mean, 4.612 G-Eval — bottom on both axes).

**Total Phase 2c-overnight cohort: 15 vLLM + 1 cloud + 1 Ollama = 17 cells per stage**
(R1-Distill cell carried forward from Phase 2b for completeness; not refreshed).

Round 1 is **safe BF16/FP8/NVFP4** for apples-to-apples across the natural
quantization each model ships in. Round 2 (separate report, gated on results)
would target NVFP4 for the winners that ship BF16-only.

## Phase-1-landscape-miss audit (2026-06-17)

The original Phase 1 selection (June 2026) missed several strong candidates
that were available at that time or shortly after. The deep audit pulled HF
catalogs for **6 vendors** (DeepSeek, Mistral, Google, Meta, Moonshot/Kimi,
NVIDIA) and found 9 GB10-fitting candidates worth eval. Misses categorized:

- **Vendor coverage gaps**: Phase 1 had no Google Gemma, no Meta Llama, no
  Moonshot Kimi, no NVIDIA-tuned model. The expanded cohort fixes all four.
- **Generation gap**: Phase 1 picked Mistral-Small-3.2 (mid-2025) when
  Mistral-Small-4-119B-NVFP4 (Jan 2026) was available. Similar generation
  miss on Ministral-3 (Oct 2025) and DeepSeek-R1-0528-Qwen3-8B (May 2025).
- **MoE shape coverage**: Phase 1 had only Qwen A3B MoE. Added Gemma-4-A4B,
  Kimi-Linear-48B-A3B, Moonlight-16B-A3B, DeepSeek-V2-Lite-A2.4B for
  cross-vendor MoE-with-small-active comparison.
- **Quantization shape coverage**: Phase 1 was BF16-only. Added NVFP4 (Mistral
  Small 4, Llama 3.3-70B), FP8 (Nemotron, Ministral), Q4 (Ollama for runtime
  reference). Lets the eval distinguish model-quality from quant-effect.

The miss-audit is itself a #928-reframe deliverable: the autoresearch
landscape-review cadence needs to be quarterly, not annual, given the
release velocity in this segment.

## Methodology

**Dataset:** `curated_5feeds_dev_v1` (10 episodes; disjoint from `curated_5feeds_benchmark_v2` held-out).

**Silvers (cross-vendor — see autoresearch/JUDGING.md §Silver/judge vendor-bias rule):**
- Primary: `silver_opus47_dev_v1_paragraph` (Claude Opus 4.7) — 10 eps
- Comparison: `silver_sonnet46_dev_v1_paragraph` (Claude Sonnet 4.6) — 10 eps
- GI silvers: `silver_opus47_gi_dev_v1`, `silver_sonnet46_gi_dev_v1` — both 10 eps
- KG silvers: `silver_opus47_kg_dev_v1`, `silver_sonnet46_kg_dev_v1` — both 10 eps

The dual silver lets us **disclose vendor bias empirically** rather than choose a side. Significant divergence (>5 ROUGE-L points) between the two silvers' rankings indicates the candidate is mimicry-style rather than genuinely better.

**Judge panel:**
- Primary: Sonnet 4.6 G-Eval rubric on all 10 episodes
- Cross-check: GPT-5.4 G-Eval rubric on all 10 episodes
- Gemini 2.5 Pro intentionally omitted (Gemini-Flash-Lite is a candidate; same family).

**Cost cap:** $25/stage = $75 total for Phase 2 judging.

## Phase 2a — Predictions

All 7 candidates have 10/10 summary predictions on disk. vLLM-backed candidates were run sequentially via the homelab autoresearch compose swap dance (one cold boot per candidate, ~10-15 min each). Generation cost: ~$0 marginal for the 5 vLLM candidates, ~$0.20 for Gemini-Flash-Lite. Ollama qwen35:35b uses cached output from a prior dev_v1 run.

| candidate | predictions path |
|-----------|------------------|
| ollama_qwen35_35b | `data/eval/runs/autoresearch_prompt_ollama_qwen35_35b_dev_paragraph_v2/predictions.jsonl` |
| vllm_qwen3_30b_a3b_2507 | `data/eval/runs/autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_paragraph_v1/predictions.jsonl` |
| vllm_qwen3_5_35b_a3b | `data/eval/runs/autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_paragraph_v1/predictions.jsonl` |
| vllm_r1distill_32b | `data/eval/runs/autoresearch_prompt_vllm_r1distill_32b_dev_paragraph_v1/predictions.jsonl` |
| vllm_magistral_small_2509 | `data/eval/runs/autoresearch_prompt_vllm_magistral_small_2509_dev_paragraph_v1/predictions.jsonl` |
| vllm_mistral_small_3_2_24b | `data/eval/runs/autoresearch_prompt_vllm_mistral_small_3_2_24b_dev_paragraph_v1/predictions.jsonl` |
| gemini25_flash_lite | `data/eval/runs/autoresearch_prompt_gemini25_flash_lite_dev_paragraph_v2/predictions.jsonl` |

## Phase 2b — Summary scoring

### Dual-silver ROUGE-L disclosure

The Sonnet-mimicry artifact reproduces: every candidate scores higher against Opus than Sonnet, but the deltas are uneven. Largest jumps:

| candidate | vs Sonnet 4.6 | vs Opus 4.7 | Δ (Opus−Sonnet) |
|-----------|--------------:|------------:|----------------:|
| ollama_qwen35_35b | 0.3029 | 0.3405 | +0.038 |
| gemini25_flash_lite | 0.2927 | 0.3385 | +0.046 ← largest |
| vllm_mistral_small_3_2_24b | 0.2817 | 0.3049 | +0.023 |
| vllm_magistral_small_2509 | 0.2664 | 0.2865 | +0.020 |
| vllm_qwen3_5_35b_a3b | 0.2596 | 0.2578 | −0.002 |
| vllm_r1distill_32b | 0.2161 | 0.2293 | +0.013 |
| vllm_qwen3_30b_a3b_2507 | 0.2064 | 0.2153 | +0.009 |

Interpretation: Gemini-Flash-Lite was being held down by Sonnet-mimicry silver (largest gain when switching silver vendor); against the Opus silver it's essentially tied with the prod incumbent qwen35:35b (0.3385 vs 0.3405). Qwen3.5-35B-A3B is the one model that LOSES slightly on Opus — strong evidence that its training distribution leans Sonnet.

### G-Eval scores (multi-vendor judge panel) — 6-candidate Round 1 cohort

Sonnet 4.6 primary + GPT-5.4 cross-check, **6 candidates** × 10 episodes,
$6.15 spend (cap $25). No contested rows; judge agreement 0.90–1.00.

**Cohort scope**: ONLY this-session-controlled Round 1 runs — 5 vLLM BF16
candidates + 1 cloud Gemini. The prior-session `ollama_qwen35_35b_dev_v2`
row that initially leaked into the table has been removed — its prompts,
cleaning profile, and runtime conditions don't match the Round 1 setup, so
cross-mixing isn't safe. A clean prod-vs-cohort comparison requires either
re-running prod under Phase 2b conditions or accepting the gap; see
**Methodological gap** below.

| Rank | Candidate | Sonnet 4.6 | GPT-5.4 | Mean | Agreement |
|-----:|-----------|-----------:|--------:|-----:|----------:|
| 1 | **vllm_mistral_small_3_2_24b** | 4.625 | 4.950 | **4.787** | 0.95 |
| 2 | gemini25_flash_lite | 4.575 | 4.900 | 4.738 | 0.93 |
| 3 | vllm_qwen3_5_35b_a3b | 4.700 | 4.725 | 4.713 | 1.00 |
| 4 | vllm_magistral_small_2509 | 4.525 | 4.850 | 4.688 | 0.90 |
| 5 | vllm_qwen3_30b_a3b_2507 (current autoresearch hot path) | 4.625 | 4.650 | 4.638 | 1.00 |
| 6 | vllm_r1distill_32b | 4.475 | 4.750 | 4.612 | 0.93 |

Per-judge bias signature is visible: Sonnet primary ranks Qwen family highest
(qwen3.5_35b on top); GPT-5.4 cross-check ranks Mistral family highest
(Mistral Small 3.2 → Magistral → R1-Distill on top). The multi-judge mean
mostly neutralises this, but the Sonnet preference for Qwen partially
reproduces the silver-mimicry pattern.

Tight 0.18-point spread across all 7 candidates — every model is in the
"good enough" envelope. Decision should weight cost + reliability heavily
(quality differences are noise-floor proximate).

### Speed signal — per-episode latency (measured from each run.log)

Quality alone is not enough — prod has to pay every episode's wall-clock cost.

| Rank | Candidate | mean | p50 | p99 | comment |
|-----:|-----------|-----:|----:|----:|---------|
| 1 | gemini25_flash_lite | **4.6s** | 2.5s | 25.4s | cloud |
| 2 | ollama_qwen35_35b (prod) | 19.2s | 7.9s | 105.6s | DGX, Ollama |
| 3 | vllm_qwen3_5_35b_a3b | 39.5s | 21.7s | 217.3s | DGX, vLLM |
| 4 | vllm_qwen3_30b_a3b_2507 | 46.6s | 25.5s | 256.3s | DGX, vLLM (current autoresearch default) |
| 5 | vllm_magistral_small_2509 | 117.8s | 66.2s | 647.7s | DGX, vLLM |
| 6 | vllm_mistral_small_3_2_24b | **132.0s** | 72.7s | 726.0s | DGX, vLLM |
| 7 | vllm_r1distill_32b | 371.2s | 188.5s | **2041.9s** | DGX, vLLM, reasoning |

Gemini-Flash-Lite is **7× faster than the best DGX option (ollama qwen35:35b)
and 28× faster than the top-quality model (Mistral Small 3.2)**. The
top-quality DGX model's p99 = 12 minutes for one episode — a single outlier
dominates a batch.

### Composite score — G-Eval rank + speed rank

Lower sum = better composite (Borda-style).

| Rank | Candidate | G-Rank | Speed-Rank | Sum |
|-----:|-----------|-------:|-----------:|----:|
| **1** | **gemini25_flash_lite** | 2 | 1 | **3** |
| 2t | vllm_qwen3_5_35b_a3b | 3 | 3 | 6 |
| 2t | **ollama_qwen35_35b (prod)** | 4 | 2 | 6 |
| 4 | vllm_mistral_small_3_2_24b | 1 | 6 | 7 |
| 5t | vllm_magistral_small_2509 | 5 | 5 | 10 |
| 5t | vllm_qwen3_30b_a3b_2507 | 6 | 4 | 10 |
| 7 | vllm_r1distill_32b | 7 | 7 | 14 |

### Per-stage decision — summary

Framing: prior research already settled **vLLM = prod runtime, Ollama =
autoresearch tooling slot**. Phase 2b therefore answers two questions, NOT
"should prod swap from Ollama to something else":
1. Within the vLLM-served cohort, which BF16 model wins for summary?
2. How close is the cloud fallback (Gemini-Flash-Lite), useful as a
   DGX-contention safety valve (#1000 axis)?

Restricted to vLLM (the prod runtime per prior research):

| Rank | vLLM candidate | Speed (mean) | G-Eval | Verdict |
|-----:|----------------|-------------:|-------:|---------|
| 1 | **vllm_qwen3_5_35b_a3b** | 39.5s | 4.713 | **Prod-default summary recommendation (BF16)** |
| 2 | vllm_mistral_small_3_2_24b | 132s | 4.787 | Top quality but BF16 too slow; Round 2 FP8 candidate |
| 3 | vllm_qwen3_30b_a3b_2507 | 46.6s | 4.638 | Dominated by qwen3.5-35B-A3B → **retire from autoresearch default** |
| 4 | vllm_magistral_small_2509 | 117.8s | 4.688 | Middling on both axes |
| 5 | vllm_r1distill_32b | 371s | 4.612 | Disqualified (slowest + lowest quality) |

Cloud counterweight: **gemini25_flash_lite** at 4.6s mean / 4.738 G-Eval is
7× faster than the vLLM winner and within 0.025 G-Eval points. NOT a prod
swap target (prior research settled the runtime choice as vLLM), but valuable
as the **DGX-contention fallback** when whisper/diarize concurrent workloads
saturate the GB10 — exactly the constraint envelope #1000 tracks.

**Conclusions:**
- **Swap autoresearch vLLM default**: Qwen3-30B-A3B-Instruct-2507 →
  **Qwen3.5-35B-A3B** (BF16). Better on both speed AND quality.
- **Round 2 candidate worth running**: Mistral Small 3.2 24B FP8. If
  quantization halves latency (132s → ~65s mean), it becomes the top vLLM
  summary option by combined rank. ~1hr eval + 1 cold boot.
- **R1-Distill production-disqualified** — 6-min mean, 34-min p99,
  bottom-quality. No further investigation warranted for summary.
- **Magistral, Qwen3-30B-2507 deprecated as candidates** — both dominated
  by Qwen3.5-35B-A3B within the vLLM cohort.

### Methodological gap (intentional, not blocking)

This Phase 2b does NOT include a head-to-head comparison against the current
prod summary model (ollama qwen3.5:35b Q4_K_M). That comparison is held
separately by prior research which established the prod runtime choice; we
do not re-litigate it here. The point of Phase 2b is to pick the best vLLM
candidate within Round 1 and identify Round 2 candidates — not to reopen
"which runtime should prod use."

### Round 2 (quantization) — gated on what gets picked

- **Gemini-Flash-Lite (cloud)**: Round 2 not applicable — quantization is the
  vendor's concern.
- **Ollama qwen35:35b (DGX)**: Round 2 has already been done via Ollama's
  default quantization; no further speedup likely without switching backend.
- **Mistral Small 3.2 24B (vLLM, currently rejected on speed)**: Round 2 is
  the only path that could reopen the swap question — if FP8/NVFP4 halves
  the latency (132s → ~65s mean), it'd land at speed-rank #4 and combined
  rank #3. Worth a single FP8 cell as a sanity check before fully closing
  this door.
- **Qwen3.5-35B-A3B (vLLM)**: similar question but lower priority — its
  Sonnet-side quality lead may be a Sonnet-mimicry artifact.

## Phase 2c scoring approach — TODO

The current `score_gi_insight_coverage.py` measures how well a SUMMARY's
bullets capture silver GI insights — i.e. it expects a summarization run as
input, not a GI extraction run. Our Phase 2c configs run the candidate AS the
GI extractor (output shape `output.gil.nodes`). Need a new scorer:
insight-to-insight coverage (compare candidate-extracted insights against
silver-extracted insights via embedding cosine + count). Same shape needed for
KG: node/edge-to-node/edge coverage. Implement before scoring Phase 2c
predictions.

## Phase 2c — Grounded Insights (GI) scoring

### Predictions

| candidate | status |
|-----------|--------|
| gemini25_flash_lite | 10/10 ✓ |
| vllm_qwen3_30b_a3b_2507 | _pending swap_ |
| vllm_qwen3_5_35b_a3b | _pending swap_ |
| vllm_r1distill_32b | _pending swap_ |
| vllm_magistral_small_2509 | _pending swap_ |
| vllm_mistral_small_3_2_24b | _pending swap_ |

### Dual-silver ROUGE-L

*Pending all 6 candidates' predictions.*

### G-Eval scores

*Pending.*

### Per-stage decision — GI

*Pending.*

## Phase 2c — Knowledge Graph (KG) scoring

### Predictions

| candidate | status |
|-----------|--------|
| gemini25_flash_lite | 10/10 ✓ |
| vllm_qwen3_30b_a3b_2507 | _pending swap_ |
| vllm_qwen3_5_35b_a3b | _pending swap_ |
| vllm_r1distill_32b | _pending swap_ |
| vllm_magistral_small_2509 | _pending swap_ |
| vllm_mistral_small_3_2_24b | _pending swap_ |

### Dual-silver ROUGE-L

*Pending.*

### G-Eval scores

*Pending.*

### Per-stage decision — KG

*Pending.*

## Overall recommendation

*Pending all three per-stage decisions.*

## Open questions for Round 2 (optimization)

Round 2 (FP8/NVFP4/INT8) is gated on this report's winners. If the per-stage decision picks any vLLM-served model, Round 2 evaluates that model at its best-supported quantization vs BF16 to size the speedup vs quality cost. If the per-stage decision picks Gemini-Flash-Lite or Ollama qwen35:35b, Round 2 is skipped for that stage.

## Reproducibility

- Configs: `data/eval/configs/{summarization,gi,kg}_autoresearch_prompt_*_dev_v1.yaml` + `*_dev_paragraph_v1.yaml`
- Silvers: `data/eval/references/silver/silver_{opus47,sonnet46}_{dev_v1_paragraph,gi_dev_v1,kg_dev_v1}/`
- Finale configs: `data/eval/configs/finale/finale_1016_phase2{b,c_gi,c_kg}_*.yaml`
- Sweep runner: `scripts/eval/phase2c_sweep.py`
- Score artifacts: `data/eval/runs/finale/finale_1016_phase2{b,c_gi,c_kg}_*/`

## Cross-references

- Parent: #1016, #928
- Methodology: `autoresearch/JUDGING.md` §Silver / judge vendor-bias rule (added this branch)
- Memory: `feedback_silver_judge_vendor_bias.md`
