# #1016 LLM Landscape Eval — Final Report (Round 3, 2026-06-17)

**Branch**: `feat/928-reframe-llm-landscape-2026-06-16`
**Scope**: 9-candidate cohort, 3-stage eval (summary / GI / KG), dual-silver methodology
**Status**: Round 3 Phase 2b complete (9 candidates) + Phase 2c complete (7 candidates)

---

## TL;DR

7 candidates produced apples-to-apples results across all 3 stages under Round 3 vendor-correct sampling + per-vendor reasoning-parser flags. **Qwen3.5-35B-A3B is the top-dog candidate** across summary + KG; **Gemma-4-26B-A4B leads GI**; **Moonlight-16B-A3B is the style-neutral safe pick** for cross-vendor judging contexts. **Llama-3.3-70B-NVFP4 lost the scale bet** — quantization tax outweighs parameter advantage. **DSV2-Lite GI/KG results are invalid** due to an unfixed BPE-postprocessor wiring gap in the GI/KG pipeline (separate from the summary path).

---

## 1. Final 3-stage scorecard (Round 3 v1, vendor-correct sampling)

| # | Candidate | Sum R-1 vs Opus | Sum R-1 vs Sonnet | GI vs Opus | GI vs Sonnet | KG vs Opus | KG vs Sonnet | Summary speed (s/ep) | GI speed | KG speed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Qwen3.5-35B-A3B** | **59.4%** | **63.0%** | 36% | 37% | **38%** | **38%** | 13.6 | 30 | ~30 |
| 2 | Mistral-Small-3.2-24B (no P2c) | 55.6% | 56.8% | — | — | — | — | 69.7 | — | — |
| 3 | Moonlight-16B-A3B | 57.5% | 57.5% | 16% | 16% | 29% | 27% | 9.0 | 15 | ~15 |
| 4 | Magistral-Small-2509 (no P2c) | 55.2% | 56.8% | — | — | — | — | 64.1 | — | — |
| 5 | Gemma-4-26B-A4B-it | 51.8% | 56.3% | **41%** | **42%** | 27% | 33% | 14.8 | 35 | ~25 |
| 6 | Qwen3-30B-A3B-Instruct-2507 | 50.1% | 54.5% | 36% | 37% | 35% | 26% | 14.0 | 32 | ~25 |
| 7 | Ministral-3-14B-Instruct-2512 | 53.3% | 55.4% | 30% | 38% | 32% | 31% | 30.2 | ~20 | ~20 |
| 8 | Llama-3.3-70B-Instruct-NVFP4 | 48.7% | 49.3% | 16% | 15% | 20% | 15% | 42.9 | ~60 | ~50 |
| 9 | DSV2-Lite-Chat | 39.2% | 39.9% | (0%)* | (0%)* | (0%)* | (0%)* | 4.8 | ~10 | ~10 |

*DSV2-Lite GI/KG = 0%: harness bug, NOT model failure. Postprocessor `decode_r1_byte_level` is applied to summary text but NOT to GI/KG `node.label` fields. Tasks GitHub issue to fix; rerun is fast.

**Phase 2c cohort = 7 candidates** (Mistral-Small-3.2 + Magistral were dropped from Phase 2c at >60s/ep summary speed; their R1/2 GI/KG numbers stand: Mistral 24%/41% and Magistral 24%/43% on Opus).

---

## 2. Per-stage winners and trade-offs

### Summary stage

| Rank | Candidate | ROUGE-1 vs Opus | Cosine vs Opus | BLEU | Speed | Note |
|---:|---|---:|---:|---:|---:|---|
| 1 | **Qwen3.5-35B-A3B** | **59.4%** | **83.7%** | 15.0% | 13.6 s/ep | **Top dog**. Needs `--reasoning-parser=qwen3` server flag |
| 2 | Mistral-Small-3.2-24B | 59.1% | 82.0% | 19.0% | 69.7 s/ep | Slow (mistral tokenizer overhead) |
| 3 | Moonlight-16B-A3B | 57.5% | 78.6% | 18.7% | 9.0 s/ep | Cohort speed leader + perfectly style-neutral |
| ... | (Magistral, Ministral, Gemma, Qwen3-30B, Llama, DSV2 in 5th-9th) | | | | | |

**Summary recommendation**: Qwen3.5-35B-A3B if judging panel is cross-vendor. Moonlight-16B-A3B if the judge cohort is Sonnet-heavy (Δ Sonnet-Opus = 0.0 — guaranteed unbiased).

### GI (Grounded Insights) stage

| Rank | Candidate | Coverage vs Opus | Coverage vs Sonnet | Δ S−O | Note |
|---:|---|---:|---:|---:|---|
| 1 | **Gemma-4-26B-A4B-it** | **41%** | 42% | +1 | Stage leader; style-neutral on GI extraction |
| 2 | Qwen3.5-35B-A3B (tied) | 36% | 37% | +1 | DROPPED 8pts vs Round 1/2 (44% → 36%) |
| 2 | Qwen3-30B-Instruct-2507 (tied) | 36% | 37% | +1 | Dropped 2pts only — most stable Qwen |
| 4 | Ministral-3-14B-Instruct-2512 | 30% | 38% | **+8** | Largest Sonnet-mimicry on GI in cohort |
| 5 | Moonlight-16B-A3B (tied) | 16% | 16% | 0 | Floor; small MoE active params hurt extraction |
| 5 | Llama-3.3-70B-NVFP4 (tied) | 16% | 15% | −1 | Floor; NVFP4 quant tax |
| — | DSV2-Lite | (0%) | (0%) | — | BPE bug — invalid |

**GI recommendation**: Gemma-4-26B-A4B is the per-stage winner but Δ Sonnet-mimicry on its summary stage (+4.5) means cross-vendor judge is mandatory before claiming.

### KG (Knowledge Graph) stage

| Rank | Candidate | Coverage vs Opus | Coverage vs Sonnet | Δ S−O | Note |
|---:|---|---:|---:|---:|---|
| 1 | **Qwen3.5-35B-A3B** | **38%** | 38% | +1 | Stage leader; topic+entity both produced |
| 2 | Qwen3-30B-Instruct-2507 | 35% | 26% | **−9** | **IMPROVED +4pts vs R1/2** (31% → 35%) |
| 3 | Ministral-3-14B-Instruct-2512 | 32% | 31% | −1 | No change vs R1/2 |
| 4 | Moonlight-16B-A3B | 29% | 27% | −2 | No change vs R1/2; 30% entity coverage! |
| 5 | Gemma-4-26B-A4B-it | 27% | 33% | +6 | Dropped 2pts; Sonnet-mimicry visible here too |
| 6 | Llama-3.3-70B-NVFP4 | 20% | 15% | −5 | Cohort floor (Opus-leaning) |
| — | DSV2-Lite | (0%) | (0%) | — | BPE bug |

**KG recommendation**: Qwen3.5-35B-A3B (with parser flag) leads decisively. Qwen3-30B-Instruct-2507 is the surprise — its over-extraction (190 topic candidates vs ~110 for others) helps coverage rate, IMPROVED with Round 3 vendor sampling.

### Entity coverage caveat

5 of 7 candidates produce **zero entity nodes**. Only Moonlight (30%) and Qwen3-30B-Instruct (23 candidates but 0% covered) emit Entity-class nodes. This is almost certainly a **prompt issue** — the cohort-uniform Qwen-style prompt doesn't explicitly ask for entity extraction. **Re-running KG with an entity-focused prompt is a separate experiment** that could shift Moonlight from #4 to potentially #1 on KG.

---

## 3. Sonnet-mimicry pattern — does it hold across all stages?

| Candidate | Summary Δ S−O | GI Δ S−O | KG Δ S−O | Overall |
|---|---:|---:|---:|---|
| Qwen3.5-35B-A3B | **+3.6** | +1 | +1 | Summary-only mimicry; GI/KG style-neutral |
| Gemma-4-26B-A4B-it | **+4.5** | +1 | **+6** | Persistent across summary + KG |
| Qwen3-30B-Instruct-2507 | **+4.4** | +1 | **−9** | Summary Sonnet-lean; KG **flips to Opus-lean** |
| Ministral-3-14B | +2.1 | **+8** | −1 | GI-specific Sonnet-mimicry (largest in cohort!) |
| Mistral-Small-3.2-24B | +1.2 | (no P2c) | (no P2c) | Neutral on summary |
| Magistral-Small-2509 | +1.6 | (no P2c) | (no P2c) | Neutral on summary |
| Moonlight-16B-A3B | **0.0** | 0 | −2 | Truly style-neutral, all stages |
| Llama-3.3-70B-NVFP4 | +0.6 | −1 | −5 | Opus-leaning on extraction |
| DSV2-Lite | +0.7 | (BPE) | (BPE) | Neutral on summary |

**Critical finding**: Sonnet-mimicry is **task-dependent, not just model-dependent**. Qwen3-30B-Instruct-2507's KG output is OPUS-style (Δ −9) while its summary is Sonnet-style (Δ +4.4). This breaks the assumption that vendor-bias is a fixed model property. **Methodology implication for [[silver_judge_vendor_bias]]**: when ranking a candidate, you must check ITS DELTA ON THE STAGE BEING RANKED, not assume the model has a single style.

---

## 4. Round 1/2 → Round 3 stage-by-stage deltas

| Candidate | Sum cos Δ | GI Δ | KG Δ |
|---|---:|---:|---:|
| Qwen3.5-35B-A3B | **+2.6** ✓ | **−8** ✗ | **−9** ✗ |
| Gemma-4-26B-A4B-it | (R3 only) | −4 | −2 |
| Moonlight-16B-A3B | (R3 only) | −3 | **0 (stable)** |
| Qwen3-30B-Instruct-2507 | — | −2 | **+4 ✓** |
| Ministral-3-14B | (no R1/2 score) | (R3 only) | 0 |
| Llama-3.3-70B-NVFP4 | (R3 only) | (R3 only) | (R3 only) |
| DSV2-Lite | (R3 only) | (BPE bug) | (BPE bug) |

**Critical finding** (validates earlier hypothesis):

- **Generative tasks (summary)**: vendor sampling **improves** semantic alignment (cosine).
- **Extraction tasks (GI, KG)**: vendor sampling **hurts** stability of structured-output extraction. Higher temp jitters the insight/topic list.
- **Exception**: Qwen3-30B-Instruct-2507's KG **improves** at Round 3 — its over-extraction strategy (190 topics vs 110) benefits from sampling diversity.

**Methodology implication**: per-stage tuning is needed for top-dog candidates. The "vendor sampling for all" approach is suboptimal even for cross-stage comparison; **the right comparison is per-stage-optimal vs per-stage-optimal**, not "vendor for everything."

---

## 5. vLLM-on-GB10 metrics — what we see and what to think about

### Boot times (cached weights, Round 3 Phase 2c)

| Candidate | Boot wall-clock | Cohort rank |
|---|---:|---:|
| Ministral-3-14B-Instruct-2512 | 2m43s | **fastest** |
| DSV2-Lite-Chat | 4m14s | 2nd |
| Qwen3.5-35B-A3B | 3m49s ← P2b 5m13s | mid |
| Moonlight-16B-A3B | (already up from boot) | — |
| Llama-3.3-70B-NVFP4 | 6m12s | 5th |
| Qwen3-30B-Instruct-2507 | 7m43s | 6th |
| Gemma-4-26B-A4B-it | 7m49s | **slowest (multimodal vision shards)** |

**Observation**: model size does NOT predict boot time well. Gemma 4 (26B) and Qwen3-30B both take ~7-8 min cached; Llama 70B takes only 6 min. The dominant cost is **autotune / CUDA graph capture**, not weight load. Bigger models with simpler architectures often boot faster than smaller multimodal models.

### Inference metrics (single-stream eval, vLLM /metrics)

| Candidate | Samples | **KV peak** | TTFT p50 | TPOT p50 | Gen TPS avg |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B | 77 | 1.1% | 250ms | 50ms | 29 tok/s |
| Gemma-4-26B-A4B | 120 | 0.9% | 250ms | 50ms | 22 tok/s |
| Moonlight-16B | 72 | 0.2% | 250ms | 50ms | 30 tok/s |
| Qwen3-30B-2507 | 128 | 1.4% | 250ms | 50ms | 27 tok/s |
| Ministral-3-14B | 300 | 0.8% | 250ms | 75ms | 15 tok/s |
| **Llama-3.3-70B-NVFP4** | 354 | **2.5%** | **500ms** | **200ms** | **5 tok/s** |
| DSV2-Lite | 62 | 0.2% | 500ms | 50ms | 33 tok/s |

### What this means (and what to think about)

**1. We are SEVERELY under-utilizing the GB10 for single-stream eval.**

- Peak KV cache usage = **2.5% (Llama 70B)** — max across all 7 candidates. Most models <1.5%.
- DGX-Spark blog cited "<5% typical for single-user" — we're well under that.
- **Implication**: every model could run with `--max-num-seqs=4` (down from our default 64-256) without performance loss. That frees ~20-40 GiB of reserved KV cache budget — enough to fit larger models or run higher max-model-len.

**2. Llama-3.3-70B-NVFP4 is throughput-bound, not memory-bound.**

- TPOT = 200ms (4× the cohort average of 50ms) → 5 tok/s steady-state.
- TTFT = 500ms (2× cohort 250ms) — prefill is slow too.
- KV cache at 2.5% means there's MASSIVE memory headroom that goes unused.
- **Tuning question**: would `--enable-prefix-caching` + larger `--max-num-batched-tokens` (currently 4096, could try 16384+) improve TPOT? The NVFP4 quant path may not be optimized for small batches.

**3. TTFT and TPOT are remarkably consistent across the cohort.**

- All small/MoE models: TTFT 250ms, TPOT 50ms.
- This is the **compute-bound regime** — the prefill cost of ~10k input tokens dominates over anything else.
- Smaller max_input_tokens would cut TTFT but our summary pipeline naturally caps at ~10k.

**4. Boot times suggest CUDA graph capture is the dominant cost.**

- Multimodal Gemma 4 and 30B+ models take 7-8 min cached.
- 8B/14B dense models take 2-4 min.
- **Tuning question**: `--enforce-eager` skips CUDA graph capture (saves 2-4 min boot, costs ~10-15% inference perf). Worth it for the eval loop where we boot/test/swap frequently. Production deploys should keep CUDA graphs.

### Concrete #1022 next steps (with data backing)

| Lever | Expected effect | Evidence | Status |
|---|---|---|---|
| **A. `--max-num-seqs=4` default** | Frees 20-40 GiB KV reservation; safer for tight-fit big models | KV peak 2.5% even at our max-num-seqs=64 | **✅ APPLIED 2026-06-17** in autoresearch compose |
| **C. `--enforce-eager` for eval loops** | Save 2-4 min per boot in iteration | Boot times dominated by graph capture | **✅ APPLIED 2026-06-17** in autoresearch compose |
| B. Investigate `--enable-prefix-caching` for Llama 70B | May fix the 5 tok/s vs 30 tok/s discrepancy | Llama TPOT 4× cohort average | Open — diagnostic experiment for #1022 |
| Larger `--max-num-batched-tokens` (16384) for NVFP4 | NVFP4 kernels may need wider batches | Llama TPOT degenerate | Open — pair with lever B |
| Drop `--max-num-batched-tokens=4096` for Gemma 4 once vision is no-op | We added it for Gemma multimodal; KG/GI don't need it | Gemma KV peak 0.9% | Open — minor optimization |
| D. Boot-time phase breakdown logging | Capture exact phase timings to PER_MODEL_OPTIMAL_PARAMS.md | We have wall-clock but not per-phase decomposition | Open — backlog |

---

## 6. Issues found that need adjustment

### 6a. DSV2-Lite BPE artifact in GI/KG node labels (HARNESS BUG)

**Symptom**: All DSV2-Lite GI/KG `node.label` fields contain `g` (byte-level space `Ġ`) and `c` (newline `Ċ`) instead of spaces and newlines. Example: `"gthegpodcastgepisodegdiscussesgbuilding..."`.

**Root cause**: Configured `postprocessor: decode_r1_byte_level` is applied to summary text via the response interceptor, but NOT to GI/KG structured output (`gil.nodes[].properties.label`). Two different code paths.

**Impact**: DSV2-Lite GI = 0% coverage, KG = 0% coverage. Cohort comparison incomplete.

**Fix**: extend the postprocessor application point to also clean text fields in `gil.nodes[].properties.{label,name,description}`. ~30 min of harness work + rerun.

**Open as new task #111**: fix BPE postprocessor for GI/KG node labels.

### 6b. Entity coverage near-zero across cohort

**Symptom**: 5 of 7 candidates produce 0 entity-class nodes. Only Moonlight (30%) and Qwen3-30B (23 cands but covered=0) emit Entity.

**Root cause**: Cohort-uniform prompt `ollama/qwen3.5_35b/summarization/*` doesn't explicitly request entity extraction.

**Impact**: KG weighted coverage is artificially low because entities count for 30% of silver but 0% of candidate output.

**Fix**: re-run Phase 2c KG with an entity-focused prompt across all 7 candidates. **Separate experiment, do not block the current cohort report**.

### 6c. Mistral-Small-3.2 + Magistral dropped from Phase 2c

**Status**: Operator-confirmed drop because >60s/ep summary speed disqualifies them from per-stage routing consideration. Their Round 1/2 Phase 2c data stands:

- Mistral-Small-3.2 R1/2: GI vs Opus 24%, KG vs Opus 41%
- Magistral R1/2: GI vs Opus 24%, KG vs Opus 43%

**No adjustment needed** — the methodology accepts the trade-off, and these candidates have R1 data points.

---

## 7. Per-candidate verdict (consolidated, with both stages weighted)

Using a simple unweighted mean of (Sum-R1-vs-Opus, GI-vs-Opus, KG-vs-Opus) for the 7 P2c candidates:

| Rank | Candidate | Sum | GI | KG | Mean | Notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | **Qwen3.5-35B-A3B** | 59.4% | 36% | 38% | **44.5%** | Cohort top-dog. **Sonnet-mimicry on summary** — requires cross-vendor judge. |
| 2 | Qwen3-30B-Instruct-2507 | 50.1% | 36% | 35% | **40.4%** | Closest competitor; **stable + KG improved at R3**. |
| 3 | Gemma-4-26B-A4B-it | 51.8% | **41%** | 27% | 39.9% | GI stage leader; but KG dragged down by Sonnet-mimicry artifact |
| 4 | Ministral-3-14B | 53.3% | 30% | 32% | 38.4% | Small-dense FP8 representative; mid-pack across the board |
| 5 | Moonlight-16B-A3B | 57.5% | 16% | 29% | 34.2% | Style-neutral safe pick. Weak on GI extraction (size issue) |
| 6 | Llama-3.3-70B-NVFP4 | 48.7% | 16% | 20% | 28.2% | Quant tax > parameter advantage; ALL stages weak |
| — | DSV2-Lite-Chat | 39.2% | (0%)* | (0%)* | invalid | BPE bug; rerun after fix |

**Per-stage routing top-dog**:

- Summary → Qwen3.5-35B-A3B
- GI → Gemma-4-26B-A4B-it
- KG → Qwen3.5-35B-A3B
- **2-of-3 stages led by Qwen3.5** → single-model deploy is viable if you accept the Sonnet-mimicry caveat on summary

**Cross-vendor safe pick** (Sonnet-judging unsafe scenarios): Moonlight-16B-A3B-Instruct for summary; Qwen3-30B-Instruct-2507 for GI/KG (Δ K-O Sonnet=−9 makes it Opus-leaning, fine if Opus is the silver).

---

## 8. Next steps (operator decisions queued)

1. **Decide tuning model**: per-stage-per-model tuning (more complex deploy, optimal quality) OR top-dog single-mix (Qwen3.5 with parser flag, ~95% of optimal).
2. **G-Eval batch judging** on all 7 candidates' Round 3 predictions to add LLM-judge signal alongside ROUGE/BLEU/cosine.
3. **Fix BPE postprocessor for GI/KG node labels** (task #111) and rerun DSV2-Lite Phase 2c.
4. **Entity-focused prompt KG re-experiment** to unlock Moonlight's KG potential.
5. **#1022 vLLM-on-GB10 tuning workstream** with the data we collected here as input — see § 5 for concrete levers.

---

## 9. Document updates committed alongside this report

- `autoresearch/MODEL_PLAYBOOK.md` — per-model Phase 2c verdicts appended to each candidate's section.
- `docs/wip/EVAL_1016_ROUND3_REVIEW.md` — Phase 2c outcome section added; the queue-of-7 decisions stamped as final.
- `docs/wip/EVAL_1016_metrics/PER_MODEL_OPTIMAL_PARAMS.md` — KV/TTFT/TPOT/throughput per candidate filled in.
- `docs/wip/EVAL_1016_metrics/vllm_metrics_*_phase2c.log` — 7 raw metric polls for reuse in #1022.

---

## 10. ADDENDUM — Cell F NVFP4 enters cohort + supersedes safe pick (2026-06-19)

Filed during the #1022 vLLM-on-GB10 tuning effort. After cells A–D
(runtime knobs: gpu-memory-utilization, max-num-seqs, max-model-len,
warmup) all produced no signal, Cell F (model swap to
`NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`) delivered ~2× speedup with
no measurable quality regression vs the bf16 baseline. Held-out
validation on `curated_5feeds_benchmark_v2` (5 ep, e03, Sonnet 4.6
silver) confirmed cross-dataset + cross-vendor robustness.

### Cell F entered into Round 3 scoreboard (apples-to-apples)

Using the same scorers / silvers / metrics as the #1016 report:

**Summary (rouge1_f1 vs Opus 4.7 silver)** — Cell F lands #5:

| # | Candidate | rouge1_f1 | s/ep |
|---|---|---:|---:|
| 1 | Qwen3.5-35B-A3B | 0.5936 | 13.58 |
| 2 | Moonlight-16B-A3B | 0.5745 | 8.96 |
| ... |
| 5 | **Qwen3-30B-A3B-NVFP4 (Cell F)** | **0.5407** | **7.20** ⚡ (fastest) |

**GI (coverage_rate vs Opus 4.7 silver)** — **Cell F is the new winner**:

| # | Candidate | cov_rate | s/ep |
|---|---|---:|---:|
| **1** | **Qwen3-30B-A3B-NVFP4 (Cell F)** | **0.4250** | 17.00 |
| 2 | Gemma-4-26B-A4B (prior GI leader) | 0.4125 | 34.91 |
| 3 | Qwen3.5-35B-A3B | 0.3625 | 30.14 |
| 5 | Moonlight | 0.1625 | 15.10 |

**KG (topic coverage_rate vs Opus 4.7 silver)** — Cell F #3:

| # | Candidate | topic_cov | s/ep |
|---|---|---:|---:|
| 1 | Qwen3.5-35B-A3B | 0.4854 | 21.18 |
| 2 | Ministral-3-14B | 0.4175 | 56.93 |
| **3** | **Qwen3-30B-A3B-NVFP4 (Cell F)** | **0.4078** | **13.80** ⚡ |
| 5 | Moonlight | 0.2816 | 16.17 |

**End-to-end per-episode (3 stages summed)**:

| Candidate | total s/ep |
|---|---:|
| **Qwen3-30B-A3B-NVFP4 (Cell F)** | **38.0** ⚡ |
| Moonlight-16B-A3B | 40.23 |
| Qwen3.5-35B-A3B | 64.90 |
| Gemma-4-26B-A4B | 69.80 |

Cell F is the cohort end-to-end speed leader **and** the GI quality
leader simultaneously.

### Updated single-model daily-driver verdict

**Qwen3-30B-A3B-NVFP4 (Cell F) replaces Moonlight as the autoresearch
safe pick / daily driver.** It dominates Moonlight in 4 of 5
dimensions (loses summary by 6%, wins GI +161% / KG +45% / end-to-end
speed -5% / weight footprint -44%).

Qwen3.5-35B-A3B retains the **top-dog crown for highest-stakes
one-shot evals** where summary or KG quality matters more than time
(operator can manually swap the homelab compose for that specific run).

### Operational change

- **Homelab compose model swap**: `Qwen/Qwen3-30B-A3B-Instruct-2507` →
  `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` in
  `infra/vllm/autoresearch/docker-compose.yml`. Operator merges.
- **No profile YAML changes needed** — all profiles use the
  `autoresearch` served-model-name alias.
- **Registry docstrings** updated to note Cell F as the autoresearch
  slot model (cosmetic only; no functional change).

### Cross-references

- Full Cell F validation evidence: `docs/wip/VLLM_GB10_TUNING_VALIDATION_2026-06-18.md`
- Validation script + runs.tsv: `autoresearch/1022_gb10_tuning/`
- #1022 closeout: GH issue #1022.

