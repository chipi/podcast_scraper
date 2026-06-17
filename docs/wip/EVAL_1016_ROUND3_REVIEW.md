# #1016 Round 3 Review — Summary stage cohort scorecard + recommendations

**Date**: 2026-06-17
**Branch**: `feat/928-reframe-llm-landscape-2026-06-16`
**Scope**: Phase 2b (summary) only. Phase 2c (GI + KG) not refreshed in Round 3.

---

## Executive summary

Round 3 was a vendor-correct sampling re-run of the #1016 cohort, motivated
by a deep documentation pass on the 14 candidate HF model cards that revealed
multiple harness gaps (missing reasoning-parser flags, mistral tokenizer
flags, vendor-recommended sampling never applied because `SummarizationParams`
silently dropped `top_p`/`top_k`/etc fields).

**9 candidates passed Phase 2b gate end-to-end.** 2 were dropped (Kimi-Linear:
doesn't fit GB10 memory; DSR1-0528-Qwen3-8B: output length intrinsically
exceeds any reasonable cohort cap). 1 deferred (Mistral-Small-4-119B-NVFP4
needs compose tuning).

**Key methodology finding**: vendor-correct sampling AMPLIFIED Sonnet-mimicry
detection that greedy decoding had masked. Three candidates show clear
Sonnet-lean (Δ ≥ +3.6 ROUGE-1); three are style-neutral; three are
ambiguous/noise. This materially affects per-stage routing if a single
judge vendor is used.

---

## Final Round 3 cohort scorecard (Phase 2b summary)

Silvers: Opus 4.7 + Sonnet 4.6 (paragraph form), dataset `curated_5feeds_dev_v1`
(10 episodes).

| Rank | Candidate | ROUGE-1 Opus | ROUGE-1 Sonnet | Δ (S−O) | Cos Opus | BLEU | Speed (mean) |
|----:|-----------|-------------:|---------------:|--------:|---------:|-----:|-------------:|
| 1 | **Qwen3.5-35B-A3B** | **59.4%** | **63.0%** | +3.6 | **83.7%** | 15.0% | 13.6s |
| 2 | Moonlight-16B-A3B-Instruct | 57.5% | 57.5% | **0.0** | 78.6% | **18.7%** | 9.0s |
| 3 | Mistral-Small-3.2-24B-Instruct-2506 | 55.6% | 56.8% | +1.2 | 80.9% | 15.3% | 69.7s |
| 4 | Magistral-Small-2509 | 55.2% | 56.8% | +1.6 | 80.9% | 14.5% | 64.1s |
| 5 | Ministral-3-14B-Instruct-2512 | 53.3% | 55.4% | +2.1 | 81.2% | 10.7% | 30.2s |
| 6 | Gemma-4-26B-A4B-it | 51.8% | 56.3% | +4.5 | 80.6% | 8.6% | 14.8s |
| 7 | Qwen3-30B-A3B-Instruct-2507 | 50.1% | 54.5% | +4.4 | 78.6% | 8.2% | 14.0s |
| 8 | Llama-3.3-70B-Instruct-NVFP4 | 48.7% | 49.3% | +0.6 | 77.9% | 8.9% | 42.9s |
| 9 | DeepSeek-V2-Lite-Chat | 39.2% | 39.9% | +0.7 | 72.5% | 4.9% | **4.8s** |

Reference: cohort gate threshold mean<100s, p99<150s, cv<0.3, chars 800-3200.
All 9 passed all 4 criteria.

---

## Sonnet-mimicry detection findings

**Methodology**: each candidate scored against TWO silvers (Anthropic Opus 4.7
and Anthropic Sonnet 4.6). The delta `Δ = ROUGE-1(Sonnet) − ROUGE-1(Opus)` is
the lean signal. Positive Δ = candidate's outputs match Sonnet-style better.

**Three clear Sonnet-lean candidates** (Δ ≥ +3.6):
- **Gemma-4-26B-A4B-it**: +4.5 (largest signal)
- **Qwen3-30B-A3B-Instruct-2507**: +4.4
- **Qwen3.5-35B-A3B**: +3.6

**Three style-neutral candidates** (|Δ| < 1.0):
- Moonlight-16B-A3B: **0.0** (perfectly neutral — unique in cohort)
- Llama-3.3-70B-NVFP4: +0.6
- DSV2-Lite: +0.7

**Three borderline** (1.0 ≤ Δ ≤ 2.1, within noise):
- Mistral-Small-3.2-24B: +1.2
- Magistral-Small-2509: +1.6
- Ministral-3-14B-Instruct-2512: +2.1

**Critical methodology note**: the Round 1/2 dual-silver pass at greedy
temp=0.0 showed Qwen3.5 at Δ ≈ 0.002 (essentially style-neutral). Round 3
with vendor-correct sampling + `--reasoning-parser=qwen3` REVEALED Qwen3.5's
Sonnet-lean at Δ = +3.6 on ROUGE-1 (and +2.3 on ROUGE-L). Implication: greedy
decoding masks style-mimicry by collapsing diverse outputs toward neutral
midpoints. **Vendor-correct sampling is the configuration that reveals the
true vendor-lean and is also the one prod would use.**

**Implication for [[silver_judge_vendor_bias]]**: when ranking candidates
that include Anthropic-mimicry-prone families (Qwen3 series, Gemma 4), the
final judge panel MUST include a non-Anthropic judge (GPT-5.4, Gemini 2.5
Pro) to disclose the bias empirically. A Sonnet-only judge would systematically
over-rank the three Sonnet-lean candidates.

---

## Per-stage routing recommendation (provisional)

### Summary stage

**Three viable top-dog candidates depending on judging philosophy**:

| Candidate | Best when... |
|-----------|--------------|
| **Qwen3.5-35B-A3B** | Cross-vendor judge panel mitigates the +3.6 Sonnet lean. Cohort-leading on every metric vs both silvers; ROUGE-1 = 59-63%, cos = 83.7%. Mid-speed (13.6s/ep). |
| **Moonlight-16B-A3B** | **Style-neutrality is a hard requirement** (auditable cross-vendor publishing, customer-facing summaries graded by mixed cohort). Δ = 0.0 (unique in cohort). Top-3 quality + cohort-#2 speed (9.0s). |
| **DSV2-Lite-Chat** | GPU minute cost dominates summary quality. Cheapest at 4.8s/ep but is the quality floor (ROUGE-1 = 39.2%). |

### GI + KG stages

**Round 3 refresh cohort (operator-set 2026-06-17)**: 7 candidates only — the
>60s/ep candidates are dropped from Phase 2c because their position on the
(quality, speed) Pareto frontier is dominated by Qwen3.5 / Moonlight at
comparable-or-better quality. Round 1/2 Phase 2c data for the dropped pair
is retained as their final landscape numbers (acceptable because they're
style-neutral and structural extraction is less sampling-sensitive than
summary quality).

**Dropped from Phase 2c refresh**:
- **Mistral-Small-3.2-24B-Instruct-2506** (69.7s/ep summary — would be even
  slower for GI + KG due to structured output complexity)
- **Magistral-Small-2509** (64.1s/ep, reasoning model, same Mistral family
  already represented by Ministral-3-14B at 30.2s)

**Round 3 Phase 2c refresh queue (FINAL, operator-vetted 2026-06-17)** —
7 candidates:
1. **Qwen3.5-35B-A3B** — Round 1/2 KG winner; must re-verify with parser flag
2. **Gemma-4-26B-A4B** — Round 1/2 GI leader (45%)
3. **Moonlight-16B-A3B** — top-3 summary; weak GI/KG in Round 1 (19%/29%)
4. **Qwen3-30B-Instruct-2507** — middle of pack
5. **Ministral-3-14B-Instruct-2512** — small-dense FP8 + Mistral family
   representative (only dense small-model angle in cohort)
6. **Llama-3.3-70B-NVFP4** — only large-dense candidate (Kimi/Small-4 both
   dropped/deferred); 42.9s acceptable for the "does scale help GI/KG" signal
7. **DSV2-Lite** — speed control candidate

**Architectural coverage of final 7**:
- Dense small (14B FP8): Ministral
- Dense large (70B NVFP4): Llama
- MoE small (16B A3B): Moonlight + DSV2-Lite
- MoE mid (26B A4B + 30B A3B): Gemma + Qwen3-30B-Instruct
- MoE large (35B A3B): Qwen3.5
- Reasoning-by-default (with parser flag): Qwen3.5
- Vendors: Anthropic-mimicry-prone (Qwen3 family, Gemma 4) + style-neutral
  (Moonshot, Mistral via Ministral, Meta via Llama, DeepSeek via DSV2-Lite)

Round 1/2 Phase 2c leaders (pre-Round 3, will likely shift with parser flag fix):
- GI vs Opus silver leader (DGX): Gemma-4-26B-A4B (45%), then Qwen3.5-35B-A3B
  (44%)
- KG vs Opus silver leader (DGX): Qwen3.5-35B-A3B (47% — decisive)

The same Sonnet-mimicry amplification effect likely applies — `--reasoning-parser=qwen3`
+ vendor sampling may shift Qwen3.5's GI/KG numbers materially.

---

## Dropped candidates

### Kimi-Linear-48B-A3B-Instruct (Moonshot)

- **Why**: 91.5 GiB BF16 weights + ~25-30 GiB activation/CUDA-graph overhead =
  ~117-122 GiB on a 128 GiB GB10 unified-mem ceiling. Three boot attempts at
  varying util (0.75/0.88/0.92) + max-model-len (32768/8192) + max-num-seqs
  (128/8/4) + `--enforce-eager` all hit `Available KV cache memory: -15 to
  -17 GiB`. Structural — model doesn't fit.
- **No quant available**: community has open feature request for FP6; not
  released. BF16 is the only weight format.
- **Substitute**: Moonlight-16B-A3B-Instruct (same vendor, smaller, already
  passed) represents Moonshot in the final landscape.

### DeepSeek-R1-0528-Qwen3-8B

- **Why**: `--reasoning-parser=deepseek_r1` server flag IS working (no `<think>`
  blocks in content). But the model's ANSWER ALONE is intrinsically 4096+
  tokens at max_length=4096 → finish_reason=length. Vendor recommends
  output length 32k+ for benchmarks (AIME averaged 23k tokens/question).
  Same conclusion as R1-Distill-Qwen-32B in Phase 2a: R1 family is
  reasoning-research-grade output, not summary-stage-grade.
- **Substitute**: DeepSeek-V2-Lite-Chat (chat model, not reasoning)
  represents DeepSeek in the final landscape.

### Mistral-Small-4-119B-NVFP4 (DEFERRED)

- **Why**: Phase B OOM was a compose-tuning issue (`--max-num-seqs=4
  --max-model-len=8192 --gpu-memory-utilization=0.70` per DGX-Spark blog —
  never wired). Not a model-class problem.
- **Status**: deferred. Could be tried in a follow-up if cohort needs a
  larger Mistral candidate, but Mistral-Small-3.2-24B already represents
  Mistral well and is faster.

---

## What Round 3 found about the harness (not the models)

### Critical gaps surfaced and fixed/documented

1. **`SummarizationParams` silently drops `top_p`/`top_k`/`presence_penalty`**:
   pydantic default-ignore extras. Round 1/2 configs that set these in
   `params:` block were producing greedy-equivalent outputs. Workaround:
   route via `backend.extra_body`. Proper fix queued as task #108.
2. **`--reasoning-parser=qwen3` was never wired for Qwen3.5 family**: model
   is thinking-by-default. Without parser flag, think content leaks into
   response. Required for cohort-correct Qwen3.5 quality.
3. **Mistral family requires 4 tokenizer flags** (`--tokenizer_mode=mistral
   --config_format=mistral --load_format=mistral --tool-call-parser=mistral`)
   per vendor — never wired in Round 1/2. Slows the Mistral candidates 2-3×
   but preserves quality.
4. **`--reasoning-parser=mistral` required for Magistral**: reasoning model
   with `[THINK]/[/THINK]` tokens. Without flag, think prose would consume
   token budget like R1 family.
5. **`extract_json_summary_field` postprocessor added** to REGISTRY for
   structured-output experiments (used in Kimi-Linear v3_json, may extend
   to other JSON-mode use cases).
6. **`onboard_model_smoke.py` bug fixes**: ExperimentConfig.from_yaml ->
   load_experiment_config; system-prompt-optional support; hello max_tokens
   bumped to 200 (was 20; too small for reasoning models that strip-on-output);
   hello content=None handling (reasoning-parser-strip leaves None content
   for too-small budgets).

### Configs landed (11 Round 3 YAMLs)

`data/eval/configs/summarization/autoresearch_prompt_vllm_*_dev_paragraph_round3_v1.yaml`
plus the 4 dropped/deferred config variants (kimi v1/v3/v4, DSR1 v1).

---

## Open follow-ups

1. **G-Eval batch judging** on all 9 Round 3 predictions (Sonnet 4.6 + GPT-5.4,
   cross-vendor). The G-Eval rank delta vs the ROUGE rank will disclose how
   much of the Sonnet-mimicry signal actually moves judge preferences.
2. **Phase 2c GI + KG refresh** for the top-3 summary candidates (Qwen3.5,
   Moonlight, Mistral-Small-3.2) using Round 3 sampling.
3. **`SummarizationParams` extension** (task #108) to make `top_p`/`top_k`/etc
   first-class — current `backend.extra_body` workaround is fragile.
4. **`SYSTEM_PROMPT.txt` fetch + render** (task #109) for the Mistral family.
   Currently the Mistral candidates use the cohort-uniform Qwen-style prompt;
   vendor recommends loading their `SYSTEM_PROMPT.txt`.
5. **Mistral-Small-4-119B-NVFP4 retry** with compose tuning, only if cohort
   wants a larger Mistral.
6. **#1022 vLLM-on-GB10 tuning exploration**: the Kimi-Linear / Mistral-Small-4
   memory ceiling problem is a real constraint; documenting per-model
   compose-flag recipes in the homelab repo is on the backlog.

---

## Tickets touched

- **#1016 Phase 2b Round 3** — closed by this review
- **#908** task tracking: #105 (Round 3 plan), #106 (Kimi-Linear deep-dive),
  #107 (DSR1 deep-dive), #110 (Round 3 execution) all CLOSED.
- **#108** (top_p/response_format harness work) — open
- **#109** (mistral tokenizer flags + SYSTEM_PROMPT.txt) — open
- **#1022** (vLLM-on-GB10 tuning) — open backlog
