# RFC-097 v2.0 baseline — silver rebuild + scoreboard re-baseline (2026-06-20)

**Status**: complete (chunk 7 of RFC-097) + post-sweep candidate re-run
**Branch**: `feat/corpus-ontology-v2`
**Dataset**: `curated_5feeds_dev_v1` (10 episodes)
**Silver models**: Claude Opus 4.7 + Sonnet 4.6 (Anthropic; vendor-disjoint from the entire candidate cohort — #939 lesson preserved)

## Headline (post-sweep)

**The KG-entity 0% coverage was a stale-cohort artifact, not a measurement bug or model failure.**
The 2026-06-17 cohort runs predated RFC-097 chunks 3-5 (v2 emit pipeline). Re-running candidates
through the current pipeline produces silver-grade entity coverage:

| Candidate | KG entities (was → now) | KG overall (was → now) | GI coverage (was → now) |
| --- | --- | --- | --- |
| `vllm_mistral_small_3_2_24b` | 0% → **100%** | 39.6% → **82.8%** (+43.2) | 25.0% → **91.2%** (+66.2) |
| `vllm_qwen3_5_35b_a3b` | 0% → **100%** | 48.5% → **82.1%** (+33.6) | 37.5% → **85.0%** (+47.5) |
| `vllm_ministral_3_14b` | 0% → **100%** | 33.6% → **79.9%** (+46.3) | 25.0% → **83.8%** (+58.8) |
| `gemini_gemini25_flash_lite` | 26.7% → **100%** | 52.2% → **76.1%** (+23.9) | 72.5% → **96.2%** (+23.7) |
| `vllm_gemma_4_26b_a4b` | 0% → **100%** | 30.6% → **74.6%** (+44.0) | 42.5% → **90.0%** (+47.5) |
| `vllm_magistral_small_2509` | 0% → **100%** | 40.3% → **72.4%** (+32.1) | 25.0% → **90.0%** (+65.0) |
| `vllm_qwen3_30b_a3b_instruct_2507` | 0% → **100%** | 32.8% → **71.6%** (+38.8) | 37.5% → **91.2%** (+53.7) |
| `vllm_moonlight_16b_a3b` | 30.0% → **83.3%** | 31.3% → **61.2%** (+29.9) | 16.2% → **63.7%** (+47.5) |

The fix had two components:

1. **v2 emit pipeline** (chunks 3-5): Person/Organization typed nodes,
   edge_class metadata, position_hint waterfall, insight_type vocab
   alignment. Old cohort runs scored against new silvers
   under-represented because the emission shapes mismatched the silver
   schema. Re-running through today's pipeline aligns both sides.
2. **Per-model vLLM:26.05-py3 flags** (chunk-7 workaround sweep,
   2026-06-21): 5 of 8 models needed specific flags to boot/run
   cleanly on the new image — `--tokenizer-mode mistral
   --config-format mistral --load-format mistral` for Mistral3
   multimodal arch, `--reasoning-parser mistral` + max_tokens 4096
   for Magistral, `--language-model-only` to skip Pixtral vision
   tower on text-only workload, corrected HF id
   `Ministral-3-14B-Instruct-2512` for Ministral. All 5 succeeded
   after the workarounds. See
   `EVAL_RFC097_CHUNK7_VLLM_WORKAROUNDS_2026_06_21.md`.

## Silver regen results (2026-06-20)

All four dev_v1 silvers regenerated against the v2/v3-emitting pipeline.
Stats (n_episodes always 10):

| Silver | Model | Topics / Insights | Entities verified |
| --- | --- | --- | --- |
| `silver_opus47_kg_dev_v1` | claude-opus-4-7 | 104 topics | 30/30 |
| `silver_opus47_gi_dev_v1` | claude-opus-4-7 | 80 insights | quotes 83/84 (99%) |
| `silver_sonnet46_kg_dev_v1` | claude-sonnet-4-6 | 97 topics | 30/30 |
| `silver_sonnet46_gi_dev_v1` | claude-sonnet-4-6 | 80 insights | quotes 87/96 (91%) |

## Updated v2-pipeline scoreboard

All 9 candidates re-run through the current pipeline. Llama-3.3-70B-NVFP4
dropped from the cohort per operator decision (deprecated weeks ago — not
re-run despite weights being on DGX). The 5 candidates that failed the
initial sweep (Gemma, Mistral-3.2, Magistral, Moonlight, Ministral) were
re-run after applying per-model flags from
`autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` — see
`EVAL_RFC097_CHUNK7_VLLM_WORKAROUNDS_2026_06_21.md` for the per-model
fix log.

### KG — overall weighted coverage / topic% / entity% (sorted by opus47 overall)

| Candidate | vs silver_opus47 | vs silver_sonnet46 |
| --- | ---: | ---: |
| `vllm_mistral_small_3_2_24b` | **82.8%** (T:78%, E:**100%**) | **81.9%** (T:76%, E:**100%**) |
| `vllm_qwen3_5_35b_a3b` | **82.1%** (T:77%, E:**100%**) | **79.5%** (T:73%, E:**100%**) |
| `vllm_ministral_3_14b` | **79.9%** (T:74%, E:**100%**) | **81.9%** (T:76%, E:**100%**) |
| `gemini_gemini25_flash_lite` | **76.1%** (T:69%, E:**100%**) | **73.2%** (T:65%, E:**100%**) |
| `vllm_gemma_4_26b_a4b` | **74.6%** (T:67%, E:**100%**) | **74.8%** (T:67%, E:**100%**) |
| `vllm_magistral_small_2509` | **72.4%** (T:64%, E:**100%**) | **72.4%** (T:64%, E:**100%**) |
| `vllm_qwen3_30b_a3b_instruct_2507` | **71.6%** (T:64%, E:**100%**) | **74.0%** (T:66%, E:**100%**) |
| `vllm_moonlight_16b_a3b` | **61.2%** (T:55%, E:83%) | **57.5%** (T:50%, E:83%) |
| `vllm_deepseek_v2_lite_chat` | **1.5%** (T:2%, E:0%) | **0.8%** (T:1%, E:0%) |

### GI — insight-to-insight coverage @ 0.65 cosine (sorted by opus47)

| Candidate | vs silver_opus47 | vs silver_sonnet46 |
| --- | ---: | ---: |
| `gemini_gemini25_flash_lite` | **96.2%** | **92.5%** |
| `vllm_qwen3_30b_a3b_instruct_2507` | **91.2%** | **90.0%** |
| `vllm_mistral_small_3_2_24b` | **91.2%** | **85.0%** |
| `vllm_gemma_4_26b_a4b` | **90.0%** | **92.5%** |
| `vllm_magistral_small_2509` | **90.0%** | **87.5%** |
| `vllm_qwen3_5_35b_a3b` | **85.0%** | **87.5%** |
| `vllm_ministral_3_14b` | **83.8%** | **85.0%** |
| `vllm_moonlight_16b_a3b` | **63.7%** | **56.2%** |
| `vllm_deepseek_v2_lite_chat` | **3.8%** | **1.2%** |

## Stale-cohort scoreboard (2026-06-17 runs, pre-chunk-3 pipeline)

Retained for reference. **These are NOT the v2 baseline** — they show the
shape-mismatch artifact, not candidate quality:

| Candidate | KG opus47 | KG sonnet46 | GI opus47 | GI sonnet46 |
| --- | ---: | ---: | ---: | ---: |
| `gemini_gemini25_flash_lite` | 52.2% | 50.4% | 72.5% | 72.5% |
| `vllm_qwen3_5_35b_a3b` | 48.5% | 45.7% | 37.5% | 41.2% |
| `vllm_magistral_small_2509` | 40.3% | 37.0% | 25.0% | 22.5% |
| `vllm_mistral_small_3_2_24b` | 39.6% | 38.6% | 25.0% | 35.0% |
| `vllm_ministral_3_14b` | 33.6% | 35.4% | 25.0% | 30.0% |
| `vllm_qwen3_30b_a3b_instruct_2507` | 32.8% | 33.9% | 37.5% | 35.0% |
| `vllm_moonlight_16b_a3b` | 31.3% | 26.8% | 16.2% | 15.0% |
| `vllm_gemma_4_26b_a4b` | 30.6% | 32.3% | 42.5% | 45.0% |
| `vllm_llama_3_3_70b_nvfp4` | 20.1% | 21.3% | 16.2% | 15.0% |
| `vllm_deepseek_v2_lite_chat` | 0.0% | 0.0% | 0.0% | 0.0% |

## Pack leader investigation — what we learned

**The diagnosis path** (2026-06-21 sideways investigation):

1. Initial observation: 7/10 candidates emit 0 KG entities; even the 3 non-zero
   candidates (Gemini, Qwen3-30B, Magistral) emit role-noun categories like
   "Trail builders" instead of named individuals (Maya, Liam).
2. Direct probe with the v4 KG prompt against today's live Qwen3-30B vLLM
   returned correct entities (Maya, Liam, Singletrack Sessions) on first try.
3. cleaning_v4 preprocessing strips episode headers (Host:/Guest: lines) and
   anonymizes speaker prefixes to A:/B:. Despite this, Qwen3-30B still
   correctly extracts named entities from in-dialogue mentions (which appear
   2-3× per episode after cleaning).
4. Re-ran Qwen3-30B against `curated_5feeds_dev_v1` through the current
   pipeline: 30/30 entities perfect match with silver, 91% GI coverage.
5. Re-ran Gemini (cloud, model unchanged from June): same result — 30/30
   entities, 96% GI coverage.

**Conclusion**: The 0%-entity scores were a stale-cohort artifact, not a
candidate-prompt or model-quality issue. The chunks 3-5 v2 emit pipeline
is the fix; no new prompt was needed.

## Sweep outcomes — all candidates (2026-06-21)

Two rounds:

1. **Initial sweep** (2026-06-21 ~06:00–10:00): 5 of 8 models failed to
   load/run cleanly on `vllm:26.05-py3` — see failure-mode notes below.
2. **Workaround retry sweep** (2026-06-21 ~12:46–17:00, per
   `EVAL_RFC097_CHUNK7_VLLM_WORKAROUNDS_2026_06_21.md`): per-model
   flags from `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` applied to
   the 5 failures. All 5 succeeded.

| Candidate | Outcome | Notes |
| --- | --- | --- |
| `Qwen3-30B-A3B-Instruct-2507` | ✅ ran | pack leader; default flags |
| `Qwen3.5-35B-A3B` | ✅ ran | default flags |
| `Gemini 2.5 Flash Lite` | ✅ ran | cloud anchor |
| `DeepSeek-V2-Lite-Chat` | ✅ ran (weak: 1.5% KG / 3.8% GI) | model genuinely weak; not a load failure |
| `Gemma-4-26B-A4B-it` | ✅ ran (workaround) | `--max-num-batched-tokens 4096 --max-num-seqs 4 --enforce-eager` (Phase 2c flags) |
| `Mistral-Small-3.2-24B-Instruct-2506` | ✅ ran (workaround) | + `--tokenizer-mode mistral --config-format mistral --load-format mistral --language-model-only` |
| `Magistral-Small-2509` | ✅ ran (workaround) | + `--reasoning-parser mistral --tool-call-parser mistral` + config max_length 800 → 4096 |
| `Moonlight-16B-A3B-Instruct` | ✅ ran (workaround) | `--max-model-len 8192` (model max_position_embeddings hard cap) |
| `Ministral-3-14B-Instruct-2512` | ✅ ran (workaround) | corrected HF id (chunk7 used non-existent `Ministral-3-14B`) + mistral tokenizer trio + `--language-model-only` |
| `Llama-3.3-70B-NVFP4` | 🚫 dropped | operator deprecated weeks ago — excluded from cohort |

**Root cause of initial failures**: chunk-7 sweep scripts
(`/tmp/sequential_runs.sh`, `/tmp/swap_run.sh`) bypassed the
`autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` per-model flag compendium
and ran plain `docker run` with default flags. The compendium
(populated by Phase 2c on 2026-06-17) had already documented exactly
the flags each model needs. New AGENTS.md rule:
> Before dispatching any multi-model vLLM sweep on the DGX-GB10
> autoresearch slot, read `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md`.
> Never default-flag a model with a documented row.

### Failure-mode notes (raw, retained for archaeology)

### Failure-mode notes (raw)

+ **Gemma-4-26B-A4B**: `vllm/v1/core/encoder_cache_manager.py:302` raises
  `ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
  is larger than max_num_batched_tokens (2048). Please increase
  max_num_batched_tokens.` Pure config-knob issue.
  **FIXED**: `--max-num-batched-tokens 4096` + the rest of Phase 2c flags.
  Final result: KG 74.6% / 74.8%, GI 90.0% / 92.5%.
+ **Moonlight-16B-A3B**: first attempt failed with `User-specified
  max_model_len (32768) > derived max_model_len (max_position_embeddings=8192)`.
  Retried with `--max-model-len 8192`; container then hung loading MoE
  shards on `vllm:26.05-py3`. MoE backend selection (`FLASHINFER_CUTLASS`)
  may not handle this checkpoint cleanly.
  **FIXED**: `--max-model-len 8192` (model's hard cap) + Phase 2c flags.
  Final result: KG 61.2% / 57.5%, GI 63.7% / 56.2% (KG entities 83% — only
  cohort model below 100%, likely context cap).
+ **Ministral-3-14B**: container died before any model-loading log lines
  reached stdout. **Root cause**: sweep used HF id `mistralai/Ministral-3-14B`
  which doesn't exist; cache has `mistralai/Ministral-3-14B-Instruct-2512`.
  **FIXED**: corrected HF id + `--tokenizer-mode mistral --config-format
  mistral --load-format mistral --language-model-only`.
  Final result: KG 79.9% / 81.9%, GI 83.8% / 85.0%.
+ **Magistral-Small-2509**: container died at 10s with no captured error.
  **Root cause**: vendor REQUIRES the mistral tokenizer trio +
  `--reasoning-parser=mistral`; chunk-7 sweep ran with HF tokenizer
  and no parser flag. Additionally a reasoning model that emits
  `[THINK]/[/THINK]` and would have burned through `max_tokens=800`
  even on a successful boot.
  **FIXED**: `--reasoning-parser mistral --tokenizer-mode mistral
  --config-format mistral --tool-call-parser mistral --language-model-only`
  + eval config `max_length: 800 → 4096`. Final result: KG 72.4% / 72.4%,
  GI 90.0% / 87.5%.
+ **Mistral-Small-3.2-24B**: vLLM came up fine and KG run started.
  Generation ran at **~5 tok/s** vs ~50 tok/s for Qwen3-30B and ~80 tok/s
  for Qwen3.5-35B-A3B. **Root cause**: Mistral-3.2 is a vision-capable
  Pixtral model; vLLM loaded the vision tower despite our text-only
  workload, throttling inference.
  **FIXED**: `--language-model-only` (skips vision tower init) + the
  mistral tokenizer trio. Final result: **KG 82.8% / 81.9% (best in
  cohort)**, GI 91.2% / 85.0%.
+ **Llama-3.3-70B-NVFP4**: weights downloaded (~13min) + 9 shards loaded
  successfully + vLLM reached READY at 360s; KG completed 10 episodes in
  ~23min; GI was on episode 9/10 when operator stopped the run because
  Llama 70B has been deprecated for weeks. NOT a vLLM failure — operator
  decision. Dropped from cohort.

## Acceptance against RFC-097 §Success Criteria

1. **Full silver rebuild scoreboards published**: ✓ (this file) — all 9 in-cohort
   candidates re-run through v2 pipeline + per-model vLLM:26.05 workarounds;
   Llama-3.3-70B-NVFP4 dropped per operator decision.
2. **Migration scripts dry-run** — separate; chunk 6 ships the scripts.
3. **Grounding contract preserved**: ✓ — silvers report 99% (Opus) / 91%
   (Sonnet) quote verification; all 9 candidates (except DeepSeek
   genuinely-too-weak) emit 100%-grounded insights post-rerun.
4. **Schemas reject legacy** — deferred to chunk 9 follow-up PR (ADR-101,
   post 2-4-week bake window per RFC-097 §161).

## DGX operating-mode trail

Chunk 7 stayed in `research` mode throughout (autoresearch vLLM on `:8003`).
Verified at start: *"autoresearch vLLM up on :8003 / coder is down / GPU
util 0% / 3 compute apps"*. No `code` slot touched (operator's IDE
untouched).

## Files

+ `data/eval/references/silver/silver_{opus47,sonnet46}_{kg,gi}_dev_v1/` — regenerated
+ `data/eval/runs/autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_*` — re-run
+ `data/eval/runs/autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_*` — re-run
+ `data/eval/runs/autoresearch_prompt_vllm_mistral_small_3_2_24b_dev_*` — re-run (workaround)
+ `data/eval/runs/autoresearch_prompt_vllm_ministral_3_14b_dev_*` — re-run (workaround)
+ `data/eval/runs/autoresearch_prompt_vllm_magistral_small_2509_dev_*` — re-run (workaround)
+ `data/eval/runs/autoresearch_prompt_vllm_gemma_4_26b_a4b_dev_*` — re-run (workaround)
+ `data/eval/runs/autoresearch_prompt_vllm_moonlight_16b_a3b_dev_*` — re-run (workaround)
+ `data/eval/runs/autoresearch_prompt_gemini_gemini25_flash_lite_dev_*` — re-run
+ `data/eval/runs/chunk7_silver_regen/*.log` — full run logs (incl. failure logs for Gemma/Moonlight/Ministral/Magistral/Mistral-3.2/Llama-70B)
