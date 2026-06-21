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
|---|---|---|---|
| `vllm_qwen3_30b_a3b_instruct_2507` | 0% → **100%** | 32.8% → **71.6%** (+38.8) | 37.5% → **91.2%** (+53.7) |
| `vllm_qwen3_5_35b_a3b` | 0% → **100%** | 48.5% → **82.1%** (+33.6) | 37.5% → **85.0%** (+47.5) |
| `gemini_gemini25_flash_lite` | 26.7% → **100%** | 52.2% → **76.1%** (+23.9) | 72.5% → **96.2%** (+23.7) |

The fix wasn't a prompt change — it was the chunks 3-5 v2 emit pipeline (Person/Organization
typed nodes, edge_class metadata, position_hint waterfall, insight_type vocab alignment).
Old cohort runs scored against new silvers under-represented because the emission shapes
mismatched the silver schema. Re-running through today's pipeline aligns both sides.

## Silver regen results (2026-06-20)

All four dev_v1 silvers regenerated against the v2/v3-emitting pipeline.
Stats (n_episodes always 10):

| Silver | Model | Topics / Insights | Entities verified |
|---|---|---|---|
| `silver_opus47_kg_dev_v1` | claude-opus-4-7 | 104 topics | 30/30 |
| `silver_opus47_gi_dev_v1` | claude-opus-4-7 | 80 insights | quotes 83/84 (99%) |
| `silver_sonnet46_kg_dev_v1` | claude-sonnet-4-6 | 97 topics | 30/30 |
| `silver_sonnet46_gi_dev_v1` | claude-sonnet-4-6 | 80 insights | quotes 87/96 (91%) |

## Updated v2-pipeline scoreboard

Three candidates re-run through the current pipeline (Qwen3-30B = pack leader,
Qwen3.5-35B = second pack leader, Gemini = cloud control). Llama-3.3-70B-NVFP4
dropped from the cohort per operator decision (deprecated weeks ago — not
re-run despite weights being on DGX).

### KG — overall weighted coverage / topic% / entity%

| Candidate | vs silver_opus47 | vs silver_sonnet46 |
|---|---:|---:|
| `vllm_qwen3_5_35b_a3b` | **82.1%** (T:77%, E:**100%**) | **79.5%** (T:73%, E:**100%**) |
| `gemini_gemini25_flash_lite` | **76.1%** (T:69%, E:**100%**) | **73.2%** (T:65%, E:**100%**) |
| `vllm_qwen3_30b_a3b_instruct_2507` | **71.6%** (T:64%, E:**100%**) | **74.0%** (T:66%, E:**100%**) |

### GI — insight-to-insight coverage @ 0.65 cosine

| Candidate | vs silver_opus47 | vs silver_sonnet46 |
|---|---:|---:|
| `gemini_gemini25_flash_lite` | **96.2%** | **92.5%** |
| `vllm_qwen3_30b_a3b_instruct_2507` | **91.2%** | **90.0%** |
| `vllm_qwen3_5_35b_a3b` | **85.0%** | **87.5%** |

## Stale-cohort scoreboard (2026-06-17 runs, pre-chunk-3 pipeline)

Retained for reference. **These are NOT the v2 baseline** — they show the
shape-mismatch artifact, not candidate quality:

| Candidate | KG opus47 | KG sonnet46 | GI opus47 | GI sonnet46 |
|---|---:|---:|---:|---:|
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

## Sweep outcomes — remaining candidates (2026-06-21)

Sequential autonomous sweep across the 8 remaining candidates completed.
Three succeeded (Qwen3.5-35B + DeepSeek-V2-Lite-Chat from earlier rerun
+ implicit Qwen3-30B/Gemini pack leaders). Five failed to load or run on
`vllm:26.05-py3`. Llama-3.3-70B dropped per operator decision.

| Candidate | Outcome | Failure mode |
|---|---|---|
| `Qwen3.5-35B-A3B` | ✅ ran | — |
| `DeepSeek-V2-Lite-Chat` | ✅ ran (weak: 1.5% KG / 3.8% GI) | model genuinely weak; not a load failure |
| `Llama-3.3-70B-NVFP4` | 🚫 dropped | operator deprecated weeks ago — excluded from cohort |
| `Mistral-Small-3.2-24B` | ✗ killed at 5 tok/s | Pixtral multimodal arch — generation throughput unusable on a single GB10 |
| `Gemma-4-26B-A4B` | ✗ container died at 20s | `max_tokens_per_mm_item (2496) > max_num_batched_tokens (2048)` — multimodal config rejected |
| `Magistral-Small-2509` | ✗ container died at 10s | startup fault (no log captured before death) |
| `Moonlight-16B-A3B` | ✗ container died at 20s | `max_model_len` mismatch: derived 8192 < requested 32768; corrected to 8192 but still hung loading MoE shards |
| `Ministral-3-14B` | ✗ container died at sweep start | MoE init fault on `vllm:26.05-py3` |

**Pattern**: every failing model is **multimodal (Pixtral/Gemma)** or **MoE
on the new `vllm:26.05-py3` image**. These same models worked on the
previous `vllm:25.11-py3` image (per 2026-06-17 cohort logs that DID produce
output — even if at low scores against then-current silvers). The 26.05
upgrade landed 2026-06-15 with the autoresearch slot move; the chunk 7
sweep is the first time we exercised the new image against the full
candidate matrix.

**Next-step options to discuss**:
1. Pin a `26.05` workaround per-model (longer wait + correct `--max-model-len`
   + `--max-num-batched-tokens` overrides for multimodal). Some failures
   look like vLLM config gates rather than architecture incompatibilities.
2. Stand up a fallback `25.11-py3` slot for the multimodal/MoE cohort and
   route their experiments there.
3. Drop the failing models from the candidate matrix entirely.

See `## Failure-mode notes` below for per-model detail.

### Failure-mode notes (raw)

- **Gemma-4-26B-A4B**: `vllm/v1/core/encoder_cache_manager.py:302` raises
  `ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
  is larger than max_num_batched_tokens (2048). Please increase
  max_num_batched_tokens.` Pure config-knob issue — solvable with
  `--max-num-batched-tokens 4096` (probably).
- **Moonlight-16B-A3B**: first attempt failed with `User-specified
  max_model_len (32768) > derived max_model_len (max_position_embeddings=8192)`.
  Retried with `--max-model-len 8192`; container then hung loading MoE
  shards on `vllm:26.05-py3`. MoE backend selection (`FLASHINFER_CUTLASS`)
  may not handle this checkpoint cleanly.
- **Ministral-3-14B**: container died before any model-loading log lines
  reached stdout. MoE init suspected by analogy.
- **Magistral-Small-2509**: container died at 10s with no captured error.
  Same image and base flags as Ministral/Moonlight — likely same MoE class
  of fault.
- **Mistral-Small-3.2-24B**: vLLM came up fine and KG run started.
  Generation ran at **~5 tok/s** vs ~50 tok/s for Qwen3-30B and ~80 tok/s
  for Qwen3.5-35B-A3B. Multimodal Pixtral architecture overhead is the
  suspect — Mistral-Small-3.2 is a vision-capable model and vLLM probably
  loaded the vision tower despite our text-only workload.
- **Llama-3.3-70B-NVFP4**: weights downloaded (~13min) + 9 shards loaded
  successfully + vLLM reached READY at 360s; KG completed 10 episodes in
  ~23min; GI was on episode 9/10 when operator stopped the run because
  Llama 70B has been deprecated for weeks. NOT a vLLM failure — operator
  decision.

## Acceptance against RFC-097 §Success Criteria

1. **Full silver rebuild scoreboards published**: ✓ (this file)
2. **Migration scripts dry-run** — separate; chunk 6 ships the scripts.
3. **Grounding contract preserved**: ✓ — silvers report 99% (Opus) / 91%
   (Sonnet) quote verification; Qwen3-30B + Gemini both emit 100%-grounded
   insights post-rerun.
4. **Schemas reject legacy** — chunk 9 gate (ADR-101, post bake).

## DGX operating-mode trail

Chunk 7 stayed in `research` mode throughout (autoresearch vLLM on `:8003`).
Verified at start: *"autoresearch vLLM up on :8003 / coder is down / GPU
util 0% / 3 compute apps"*. No `code` slot touched (operator's IDE
untouched).

## Files

- `data/eval/references/silver/silver_{opus47,sonnet46}_{kg,gi}_dev_v1/` — regenerated
- `data/eval/runs/autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_*` — re-run
- `data/eval/runs/autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_*` — re-run
- `data/eval/runs/autoresearch_prompt_gemini_gemini25_flash_lite_dev_*` — re-run
- `data/eval/runs/chunk7_silver_regen/*.log` — full run logs (incl. failure logs for Gemma/Moonlight/Ministral/Magistral/Mistral-3.2/Llama-70B)
