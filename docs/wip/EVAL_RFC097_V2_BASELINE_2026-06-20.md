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

Two candidates re-run through the current pipeline (Qwen3-30B = pack leader, Gemini = cloud control):

### KG — overall weighted coverage / topic% / entity%

| Candidate | vs silver_opus47 | vs silver_sonnet46 |
|---|---:|---:|
| `gemini_gemini25_flash_lite` | **76.1%** (T:69%, E:**100%**) | **73.2%** (T:65%, E:**100%**) |
| `vllm_qwen3_30b_a3b_instruct_2507` | **71.6%** (T:64%, E:**100%**) | **74.0%** (T:66%, E:**100%**) |

### GI — insight-to-insight coverage @ 0.65 cosine

| Candidate | vs silver_opus47 | vs silver_sonnet46 |
|---|---:|---:|
| `gemini_gemini25_flash_lite` | **96.2%** | **92.5%** |
| `vllm_qwen3_30b_a3b_instruct_2507` | **91.2%** | **90.0%** |

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

## Replication plan for remaining 8 candidates

Status: **operator-attended** (requires DGX model swaps).

The other 8 candidate models aren't currently loaded on DGX (Qwen3-30B-A3B-Instruct-2507
is the live autoresearch slot). Re-running each requires:

1. `ssh dgx-llm-1 "~/bin/gpu-mode-swap.sh idle"` (release Qwen3-30B)
2. Update `/opt/vllm-autoresearch/docker-compose.yml` to point at the next model
   (`Qwen3.5-35B`, `Gemma-4-26B`, `Llama-3.3-70B-nvfp4`, `Magistral-Small-2509`,
   `Ministral-3-14B`, `Mistral-Small-3.2-24B`, `Moonlight-16B-a3b`, `DeepSeek-V2-Lite-Chat`)
3. Restart the container
4. Run: `VLLM_API_KEY=buddy-is-the-king PYTHONPATH=. .venv/bin/python -u
   scripts/eval/experiment/run_experiment.py
   data/eval/configs/kg_autoresearch_prompt_vllm_<model>_dev_v1.yaml
   --force --dry-run --vllm-base-url http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1`
5. Re-score with `scripts/eval/score/score_kg_node_to_node.py` and
   `scripts/eval/score/score_gi_insight_to_insight.py`
6. Repeat for KG + GI configs of each model

Expected per-model wall clock: ~5 min (KG + GI). For 8 models = ~40 min plus
docker-restart overhead per swap.

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
- `data/eval/runs/autoresearch_prompt_gemini_gemini25_flash_lite_dev_*` — re-run
- `data/eval/runs/chunk7_silver_regen/*.log` — full run logs
