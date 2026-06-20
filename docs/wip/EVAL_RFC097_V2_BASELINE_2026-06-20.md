# RFC-097 v2.0 baseline — silver rebuild + scoreboard re-baseline (2026-06-20)

**Status**: complete (chunk 7 of RFC-097)
**Branch**: `feat/corpus-ontology-v2`
**Dataset**: `curated_5feeds_dev_v1` (10 episodes)
**Silver models**: Claude Opus 4.7 + Sonnet 4.6 (Anthropic; vendor-disjoint from the entire candidate cohort — #939 lesson preserved)
**Candidate cohort**: 10 KG + 10 GI extractors via DGX-vLLM (autoresearch slot) + 1 Gemini

## Silver regen results

All four dev_v1 silvers regenerated against the v2/v3-emitting pipeline.
Stats (n_episodes always 10):

| Silver | Model | Topics / Insights | Entities verified |
|---|---|---|---|
| `silver_opus47_kg_dev_v1` | claude-opus-4-7 | 104 topics | 30/30 |
| `silver_opus47_gi_dev_v1` | claude-opus-4-7 | 80 insights | quotes 83/84 (99%) |
| `silver_sonnet46_kg_dev_v1` | claude-sonnet-4-6 | 97 topics | 30/30 |
| `silver_sonnet46_gi_dev_v1` | claude-sonnet-4-6 | 80 insights | quotes 87/96 (91%) |

Delta vs prior:

- silver_opus47_kg_dev_v1: 103 → 104 topics (+1), entities flat at 30
- silver_opus47_gi_dev_v1: GI shape now carries v3 vocab (claim/recommendation/observation/question)
- silver_sonnet46_*: regenerated with the canonical `claude-sonnet-4-6` model id (was `claude-sonnet-4-20250514` previously)

Per-model topic vocab + entity surface set are refreshed; ID derivation
unchanged (canonical `person:` / `org:` / `topic:` slugs).

## KG scoreboard — overall weighted coverage (topics + entities + claims)

Coverage = fraction of silver nodes that have a candidate node with
cosine similarity ≥ 0.65 (via `all-MiniLM-L6-v2`).

| Candidate | vs silver_opus47 | vs silver_sonnet46 |
|---|---:|---:|
| `gemini_gemini25_flash_lite` | **52.2%** | **50.4%** |
| `vllm_qwen3_5_35b_a3b` | **48.5%** | **45.7%** |
| `vllm_magistral_small_2509` | 40.3% | 37.0% |
| `vllm_mistral_small_3_2_24b` | 39.6% | 38.6% |
| `vllm_ministral_3_14b` | 33.6% | 35.4% |
| `vllm_qwen3_30b_a3b_instruct_2507` | 32.8% | 33.9% |
| `vllm_moonlight_16b_a3b` | 31.3% | 26.8% |
| `vllm_gemma_4_26b_a4b` | 30.6% | 32.3% |
| `vllm_llama_3_3_70b_nvfp4` | 20.1% | 21.3% |
| `vllm_deepseek_v2_lite_chat` | 0.0% | 0.0% |

### KG topic vs entity breakdown (vs `silver_opus47_kg_dev_v1`)

| Candidate | Topics covered | Entities covered |
|---|---:|---:|
| `vllm_qwen3_5_35b_a3b` | **62.5%** | 0.0% |
| `gemini_gemini25_flash_lite` | 59.6% | **26.7%** |
| `vllm_magistral_small_2509` | 51.9% | 0.0% |
| `vllm_mistral_small_3_2_24b` | 51.0% | 0.0% |
| `vllm_ministral_3_14b` | 43.3% | 0.0% |
| `vllm_qwen3_30b_a3b_instruct_2507` | 42.3% | 0.0% |
| `vllm_gemma_4_26b_a4b` | 39.4% | 0.0% |
| `vllm_moonlight_16b_a3b` | 31.7% | 30.0% |
| `vllm_llama_3_3_70b_nvfp4` | 26.0% | 0.0% |
| `vllm_deepseek_v2_lite_chat` | 0.0% | 0.0% |

**Observation**: Only Gemini and Moonlight produce non-zero entity
coverage. The other vLLM-served candidates extract entities but the
embeddings don't match — likely because the candidate Entity-class
nodes are named differently than the silver's. Worth a candidate-side
emission audit (separate workstream).

## GI scoreboard — insight coverage (insight-to-insight semantic match)

Coverage = fraction of silver insights with at least one candidate
insight at cosine ≥ 0.65.

| Candidate | vs silver_opus47 | vs silver_sonnet46 |
|---|---:|---:|
| `gemini_gemini25_flash_lite` | **72.5%** | **72.5%** |
| `vllm_gemma_4_26b_a4b` | 42.5% | 45.0% |
| `vllm_qwen3_5_35b_a3b` | 37.5% | 41.2% |
| `vllm_qwen3_30b_a3b_instruct_2507` | 37.5% | 35.0% |
| `vllm_mistral_small_3_2_24b` | 25.0% | 35.0% |
| `vllm_ministral_3_14b` | 25.0% | 30.0% |
| `vllm_magistral_small_2509` | 25.0% | 22.5% |
| `vllm_llama_3_3_70b_nvfp4` | 16.2% | 15.0% |
| `vllm_moonlight_16b_a3b` | 16.2% | 15.0% |
| `vllm_deepseek_v2_lite_chat` | 0.0% | 0.0% |

## Acceptance against RFC-097 §Success Criteria

1. **Full silver rebuild scoreboards published**: ✓ (this file)
2. **Migration scripts dry-run** — separate; chunk 6 ships the scripts.
   Operator can `python scripts/migrate_kg_entity_to_person_org.py
   --corpus .test_outputs/manual/prod-v2/corpus` whenever convenient.
3. **Grounding contract preserved**: ✓ — silver_opus47_gi_dev_v1
   reports 83/84 quotes verified (99%); silver_sonnet46 87/96 (91%).
   Every silver insight has at least one verbatim quote where verified.
4. **Schemas reject legacy** — chunk 9 gate (ADR-101, post bake).

## Notes & gotchas

- DGX swap rule (per `feedback_never_use_coder_next`): chunk 7 ran with
  DGX in `research` mode the entire time (autoresearch vLLM up on
  `:8003`). No `code` mode touched. `gpu-mode-swap.sh status` at start:
  *"autoresearch vLLM up on :8003 / coder is down / GPU util 0% /
  3 compute apps"*.
- Silver/judge vendor disjoint from all candidates: ✓ — silvers are
  Anthropic, candidates are all non-Anthropic (Gemini + open-source
  vLLM-served). No #939 vendor-bias to compensate for.
- KG entity coverage cliff (most candidates at 0%) is the most actionable
  finding from this v2 baseline — recommend a follow-up that audits
  candidate-side entity emission shapes before the next sweep.

## Reproduction

```bash
# Confirm DGX research mode
ssh dgx-llm-1 "~/bin/gpu-mode-swap.sh status"

# Regenerate silvers (KG)
.venv/bin/python scripts/eval/data/generate_kg_silver.py \
  --dataset curated_5feeds_dev_v1 --output-id silver_opus47_kg_dev_v1 --model claude-opus-4-7
.venv/bin/python scripts/eval/data/generate_kg_silver.py \
  --dataset curated_5feeds_dev_v1 --output-id silver_sonnet46_kg_dev_v1 --model claude-sonnet-4-6

# Regenerate silvers (GI)
.venv/bin/python scripts/eval/data/generate_gi_silver.py \
  --dataset curated_5feeds_dev_v1 --output-id silver_opus47_gi_dev_v1 --model claude-opus-4-7
.venv/bin/python scripts/eval/data/generate_gi_silver.py \
  --dataset curated_5feeds_dev_v1 --output-id silver_sonnet46_gi_dev_v1 --model claude-sonnet-4-6

# Re-score every candidate vs both silvers (see /tmp/rescore_v2.sh in this branch)
# Per-run metrics_vs_silver_*.json are gitignored under data/eval/runs/
```

End-to-end wall clock: ~9 minutes (silver regen ~5 min in parallel +
40-pair scoring ~4 min).
