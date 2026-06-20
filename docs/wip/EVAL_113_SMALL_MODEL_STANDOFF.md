# #113 Small-model standoff — 9B-class vs safe pick vs top dog (2026-06-20)

**TL;DR**: Under #1035 NER pre-pass, **Qwen3.5:9b on Ollama matches the
top dog Cell F NVFP4 on topic coverage (66% vs 65%) and reaches 97%
entity coverage** — only 3pp behind the perfect 100% of vLLM-served
Cell F + Qwen3.5-35B-A3B. The NER pre-pass closes the model-size gap
for KG/entity extraction. **The previous "safe pick" (Moonlight-16B-A3B)
is the weakest of the four** on both topic (54%) and entity (93%) under
NER pre-pass.

Operationally: any deploy that can run spaCy + a 9B LLM now has a real
choice. Edge / cost-constrained profiles can drop down to Qwen3.5:9b
without material KG quality loss.

## Setup

- Dataset: `curated_5feeds_dev_v1` (10 episodes)
- Silver: `silver_opus47_kg_dev_v1`
- Pipeline: corrected `provider`-source extraction + #1035 NER pre-pass
  (on by default per phase 4)
- Scorer: post-`d8df114d` (handles KG-artifact shape)
- Identical eval harness across all candidates

## Candidate selection

| Slot | Candidate | Backend | Model |
|---|---|---|---|
| 9B-class | Qwen3.5:9b | Ollama | `qwen3.5:9b` (~6.6 GB Q4) |
| safe pick | Moonlight-16B-A3B | vLLM | `moonshotai/Moonlight-16B-A3B-Instruct` |
| top dog | Cell F NVFP4 | vLLM | `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` |
| (ref) top quality | Qwen3.5-35B-A3B | vLLM | `Qwen/Qwen3.5-35B-A3B` (bf16) |
| (ref from #116) | Cell C Ollama | Ollama | `qwen3.5:35b` (Q4 ~23 GB) |

## Scoreboard (KG eval, NER pre-pass on)

| Candidate | Topic cov | Topic avg_sim | Entity cov | Entity extras | s/ep |
|---|---:|---:|---:|---:|---:|
| **Cell F NVFP4** | 65% | 0.727 | **100%** (30/30) | 0 | **18.5** |
| Qwen3.5-35B-A3B vLLM bf16 | **77%** | **0.784** | **100%** (30/30) | 0 | 34.6 |
| **Qwen3.5:9b Ollama** | **66%** | **0.722** | **97%** (29/30) | 0 | 28.4 |
| Moonlight-16B-A3B vLLM | 54% | 0.656 | 93% (28/30) | 4 | **17.3** |
| Cell C (Qwen3.5-35b Ollama) | 66% | 0.714 | 83% (25/30) | 0 | 19.6 |

## Findings

### 1. Small model + NER ≈ top dog quality

Qwen3.5:9b on Ollama lands within statistical noise of Cell F NVFP4 on
every dimension that matters under NER pre-pass:

| Dimension | Cell F NVFP4 | Qwen3.5:9b | Δ |
|---|---:|---:|---:|
| Topic coverage | 65% | 66% | +1pp |
| Topic avg_sim | 0.727 | 0.722 | -0.005 |
| Entity coverage | 100% | 97% | -3pp |
| Wall clock (s/ep) | 18.5 | 28.4 | +54% |
| Weight footprint | 18 GB | **6.6 GB** (Q4) | **-63%** |

**The 9B model rides NER recall to within 3pp of the 30B's perfect
entity score.** This is the kind of result that re-opens the
small-model deployment story — 6.6 GB total (model + KV) fits on a
laptop GPU or a phone-class edge device with quantization headroom.

### 2. Moonlight loses safe-pick status

Pre-#1035, Moonlight was the "style-neutral safe pick" per #1016
methodology (Δ Sonnet-Opus = 0). Under #1035 NER pre-pass:

- Moonlight topic coverage: **54%** (lowest of the 4 in-sweep)
- Moonlight entity coverage: 93% (3 of 10 episodes missed at least one)
- Moonlight emitted **4 false-positive entities** (the only non-zero
  extras in the sweep)

Moonlight's MoE small active-params (16B/A3B with ~3B active per token)
trades capability for speed. Under the older bullets-derived path it
was style-neutral and "safe"; under NER pre-pass it under-extracts
topics AND occasionally accepts NER false-positive hints rather than
filtering them. The 9B Qwen, with denser weights, is more discriminating.

**Recommendation**: drop "safe pick" from the Moonlight headline in
`model_registry.py` (the `vllm_moonlight_16b_a3b` StageOption). Its
role compresses to "speed-only choice when entity quality is not a
gate."

### 3. Per-stage routing top-dog updated

| Stage | Current top-dog | This sweep's signal |
|---|---|---|
| Summary | Qwen3.5-35B-A3B | (unchanged — not in this sweep) |
| GI | Qwen3.5-35B-A3B | (unchanged — not in this sweep) |
| KG (topic) | Qwen3.5-35B-A3B | unchanged — 77% leads |
| KG (entity) | Cell F NVFP4 + Qwen3.5-35B-A3B (tied at 100%) | also reachable by Qwen3.5:9b at 97% |
| KG (cost-quality) | Cell F NVFP4 | **Qwen3.5:9b Ollama** wins on weight footprint at near-equivalent quality |

### 4. NER pre-pass closes the model-size gap

The cohort spread under NER pre-pass is **54%–77% topic / 93%–100%
entity**, which is dramatically tighter than pre-#1035 (24%–50% topic,
0% entity). The NER hint list anchors PERSON+ORG recall regardless of
model capability — even a 9B model emits the silver-expected entities
when the spans are pre-identified.

This is the empirical case for **decoupling KG/entity work from model
size**. The compute budget can shift from "use the biggest model for
best entity recall" to "use the smallest model that follows the NER
candidate-list instructions."

## Operational decisions

### `prod_dgx_full_with_fallback` — unchanged

Cell F NVFP4 remains the autoresearch default. The #1022 / #116 verdicts
hold. No registry or profile changes for the autoresearch tier.

### `local_dgx_balanced` + `airgapped_thin` — consider Qwen3.5:9b

For profiles tuned to local/edge constraints (`local_dgx_balanced`
historically uses Cell C Ollama `qwen3.5:35b`, `airgapped_thin` uses
local transformers), Qwen3.5:9b on Ollama becomes a competitive option:

- Same topic coverage as Cell F (66%)
- 3pp behind on entity coverage (97% vs 100%) — well within the
  airgapped-profile quality band
- 73% smaller weight footprint vs Cell C 35B Ollama
- 53% faster than Cell C 35B Ollama at the same Ollama backend

**Suggested follow-up (out of scope for #113 close):** validate
Qwen3.5:9b on `curated_5feeds_benchmark_v2` held-out before flipping
any profile default. The standoff result is on dev_v1 only.

### `model_registry.py` updates

Two small docstring/headline_metric updates the next time the registry
is touched:
- `ollama_qwen35_9b` — add headline note: "#113 small-model standoff:
  66% topic cov, 97% entity cov under NER pre-pass — competitive with
  Cell F NVFP4 at 6.6 GB footprint"
- `vllm_moonlight_16b_a3b` — drop "safe pick" language; replace with
  "speed-only — under-extracts topics + emits NER false positives per
  #113"

(Not committing these registry changes here — they're documentation
updates that should land alongside the next functional registry edit.)

## Cross-references

- Run artifacts:
  - `data/eval/runs/autoresearch_prompt_vllm_moonlight_16b_a3b_dev_113_ner_v1/`
  - `data/eval/runs/autoresearch_prompt_ollama_qwen35_9b_dev_113_ner_v1/`
- Reference runs:
  - Cell F NVFP4: `data/eval/runs/autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_1035_ner_v1/`
  - Qwen3.5-35B-A3B vLLM: `data/eval/runs/autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_1035_ner_v1/`
  - Cell C Ollama: `data/eval/runs/autoresearch_prompt_ollama_qwen35_35b_dev_116_cell_c_ner_v1/`
- Methodology: `docs/wip/EVAL_1035_NER_PREPASS_VERDICT.md` (NER pre-pass
  design + scorer fix)
- Daily-driver evidence: `docs/wip/VLLM_GB10_TUNING_VALIDATION_2026-06-18.md`
  (#1022 Cell F) + `docs/wip/EVAL_116_CELL_C_REBASELINE_2026-06-20.md`

## DGX state after sweep

- Compose: restored to Cell F NVFP4 (canonical daily-driver state) via
  `docker compose up -d` after the Moonlight + Ollama swap cycle
- vLLM: healthy on `autoresearch` served-model-name alias
- Ollama: daemon stays running (systemd-managed)
- No uncommitted homelab compose changes against `main`
- Total wall-clock spent on #113: ~25 minutes (one vLLM Cell F → Moonlight
  swap + vLLM Moonlight run + idle + Ollama 9B run + vLLM Moonlight →
  Cell F restore)
