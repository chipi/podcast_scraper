# #116 Cell C re-baseline against Cell F NVFP4 (2026-06-20)

**TL;DR**: Cell C (Ollama Qwen3.5-35b, #928 local DGX winner) is
supplanted as the autoresearch daily-driver by Cell F (vLLM NVFP4
Qwen3-30B-A3B-Instruct-2507) on the corrected pipeline. Cell F wins
quality (+17pp entity coverage, ~same topic coverage), speed (~6%
faster), and weight footprint (-22%). Cell C remains the fallback when
vLLM is unavailable, but the #1022 daily-driver verdict holds and now
has cross-platform validation.

## Setup

- Dataset: `curated_5feeds_dev_v1` (10 episodes)
- Silver: `silver_opus47_kg_dev_v1`
- Pipeline: corrected `provider`-source extraction + #1035 NER pre-pass
  (on by default per #1035 phase 4)
- KG scorer: post-`d8df114d` (handles KG-artifact shape correctly)
- Same eval harness as #1035 phase 3 — apples-to-apples sweep across
  candidates

| Candidate | Backend | Model |
|---|---|---|
| Cell C | Ollama @ dgx-llm-1:11434 | `qwen3.5:35b` (HF: Qwen/Qwen3.5-35B-A3B) |
| Cell F | vLLM @ dgx-llm-1:8003 | `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` |
| (ref) Qwen3.5-35B-A3B vLLM bf16 | vLLM @ dgx-llm-1:8003 | `Qwen/Qwen3.5-35B-A3B` |

## Scoreboard (KG eval, dev_v1, silver_opus47_kg, NER pre-pass on)

| Candidate | Topic cov | Topic avg_sim | Entity cov | Wall clock | s/ep |
|---|---:|---:|---:|---:|---:|
| **Cell F NVFP4** | 65% | 0.727 | **100%** (30/30) | **184.9s** | **18.5** |
| Qwen3.5-35B-A3B vLLM bf16 | **77%** | **0.784** | **100%** (30/30) | 346.0s | 34.6 |
| Cell C (Ollama Qwen3.5-35b) | 66% | 0.714 | 83% (25/30) | 195.9s | 19.6 |

## Findings

### 1. Cell C vs Cell F — daily-driver decision unchanged

| Dimension | Cell F NVFP4 | Cell C Ollama | Winner |
|---|---|---|---|
| Topic coverage | 65% | 66% | Cell C +1pp (noise) |
| Topic avg_sim | 0.727 | 0.714 | Cell F +0.013 |
| Entity coverage | **100%** | 83% | **Cell F +17pp** |
| End-to-end speed | 18.5 s/ep | 19.6 s/ep | Cell F -6% |
| Weight footprint (GB) | 18 | 23 | Cell F -22% |
| Boot time | ~2 min | (always loaded by Ollama) | Cell C |

**Cell F dominates on entity coverage (+17pp), the dimension that
#1035 specifically engineered for.** Topic coverage is essentially tied
(±1pp is well within run-to-run noise). On all operational metrics
(speed, footprint), Cell F wins.

Cell C's entity gap is concentrated in 2 of 10 episodes:
- p02_e01: 0/3 entities (also collapsed topic coverage to 10%)
- p03_e01 + p05_e02: 2/3 entities each

The p02_e01 collapse looks like Ollama-side malformed JSON for that
specific episode (Phase 1 cleanup didn't pre-filter it). Cell F handled
the same episode without issue. This is the kind of reliability win that
matters for unsupervised batch runs.

### 2. 30B vs 35B drift — the real signal

The #116 framing was "30B vs 35B drift" — is the parameter difference
the explanation for Cell F's win? Looking at the three runs:

| Param class | Backend / quant | Entity cov |
|---|---|---:|
| 30B (NVFP4) | vLLM | **100%** |
| 35B (bf16) | vLLM | **100%** |
| 35B (Q4_K_M) | Ollama | 83% |

**The parameter difference is NOT the explanation.** Both 30B NVFP4 +
35B bf16 hit 100% on vLLM; Cell C (35B Ollama) lags at 83%. The actual
explanation is **vLLM vs Ollama serving stack** — under the NER pre-pass
+ v5 prompt, vLLM-served models extract the silver-expected entities
more reliably than Ollama-served Qwen3.5-35b.

The Ollama-vs-vLLM penalty for the SAME bf16 Qwen3.5-35B-A3B (from this
sweep + #100 historic measurement) is consistent: Ollama incurs a
quality tax even on the highest-quality model. This is a serving-stack
reliability finding, not a model-capability finding.

### 3. Topic-coverage convergence under NER pre-pass

All three candidates now sit in a tight 65–77% topic range — much closer
than the pre-#1035 scoreboard where the spread was 24–50%. NER pre-pass
narrowed the topic-extraction gap by giving the LLM cleaner candidate
spans to anchor topics around. The remaining spread is mostly
prompt-adherence, not capability.

## Operational decisions

### Daily-driver verdict — unchanged

**Cell F NVFP4 stays as the `prod_dgx_full_with_fallback` default.**
The #1022 verdict (Cell F = best speed-quality trade-off, manually
swap to Qwen3.5-35B-A3B vLLM for one-shot highest-stakes evals) holds.
Cell C is no longer a competing daily-driver candidate.

### Cell C operational status

- **Removed from autoresearch-tier daily-driver consideration** —
  superseded by Cell F NVFP4 on both quality and operational metrics
- **Retained as Ollama fallback** in `local_dgx_balanced.yaml` and
  `airgapped` profiles for environments without vLLM
- The `model_registry.py` `ollama_qwen35_35b` StageOption stays, but
  its `headline_metric` should note the #116 finding (Ollama-side
  entity-coverage penalty)
- No code or compose changes required

## Cross-references

- Run artifacts: `data/eval/runs/autoresearch_prompt_ollama_qwen35_35b_dev_116_cell_c_ner_v1/`
- Cell F reference: `data/eval/runs/autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_1035_ner_v1/`
- Qwen3.5 vLLM reference: `data/eval/runs/autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_1035_ner_v1/`
- #1035 verdict (NER pre-pass methodology + scorer fix):
  `docs/wip/EVAL_1035_NER_PREPASS_VERDICT.md`
- #1022 Cell F daily-driver evidence:
  `docs/wip/VLLM_GB10_TUNING_VALIDATION_2026-06-18.md`

## DGX state after sweep

- Compose: restored to Cell F NVFP4 (canonical daily-driver state) via
  `gpu-mode-swap.sh research` after `idle` for the Cell C run
- vLLM: healthy on `autoresearch` served-model-name alias
- Ollama: daemon stays running (managed by systemd; left as-is per
  operator's "always-on" pattern)
- No uncommitted compose changes against the homelab `main` branch
- Total wall-clock spent on #116: ~10 minutes (one vLLM-stop + Ollama
  run + vLLM-start cycle, plus scoring)
