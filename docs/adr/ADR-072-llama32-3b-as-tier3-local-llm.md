# ADR-072: Llama 3.2:3b as Tier 3 Local LLM

- **Status**: Accepted
- **Date**: 2026-04-05
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md),
  [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md)
- **See Also**: [ADR-069](ADR-069-hybrid-ml-pipeline-as-production-direction.md),
  [ADR-071](ADR-071-four-tier-summarization-strategy.md)

## Context & Problem Statement

RFC-057 Track B required selecting the best Ollama model for the LLM reduce stage of the
hybrid pipeline (and later, direct LLM inference). Multiple models were evaluated across
the 3B–12B parameter range. The selection needed to balance ROUGE-L quality, latency,
and local resource constraints (Apple Silicon MPS, no GPU).

## Decision

Use **`llama3.2:3b`** (Llama 3.2 3B Instruct) as the Tier 3 local LLM for direct
podcast summarization.

## Evidence

### Model comparison sweep (smoke dataset, 5 episodes)

All models used temperature=0.5, top_p=1.0 (hybrid sweep winners) as REDUCE stage of
`BART-base MAP + Ollama REDUCE` pipeline.

| Model | Params | ROUGE-L | Embed | Latency | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **llama3.2:3b** | **3B** | **23.7%** | **72.9%** | **15.7s** | **Winner** |
| mistral-nemo:12b | 12B | 19.4% | 71.4% | 48.2s | Worse quality at 4x latency |
| qwen2.5:7b | 7B | 20.1% | 70.8% | 28.4s | Generic text style mismatch |
| llama3.1:8b | 8B | 18.8% | 69.3% | 32.1s | Instruction-following weaker |
| mistral:7b | 7B | 17.2% | 68.1% | 25.3s | Older instruct tuning |

### Direct inference (Tier 3 benchmark, 10 episodes)

After establishing that the MAP stage is lossy (see ADR-071), `llama3.2:3b` was evaluated
in direct mode with autoresearch-tuned parameters.

```text
Config:   direct_llama32_3b_autoresearch_v1.yaml
Dataset:  curated_5feeds_benchmark_v1 (10 episodes)
Ref:      silver_sonnet46_benchmark_v1
```

| Param | Hybrid sweep winner | Direct sweep winner |
| :--- | :---: | :---: |
| temperature | 0.5 | **0.3** |
| max_tokens / max_length | 1000 | 1000 |
| ROUGE-L | 23.7% (hybrid) | **26.4%** (direct) |
| Embed | 72.9% | **79.5%** |
| Latency | 15.7s/ep | **7.5s/ep** |

Direct inference with temperature=0.3 outperforms the hybrid on all metrics.

## Why Instruction-Following Beats Size

The key finding from the sweep is that **3B beats 7-12B models for this task**. The
mechanism is instruction-following quality, not raw capacity:

1. **Task is structured extraction, not open-ended generation.** Podcast summarization
   requires following a format contract (paragraph or bullet JSON), respecting length
   bounds, and honouring "no invented facts" and "use transcript vocabulary" constraints.
   Instruction-tuned 3B models outperform larger models with weaker instruction tuning.

2. **Llama 3.2:3b is specifically optimised for edge/local deployment.** Meta's training
   for this model emphasises instruction-following fidelity at small scale. Larger
   general-purpose models (Mistral-Nemo 12B, Qwen 2.5 7B) have more capacity but are
   not trained to the same instruction-compliance standard for constrained tasks.

3. **BART chunk input is already compressed.** In hybrid mode, the LLM receives
   pre-summarised chunk notes, not raw text. The synthesis task is bounded — larger
   context windows offer no advantage when the input is already short.

4. **Temperature asymmetry reveals input quality difference.** Hybrid optimal temp=0.5
   (BART chunk noise requires more diversity); direct optimal temp=0.3 (clean transcript
   requires focused synthesis). This confirms the model is performing different cognitive
   tasks, and 3B is well-suited to both.

## Temperature Findings

| Mode | Optimal temp | Reason |
| :--- | :---: | :--- |
| Hybrid (BART MAP input) | 0.5 | BART chunk compression introduces noise; LLM needs creative diversity to synthesise coherently |
| Direct (full transcript) | 0.3 | Clean, uncompressed input; focused deterministic generation wins |

## Deployment Configuration

```python
# config_constants.py
OLLAMA_DEFAULT_SUMMARY_MODEL = "llama3.2:3b"
```

```yaml
# direct_llama32_3b_autoresearch_v1.yaml
params:
  temperature: 0.3
  max_length: 1000
  min_length: 200
```

## Consequences

- **Positive**: 3B model is fast (~7.5s/ep on Apple Silicon MPS) and fits in ~2GB RAM —
  practical for local deployment without dedicated GPU.
- **Positive**: `llama3.2:3b` pull is ~2GB vs 4-8GB for 7-12B alternatives — lower barrier
  to entry for new deployments.
- **Positive**: Direct mode removes the BART dependency for Tier 3, simplifying the stack.
- **Neutral**: Quality ceiling for local inference is ~26% ROUGE-L vs ~33% for cloud
  (Tier 4). The ~7pp gap is the cost of local privacy.
- **Negative**: If episode lengths grow beyond ~15K tokens, the 3B context window may
  become a constraint and the hybrid architecture would need to be revisited.

## References

- [RFC-057: AutoResearch Optimization Loop](../rfc/RFC-057-autoresearch-optimization-loop.md)
- [ADR-069: Hybrid ML Pipeline as Production Direction](ADR-069-hybrid-ml-pipeline-as-production-direction.md)
- [ADR-070: BART-base as Hybrid MAP Stage](ADR-070-bart-base-as-hybrid-map-stage.md)
- [ADR-071: Four-Tier Summarization Strategy](ADR-071-four-tier-summarization-strategy.md)
- Sweep TSV: `autoresearch/ml_param_tuning/results/direct_llama32_3b_sweep_20260404_183918.tsv`
- Canonical config: `data/eval/configs/summarization/direct_llama32_3b_autoresearch_v1.yaml`
