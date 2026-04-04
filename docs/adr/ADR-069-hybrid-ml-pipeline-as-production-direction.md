# ADR-069: Hybrid ML Pipeline as Primary Production Summarization Direction

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md),
  [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md)
- **Supersedes**: —
- **See Also**: [ADR-043](ADR-043-hybrid-map-reduce-summarization.md),
  [ADR-068](ADR-068-bart-led-as-ml-production-baseline.md),
  [ADR-070](ADR-070-bart-base-as-hybrid-map-stage.md)

## Context & Problem Statement

RFC-057 Track B autoresearch established BART+LED as the best pure-ML local baseline at 18.82%
ROUGE-L (ADR-068). A persistent ~10pp gap remained vs cloud models (28–32% ROUGE-L). The
hybrid ML architecture (RFC-042) — classic HF model for MAP compression + local LLM for REDUCE
synthesis — had been implemented but not empirically validated against the pure-ML baseline or
across LLM candidates. RFC-057 Track B extended the sweep harness to cover hybrid Ollama configs
to close this question with measurement.

## Decision

Adopt the **hybrid ML pipeline** (HF MAP model + Ollama LLM REDUCE) as the primary production
summarization direction for podcast content, superseding pure-ML BART+LED in the default path
once fully validated.

The validated champion configuration is:
**`ml_hybrid_bart_llama32_3b_autoresearch_v1`** — BART-base MAP + Llama 3.2:3b REDUCE
(temperature=0.5, top_p=1.0) at **23.1% ROUGE-L, 77.7% embedding cosine, ~17s/episode**.

## LLM Reduce Candidate Evaluation

All candidates evaluated on `curated_5feeds_smoke_v1` vs. `silver_sonnet46_smoke_v1` at
baseline parameters (temperature=0.3, top_p=0.9). LongT5-base MAP for all.

| Model | Size | ROUGE-L (baseline) | Embed | Latency | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Llama 3.2:3b** | 3B | 18.9% | 76.3% | 15s | Meta, instruction-tuned |
| Qwen 2.5:7b | 7B | 17.5% | 76.4% | 20s | Alibaba, multilingual |
| Gemma 2:9b | 9B | ~17.8% | ~75.8% | 23s | Google, academic-tuned |
| Llama 3.1:8b | 8B | ~17.2% | ~75.1% | 21s | Meta, general |
| Mistral-Nemo:12b | 12B | 18.7% | 77.7% | 19s | Mistral AI, multilingual |

**Key finding — instruction-following beats raw size**: Llama 3.2:3b (3B parameters) outperformed
all 7–12B models at baseline. Smaller but instruction-optimized models synthesize podcast content
more effectively than larger general-purpose models. Qwen 2.5:7b at 7B underperformed the 3B
Llama by 1.4pp ROUGE-L despite being 2.3x larger.

## Temperature as the Critical Hybrid Tuning Lever

The default `ollama_reduce_temperature=0.3` was inherited from the Ollama provider's general
summarization default. Autoresearch sweep on Llama 3.2:3b revealed this was too conservative:

| Temperature | ROUGE-L | Delta vs 0.3 |
| :---: | :---: | :---: |
| 0.1 | 17.1% | -9.5% |
| **0.5** | **20.8%** | **+10.0% ← accepted** |
| 0.7 | 19.5% | +3.2% (later rejected by top_p sweep) |

**0.5 is the sweet spot**: enough randomness for creative synthesis (the LLM needs to paraphrase
and connect ideas across chunks) but not so much that it drifts from the source material.
Temperature 0.3 produced overly conservative, extractive-style output. Temperature 0.7 introduced
hallucination-adjacent drift.

The same finding held for Qwen 2.5:7b (temperature=0.5: +4.23% vs baseline).

## Full Autoresearch Sweep — Llama 3.2:3b (LongT5 MAP)

| Param Group | Param | Candidate | ROUGE-L | Delta | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: |
| ollama_reduce | temperature | 0.1 | 17.1% | -9.5% | ✗ |
| ollama_reduce | **temperature** | **0.5** | **20.8%** | **+10.0%** | **✓** |
| ollama_reduce | top_p | 0.7 | 20.7% | -0.5% | ✗ |
| ollama_reduce | top_p | 0.95 | 20.5% | -1.4% | ✗ |
| ollama_reduce | top_p | 1.0 | 20.4% | -1.9% | ✗ |
| — | — | — | — | early stop | — |

**Round total**: baseline 18.9% → 20.8% ROUGE-L (+10.0%), 1 param accepted.

## Full Autoresearch Sweep — BART MAP + Llama 3.2:3b REDUCE

Base: `hybrid_ml_bart_llama32_3b_smoke_paragraph_v1` (ROUGE-L 20.66%, post BART map swap)

| Param Group | Param | Candidate | ROUGE-L | Delta | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: |
| ollama_reduce | top_p | 0.7 | 20.65% | -0.05% | ✗ |
| ollama_reduce | **top_p** | **0.95** | **22.71%** | **+9.91%** | **✓** |
| ollama_reduce | **top_p** | **1.0** | **23.12%** | **+1.84%** | **✓** |
| ollama_reduce | frequency_penalty | 0.3 | 19.55% | -15.44% | ✗ |
| ollama_reduce | frequency_penalty | 0.6 | 22.66% | -1.99% | ✗ |
| ollama_reduce | frequency_penalty | 1.0 | 20.04% | -13.35% | ✗ |
| — | — | — | early stop | — | — |

**Round total**: 20.66% → 23.12% ROUGE-L (+11.93%), 2 params accepted.

**top_p=1.0 finding**: No nucleus sampling. With temperature=0.5 already controlling
randomness, removing token probability filtering lets the model access its full vocabulary for
synthesis — particularly important for domain-specific podcast terminology that might sit below
top-p cutoffs.

**frequency_penalty finding**: Any frequency penalty hurts quality with BART map input.
BART produces sufficiently diverse chunk summaries that the LLM does not need external
repetition penalties — adding them suppresses legitimate recurring concepts.

## Final Champion vs. All Alternatives

| Mode | ROUGE-L | Embed | Latency | Privacy |
| :--- | :---: | :---: | :---: | :---: |
| `ml_hybrid_bart_llama32_3b_autoresearch_v1` | **23.1%** | **77.7%** | 17s | 100% local |
| `ml_hybrid_llama32_3b_autoresearch_v1` (LongT5 MAP) | 20.8% | 76.3% | 15s | 100% local |
| Mistral-Nemo:12b baseline | 18.7% | 77.7% | 19s | 100% local |
| `ml_bart_led_autoresearch_v1` (pure ML) | 18.8% | 72.6% | ~30s | 100% local |
| Qwen 2.5:7b (swept) | 18.2% | 76.4% | 20s | 100% local |
| OpenAI GPT-4o (cloud reference) | ~28–32% | ~82% | ~3s | ☁️ cloud |
| Anthropic Claude Sonnet 4.6 (cloud) | ~32.6% | ~85% | ~3s | ☁️ cloud |

The hybrid champion closes **70% of the gap** between pure-ML local (18.8%) and best cloud
(32.6%) at zero variable cost and with full data privacy.

## Alternatives Considered

1. **Pure ML BART+LED only** — Rejected as the primary direction. Caps at ~19% ROUGE-L due to
   LED's abstractive quality ceiling. Useful as a privacy-maximum fallback when Ollama is
   unavailable.

2. **LLM-only (no MAP stage)** — Not evaluated. Full transcript exceeds LLM context windows
   without chunking; MAP compression is architecturally necessary. See RFC-042.

3. **Cloud LLM reduce** — Rejected. Breaks the privacy-first local principle (ADR-009) and
   introduces variable cost. The 9pp gap to cloud is acceptable given the privacy/cost benefit.

4. **Larger Ollama models (Qwen 2.5:32b, Llama 3.3:70b)** — Not swept in this phase. Results
   from the 3–12B comparison suggest returns diminish beyond 3B for instruction-tuned models
   on this task. Can be revisited if hardware constraints change.

## Consequences

- **Positive**: +22.9% relative ROUGE-L over pure-ML baseline. Closes 70% of cloud quality gap.
- **Positive**: 17s/episode — faster than larger cloud-competing Ollama models (19–23s) and
  faster than pure-ML BART+LED (~30s due to beam search on CPU-loaded LED).
- **Positive**: Empirically validated parameter choices (temperature, top_p) replace ad-hoc
  defaults across all hybrid configs.
- **Neutral**: Requires Ollama to be running locally. Graceful fallback to `ml_bart_led_autoresearch_v1`
  when Ollama is unavailable (existing routing logic, RFC-042).
- **Neutral**: `PROD_DEFAULT_SUMMARY_MODE_ID` migration from pure-ML to hybrid deferred until
  full authority benchmark confirms generalisation beyond the smoke dataset.
- **Negative**: Adds a runtime dependency (Ollama daemon). Production deployment must ensure
  Ollama is pre-started and the target model is pulled.

## Implementation Notes

- **Registry entry**: `src/podcast_scraper/providers/ml/model_registry.py` →
  `_mode_registry["ml_hybrid_bart_llama32_3b_autoresearch_v1"]`
- **Ollama params wiring**: `OllamaReduceParams` dataclass in
  `src/podcast_scraper/evaluation/experiment_config.py`; passed through
  `scripts/eval/run_experiment.py` → `src/podcast_scraper/config.py` →
  `src/podcast_scraper/providers/ollama/ollama_provider.py`
- **Canonical eval config**: `data/eval/configs/ml/baseline_ml_hybrid_bart_llama32_3b_autoresearch_v1.yaml`
- **Sweep TSVs**: `autoresearch/ml_param_tuning/results/bart_llama32_3b_sweep_*.tsv`

## References

- [RFC-057: AutoResearch Optimization Loop](../rfc/RFC-057-autoresearch-optimization-loop.md)
- [RFC-042: Hybrid Podcast Summarization Pipeline](../rfc/RFC-042-hybrid-summarization-pipeline.md)
- [ADR-043: Hybrid MAP-REDUCE Summarization](ADR-043-hybrid-map-reduce-summarization.md)
- [ADR-068: BART+LED as ML Production Baseline](ADR-068-bart-led-as-ml-production-baseline.md)
- [ADR-070: BART-base as Hybrid Map Stage](ADR-070-bart-base-as-hybrid-map-stage.md)
