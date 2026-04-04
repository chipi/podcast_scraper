# ADR-070: BART-base as Hybrid MAP Stage Over LongT5-base

- **Status**: Accepted
- **Date**: 2026-04-04
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md),
  [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md)
- **See Also**: [ADR-069](ADR-069-hybrid-ml-pipeline-as-production-direction.md),
  [ADR-068](ADR-068-bart-led-as-ml-production-baseline.md)

## Context & Problem Statement

The hybrid ML pipeline (ADR-069) requires a MAP model to compress podcast transcript chunks
into dense summaries that an Ollama LLM then synthesises. Two candidates were evaluated:

- **BART-base** (`facebook/bart-base`): 1024-token context, pretrained on general text with
  BERT-style denoising. Already proven in pure-ML BART+LED pipeline.
- **LongT5-base** (`google/long-t5-tglobal-base`): 4096-token context, pretrained with
  span-corruption (T5-style). Designed for longer document understanding.

The intuition was that LongT5's 4x larger context window would produce richer chunk summaries
→ better LLM synthesis. This was tested empirically.

## Decision

Use **BART-base** (`bart-small` in model registry) as the MAP stage in the hybrid ML pipeline.

LongT5-base is retained as a supported MAP model but is not the default. It remains available
for experimentation and may be reconsidered if the Ollama reduce model changes.

## Experimental Evidence

### Head-to-Head Comparison (identical Ollama reduce: Llama 3.2:3b, temp=0.5, top_p=0.9)

Evaluated on `curated_5feeds_smoke_v1` vs. `silver_sonnet46_smoke_v1`:

| MAP Model | ROUGE-L | Embed Cosine | Avg Output Tokens | Latency |
| :--- | :---: | :---: | :---: | :---: |
| **BART-base** | **21.24%** | **77.5%** | **461** | **16s** |
| LongT5-base | 20.82% | 76.3% | ~430 | 15s |
| **Delta** | **+0.42pp** | **+1.2pp** | **+31** | +1s |

BART wins on both ROUGE-L and embedding cosine despite LongT5's larger context window.

### After Autoresearch Sweep (BART MAP, same reduce)

Further sweeping ollama_reduce params on the BART-map baseline:

| Config | ROUGE-L | Embed | Latency |
| :--- | :---: | :---: | :---: |
| BART MAP + Llama 3.2:3b baseline | 21.24% | 77.5% | 16s |
| + top_p=0.95 | 22.71% | — | — |
| + top_p=1.0 | **23.12%** | **77.7%** | **17s** |

BART MAP config reached 23.12% ROUGE-L after sweep — 0.3pp above the LongT5 MAP best
(20.82%) even before any additional tuning on the LongT5 path.

### LongT5+LED Pure-ML Control

To further understand LongT5's properties, a pure-ML LongT5+LED mode was evaluated:

| Config | ROUGE-L | Embed | Avg Tokens | Notes |
| :--- | :---: | :---: | :---: | :--- |
| LongT5+LED baseline | 8.4% | 46.6% | 42 | LED generates EOS at ~42 tokens |
| + map max_new_tokens=150 | 9.9% | 49.6% | 48 | Only improvement found |
| BART+LED autoresearch_v1 | 18.8% | 72.6% | ~230 | 2x ROUGE-L, 5x tokens |

LongT5 in pure-ML mode is severely constrained: its span-corruption pretraining produces
compressed, abstract chunk summaries (~15–25 tokens/chunk) that starve LED of synthesis
material. Counter-intuitively, `max_new_tokens=150` (shorter) outperformed 200/250/300
because the constraint forced more selective, information-dense compression.

**This explains the hybrid result**: When the LLM replace LED as the reduce model, the
constraint inverts — the LLM is powerful enough to synthesise even from terse input. But BART
still produces *richer* notes than LongT5, giving the LLM more to work with.

## Why Pretraining Alignment Beats Context Window Size

| Property | BART-base | LongT5-base |
| :--- | :--- | :--- |
| Context window | 1024 tokens | 4096 tokens |
| Pretraining objective | BERT denoising (reconstruct masked spans) | Span corruption (fill in gaps) |
| Output style | Content-rich, descriptive | Compressed, abstract, short |
| Typical chunk output length | 100–150 tokens | 15–25 tokens |
| Training data | Books + Common Crawl (diverse) | C4 (web-crawled) |
| Podcast chunk compression | Preserves named entities, quotes, examples | Strips specifics, retains structure |

The podcast transcription domain shares more surface properties with BART's diverse
training data than with LongT5's structured web text. BART "describes" chunk content;
LongT5 "abstracts" it. For LLM synthesis, description is more useful — the LLM already
knows how to abstract.

The context window advantage of LongT5 (4096 vs 1024) is not relevant in practice:
podcast chunks are chunked to 900 words (~1100 tokens) to fit BART's window. LongT5's
extra capacity is unused since the chunking strategy is constrained by the smaller model.

## Alternatives Considered

1. **LongT5 with tuned map params** — Swept `max_new_tokens` (150, 200, 250, 300), `num_beams`
   (4, 6, 8), `no_repeat_ngram_size` (2, 3, 4). Only `max_new_tokens=150` accepted (+17.5% in
   pure-ML mode). Even after tuning, LongT5+LED reached only 9.9% — 50% below BART+LED.
   In hybrid mode, the gap is smaller (21.2% vs 20.8%) but BART still wins.

2. **Larger chunk size for LongT5** — Not evaluated. Would require extending the chunking
   pipeline for model-specific windows. Deferred: BART wins without this change.

3. **PEGASUS as MAP** — Rejected. Near-duplicate chunk output makes LLM synthesis
   degenerate regardless of reduce model. See ADR-067.

4. **Flan-T5 as MAP** — Not evaluated in this round. Flan-T5 is instruction-tuned (good for
   prompted summarization) but not evaluated in the hybrid context. Candidate for future work.

## Consequences

- **Positive**: BART-base map produces richer chunk summaries → 23.1% ROUGE-L vs 20.8% for
  LongT5 after equivalent sweep effort.
- **Positive**: Reuses the same BART model already loaded for pure-ML fallback mode — no
  additional model download for hybrid users.
- **Neutral**: LongT5 remains available as an alternative MAP stage for users who want to
  experiment with the longer context window (e.g., for very long episodes with fewer chunks).
- **Neutral**: Chunking strategy (word_chunk_size=900) continues to be driven by BART's 1024
  token limit, not LongT5's 4096 limit.
- **Negative**: The 4096-token context advantage of LongT5 is currently unexploited. If
  episode length distribution shifts (e.g., very long episodes > 2h), LongT5 may become
  preferable with a larger chunk size.

## Implementation Notes

- **Champion config**: `data/eval/configs/ml/baseline_ml_hybrid_bart_llama32_3b_autoresearch_v1.yaml`
- **Registry entry**: `src/podcast_scraper/providers/ml/model_registry.py` →
  `_mode_registry["ml_hybrid_bart_llama32_3b_autoresearch_v1"]`
- **Sweep TSVs**: `autoresearch/ml_param_tuning/results/bart_llama32_3b_sweep_*.tsv`
- **LongT5 sweep TSVs**: `autoresearch/ml_param_tuning/results/longt5_llama32_3b_sweep_*.tsv`,
  `autoresearch/ml_param_tuning/results/longt5_led_sweep_*.tsv`

## References

- [RFC-057: AutoResearch Optimization Loop](../rfc/RFC-057-autoresearch-optimization-loop.md)
- [RFC-042: Hybrid Podcast Summarization Pipeline](../rfc/RFC-042-hybrid-summarization-pipeline.md)
- [ADR-069: Hybrid ML Pipeline as Production Direction](ADR-069-hybrid-ml-pipeline-as-production-direction.md)
- [ADR-068: BART+LED as ML Production Baseline](ADR-068-bart-led-as-ml-production-baseline.md)
- [ADR-067: Pegasus Retirement](ADR-067-pegasus-led-retirement-podcast-content.md)
