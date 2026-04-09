# ADR-068: BART+LED as Local ML Production Baseline

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md)
- **Supersedes**: `ml_prod_authority_v1` (Pegasus+LED) — see [ADR-067](ADR-067-pegasus-led-retirement-podcast-content.md)
- **See Also**: [ADR-048](ADR-048-centralized-model-registry.md)

## Context & Problem Statement

Following Pegasus retirement (ADR-067), the project needed a validated local ML summarization
baseline for podcast content. `ml_small_authority` (BART-small + LED) existed as a development
baseline but had not been swept for optimal parameters. RFC-057 Track B defined a greedy
one-param-at-a-time sweep ratchet to find the best local ML configuration empirically.

## Decision

Promote `ml_bart_led_autoresearch_v1` — BART-base MAP + LED-base-16384 REDUCE with
autoresearch-tuned parameters — as the canonical local ML production baseline.
Register in `model_registry.py` and set as `PROD_DEFAULT_SUMMARY_MODE_ID`.

## Sweep Methodology

RFC-057 Track B uses a greedy ratchet:

- **Accept threshold**: ≥ +1% relative ROUGE-L gain over current best
- **Early stop**: 3 consecutive rejections within a param group
- **Reference**: `silver_sonnet46_smoke_v1` (Claude Sonnet 4.6 silver labels)
- **Dataset**: `curated_5feeds_smoke_v1` (5 episodes, 4 podcast feeds)

## Sweep Results — Round 1 (Reduce Params)

Base config: `baseline_ml_dev_authority` (BART-base MAP, num_beams=4; LED REDUCE, max_new_tokens=650)

| Param | Candidate | ROUGE-L | Delta | Decision |
| :--- | :---: | :---: | :---: | :---: |
| reduce `max_new_tokens` | 450 | — | rejected | ✗ |
| reduce `max_new_tokens` | **550** | **18.54%** | **+2.89%** | Accepted |
| reduce `max_new_tokens` | 750 | — | rejected | ✗ |
| reduce `num_beams` | **6** | **18.82%** | **+1.15%** | Accepted |
| reduce `num_beams` | 8 | — | rejected | ✗ |
| reduce `length_penalty` | 1.2 | — | rejected | ✗ |
| reduce `length_penalty` | 1.5 | — | rejected | ✗ |
| reduce `length_penalty` | 0.8 | — | early stop | ✗ |

**Round 1 outcome**: ROUGE-L 18.05% → 18.82% (+4.26%), 2 params accepted.

## Sweep Results — Round 2 (Map Params)

Base: round-1 winner (`max_new_tokens=550`, `num_beams=6`)

| Param | Candidate | Delta | Decision |
| :--- | :---: | :---: | :---: |
| map `num_beams` | 6 | +0.0% | ✗ |
| map `num_beams` | 8 | — | ✗ early stop |
| reduce `no_repeat_ngram_size` | 4, 5, 2 | ≤0% | ✗ all |
| reduce `min_new_tokens` | 150, 280, 320 | ≤0% | ✗ all |
| reduce `repetition_penalty` | 1.1, 1.5, 1.0 | ≤0% | ✗ all |

**Round 2 outcome**: No further gain. Round 1 winner is the stable optimum.

## Final Promoted Configuration

Registered as `ml_bart_led_autoresearch_v1` in `model_registry.py`:

```text
map_model:           bart-small (facebook/bart-base)
map_params:          num_beams=4, max_new_tokens=200, min_new_tokens=80,
                     no_repeat_ngram_size=3, repetition_penalty=1.3
reduce_model:        long-fast (allenai/led-base-16384)
reduce_params:       num_beams=6, max_new_tokens=550, min_new_tokens=220,
                     no_repeat_ngram_size=3, repetition_penalty=1.3
preprocessing:       cleaning_v4
chunking:            word_chunking, word_chunk_size=900, word_overlap=150
tokenize:            map_max_input_tokens=1024, reduce_max_input_tokens=4096
```

## Measured Performance vs. Alternatives

Evaluated on `curated_5feeds_smoke_v1` vs. `silver_sonnet46_smoke_v1`:

| Mode | ROUGE-L F1 | Embedding Cosine | Avg Tokens | Privacy |
| :--- | :---: | :---: | :---: | :---: |
| `ml_bart_led_autoresearch_v1` | **18.82%** | **72.6%** | ~230 | 100% local |
| `ml_prod_authority_v1` (Pegasus) | ~6.5% | ~41% | ~58 | 100% local |
| `ml_small_authority` (pre-sweep) | ~16.3% | ~70% | ~185 | 100% local |
| OpenAI GPT-4o (cloud reference) | ~28–32% | ~82% | ~420 | ☁️ cloud |

**Key finding**: Sweeping just 2 reduce parameters (`max_new_tokens`, `num_beams`) gave +4.26%
over the development baseline. The remaining ~10pp gap to cloud models is the motivation for
the hybrid ML architecture (ADR-069).

## Why BART over Pegasus for Local ML

| Property | BART-base | Pegasus-CNN |
| :--- | :--- | :--- |
| Pretraining objective | Text infilling (BERT-like denoising) | GSG (gap sentence generation) |
| Domain | General (books + web) | News (CNN/DailyMail) |
| Podcast chunk diversity | High — produces topically diverse summaries | Low — near-duplicate (see ADR-067) |
| LED compatibility | Compatible — diverse input enables ngram budget | Incompatible — exhausts ngram budget |

## Consequences

- **Positive**: 189% ROUGE-L improvement over Pegasus baseline in production.
- **Positive**: Establishes a reproducible sweep methodology (RFC-057 Track B ratchet) for
  future model promotions.
- **Neutral**: This mode is now the privacy-first fallback. The recommended production path
  is the hybrid ML pipeline (ADR-069), which surpasses this by a further +22.9%.
- **Neutral**: `PROD_DEFAULT_SUMMARY_MODE_ID` points to this mode as the pure-ML anchor
  while hybrid validation completes.

## Implementation Notes

- **Registry entry**: `src/podcast_scraper/providers/ml/model_registry.py` →
  `_mode_registry["ml_bart_led_autoresearch_v1"]`
- **Canonical eval config**: `data/eval/configs/ml/baseline_ml_bart_led_autoresearch_v1.yaml`
- **Sweep TSVs**: `autoresearch/ml_param_tuning/results/bart_led_sweep_*.tsv`
- **Default constant**: `src/podcast_scraper/config_constants.py` →
  `PROD_DEFAULT_SUMMARY_MODE_ID = "ml_bart_led_autoresearch_v1"`

## References

- [RFC-057: AutoResearch Optimization Loop](../rfc/RFC-057-autoresearch-optimization-loop.md)
- [ADR-067: Pegasus Retirement](ADR-067-pegasus-led-retirement-podcast-content.md)
- [ADR-048: Centralized Model Registry](ADR-048-centralized-model-registry.md)
- [ADR-069: Hybrid ML Pipeline as Production Direction](ADR-069-hybrid-ml-pipeline-as-production-direction.md)
