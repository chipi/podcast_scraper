# ADR-067: Pegasus/LED Retirement for Podcast Content

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md)
- **Supersedes**: —
- **See Also**: [ADR-010](ADR-010-hierarchical-summarization-pattern.md), [ADR-043](ADR-043-hybrid-map-reduce-summarization.md)

## Context & Problem Statement

Pegasus/CNN-DailyMail + LED-base-16384 (`ml_prod_authority_v1`) was the production summarization
mode for podcast content. During RFC-057 Track B autoresearch, empirical evaluation on
`curated_5feeds_smoke_v1` revealed a fundamental architectural mismatch that cannot be resolved
through parameter tuning.

## Decision

Retire `ml_prod_authority_v1` (Pegasus MAP + LED REDUCE) as the production summarization mode
for podcast content. Preserve the entry in `model_registry.py` as deprecated, reserved for a
future news content type where the architectural fit is correct.

## Root Cause Analysis: Three-Layer Failure

### Layer 1 — Pretraining Domain Mismatch

Pegasus-CNN/DailyMail was pretrained with **Gap Sentence Generation (GSG)** on the CNN/DailyMail
news corpus. GSG selects the most "important" sentences (by ROUGE to the rest of the document)
as prediction targets. News articles have a canonical inverted-pyramid structure — a single
dominant lead paragraph contains the key facts, with subsequent paragraphs adding detail.

Podcast transcripts have the opposite structure: conversational, topic-distributed, no single
dominant segment. GSG applied to podcast chunks consistently selects overlapping sentences
because all chunks are roughly equal in salience.

**Measured evidence**: Across `curated_5feeds_smoke_v1`, Pegasus MAP produced near-duplicate
chunk summaries — p02_e01 had 4/5 chunks with cosine similarity > 0.93 between each other.

### Layer 2 — Redundant Input Exhausts LED's Diversity Mechanism

LED-base-16384 uses `no_repeat_ngram_size=3` to prevent repetitive output. When its input
(combined Pegasus chunk summaries) is near-duplicate, the constraint fires within the first
50–70 tokens of generation. Every valid continuation is blocked — LED produces an EOS token
at ~55–65 tokens regardless of `max_new_tokens`.

**Measured evidence**: All reduce `max_new_tokens` candidates (450, 550, 750) produced
identical output length of ~58 avg tokens — proof that `max_new_tokens` was not the constraint.

### Layer 3 — Parameter Tuning Cannot Fix Architecture

Sweeping `no_repeat_ngram_size` (2, 4, 5, 6, 8) produced no improvement — relaxing the
constraint did not help because the input redundancy is the root cause, not the constraint
itself. Sweeping `length_penalty` (1.2, 1.5, 2.0) and `min_new_tokens` also produced no gain.

**Sweep result**: 0 accepted params across all reduce and map param groups. Baseline ROUGE-L
remained at ~5–7% throughout — well below BART+LED (18.8%) and hybrid modes (20–23%).

## Measured Performance

Evaluated against `silver_sonnet46_smoke_v1` reference on `curated_5feeds_smoke_v1`:

| Metric | ml_prod_authority_v1 (Pegasus+LED) | ml_bart_led_autoresearch_v1 (BART+LED) | Delta |
| :--- | :---: | :---: | :---: |
| ROUGE-L F1 | ~6.5% | 18.8% | +189% |
| Embedding Cosine | ~0.41 | 72.6% | +77% |
| Avg Output Tokens | ~58 | ~230 | +297% |
| Truncation Rate | 0% | 0% | — |

## Why Pegasus Is Appropriate for News

The GSG pretraining objective is well-matched to news content:

- News has a dominant lead paragraph → GSG selects it consistently → diverse chunk summaries
- Factual, formal register matches CNN/DailyMail pretraining distribution
- Inverted-pyramid structure means later chunks genuinely add less, so LED gets real signal

When a news content type is added, `ml_prod_authority_v1` should be re-evaluated without
modifications. Its architectural properties are correct for that domain.

## Alternatives Considered

1. **Tune Pegasus map params** — Rejected. `repetition_penalty` sweeps (1.3, 1.5, 1.7) did not
   produce diverse chunk summaries. The near-duplication is structural, not a decoding artifact.

2. **Replace LED reduce with Ollama LLM** — Rejected as a Pegasus fix. This becomes the hybrid
   ML architecture (see ADR-069), which uses BART not Pegasus as the map model.

3. **Fine-tune Pegasus on podcast data** — Not evaluated. Out of scope for RFC-057 Track B,
   which focuses on production-ready zero-shot models.

## Consequences

- **Positive**: Removes a consistently underperforming mode from the default path. Reduces
  confusion from having a deprecated-in-practice mode still listed as production.
- **Positive**: Documented retirement creates a clear record for the news content type decision.
- **Neutral**: `ml_prod_authority_v1` remains in `model_registry.py` with `deprecated_at` and
  `deprecation_reason` fields. Existing configs referencing it continue to work.
- **Negative**: Any existing deployments relying on `ml_prod_authority_v1` must migrate to
  `ml_bart_led_autoresearch_v1` (pure ML) or `ml_hybrid_bart_llama32_3b_autoresearch_v1`
  (hybrid, recommended).

## Implementation Notes

- **Registry**: `src/podcast_scraper/providers/ml/model_registry.py` — `ml_prod_authority_v1`
  marked with `deprecated_at="2026-04-03"` and `deprecation_reason` pointing to news content type.
- **Tombstone experiment**: `data/eval/baselines/baseline_ml_pegasus_retirement_smoke_v1/`
  with `RETIREMENT.md` documenting the full root cause.
- **Tombstone config**: `data/eval/configs/ml/baseline_ml_pegasus_retirement_smoke_v1.yaml`
- **Default**: `PROD_DEFAULT_SUMMARY_MODE_ID` updated to `ml_bart_led_autoresearch_v1`
  in `src/podcast_scraper/config_constants.py`.

## References

- [RFC-057: AutoResearch Optimization Loop](../rfc/RFC-057-autoresearch-optimization-loop.md)
- [ADR-010: Hierarchical Summarization Pattern](ADR-010-hierarchical-summarization-pattern.md)
- [ADR-043: Hybrid MAP-REDUCE Summarization](ADR-043-hybrid-map-reduce-summarization.md)
- Tombstone: `data/eval/baselines/baseline_ml_pegasus_retirement_smoke_v1/RETIREMENT.md`
