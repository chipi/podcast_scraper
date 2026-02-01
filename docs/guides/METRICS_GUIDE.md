# Metrics Guide

This guide explains how metrics are calculated for experiment runs and what each metric means.

## Overview

When you run an experiment using `make experiment-run`, the system computes two types of metrics:

1. **Intrinsic Metrics** - Always computed from predictions alone (no reference needed)
2. **vs_reference Metrics** - Computed when reference summaries are provided (optional)

## Metrics Calculation Flow

```text
experiment-run → run_experiment.py → score_run() → metrics.json
```

1. **Run Experiment**: `scripts/eval/run_experiment.py` processes episodes and generates `predictions.jsonl`
2. **Compute Metrics**: `score_run()` in `src/podcast_scraper/evaluation/scorer.py` reads predictions and computes metrics
3. **Save Results**: Metrics are saved to `data/eval/runs/<run_id>/metrics.json` and `metrics_report.md`

## Intrinsic Metrics

Intrinsic metrics are computed from predictions alone and don't require reference summaries. They include:

### 1. Quality Gates

Detect common issues in generated summaries:

- **`boilerplate_leak_rate`**: Fraction of episodes with promotional/sponsor content leaks
  - Patterns detected: "subscribe to our newsletter", "follow us on", "rate and review", etc.
- **`speaker_leak_rate`**: Fraction of episodes with speaker annotations leaking through
  - Patterns detected: "Host:", "Speaker 1:", "[laughter]", etc.
- **`truncation_rate`**: Fraction of episodes that appear truncated
  - Detected by truncation markers ("...", "[TRUNCATED]") or suspiciously short outputs
- **`failed_episodes`**: List of episode IDs that failed quality gates

### 2. Length Metrics

Token-based length statistics:

- **`avg_tokens`**: Average number of tokens per summary (estimated as chars/4)
- **`min_tokens`**: Minimum tokens across all summaries
- **`max_tokens`**: Maximum tokens across all summaries

### 3. Performance Metrics

Latency measurements:

- **`avg_latency_ms`**: Average processing time per episode in milliseconds
  - Extracted from `metadata.processing_time_seconds` in predictions

### 4. Cost Metrics (OpenAI Only)

**Note**: Cost metrics are only included for OpenAI runs. ML model runs skip this section entirely.

- **`avg_cost_usd`**: Average cost per episode in USD
- **`total_cost_usd`**: Total cost for all episodes in USD

Cost is computed from:

- `metadata.cost_usd` (if directly provided by provider)
- `metadata.usage` (token counts) with model-specific pricing:
  - GPT-4o-mini: $0.15/1M input, $0.60/1M output
  - GPT-4o: $2.50/1M input, $10.00/1M output

## vs_reference Metrics

**vs_reference** metrics compare your predictions against reference summaries (golden or silver standards). These are **optional** and only computed when references are provided.

### When is vs_reference null?

`vs_reference` is `null` when:

- No references were provided via `--reference` CLI argument
- The experiment was run without reference evaluation

### How to provide references

```bash
# Single reference
make experiment-run CONFIG=... REFERENCE_IDS=golden_v1

# Multiple references
make experiment-run CONFIG=... REFERENCE_IDS="golden_v1 silver_v2"
```

Or via CLI:

```bash
python scripts/eval/run_experiment.py config.yaml --reference golden_v1 --reference silver_v2
```

### Reference Structure

References can be:

- **Baselines**: `data/eval/baselines/<baseline_id>/`
- **References**: `data/eval/references/<dataset_id>/<reference_id>/`
- **Legacy baselines**: `benchmarks/baselines/<baseline_id>/`

Each reference must have a `predictions.jsonl` file with the same episode IDs as your run.

### vs_reference Metrics Computed

When references are provided, the following metrics are computed:

1. **`reference_quality`**: Metadata about the reference (episode count, etc.)

2. **ROUGE Scores** (requires `rouge-score` package):
   - `rouge1_f1`: ROUGE-1 F1 score (unigram overlap)
   - `rouge2_f1`: ROUGE-2 F1 score (bigram overlap)
   - `rougeL_f1`: ROUGE-L F1 score (longest common subsequence)

3. **BLEU Score** (requires `nltk` package):
   - `bleu`: BLEU score (n-gram precision with brevity penalty)

4. **WER (Word Error Rate)** (requires `jiwer` package):
   - `wer`: Word-level edit distance normalized by reference length

5. **Embedding Similarity** (requires `sentence-transformers` package):
   - `embedding_similarity`: Cosine similarity between embeddings of predictions and references

6. **`numbers_retained`**: TODO - Not yet implemented

### Example vs_reference Structure

```json
{
  "vs_reference": {
    "golden_v1": {
      "reference_quality": {...},
      "rouge1_f1": 0.45,
      "rouge2_f1": 0.32,
      "rougeL_f1": 0.42,
      "bleu": 0.38,
      "wer": 0.15,
      "embedding_similarity": 0.87
    }
  }
}
```

## Metrics Output Files

### `metrics.json`

Complete metrics dictionary with all computed values:

```json
{
  "dataset_id": "curated_5feeds_smoke_v1",
  "run_id": "baseline_bart_small_led_long_fast",
  "episode_count": 5,
  "intrinsic": {
    "gates": {...},
    "length": {...},
    "performance": {...},
    "cost": {...}  // Only for OpenAI runs
  },
  "vs_reference": null  // or {...} if references provided
}
```

### `metrics_report.md`

Human-readable markdown report with formatted metrics, suitable for viewing in GitHub or documentation.

## Usage Examples

### Run with ML models (no cost metrics)

```bash
make experiment-run CONFIG=data/eval/configs/baseline_bart_small_led_long_fast.yaml
```

Result: Intrinsic metrics only, no cost section, `vs_reference: null`

### Run with OpenAI (includes cost)

```bash
make experiment-run CONFIG=data/eval/configs/openai_gpt4o_mini.yaml
```

Result: Intrinsic metrics with cost section, `vs_reference: null`

### Run with references (includes vs_reference)

```bash
make experiment-run CONFIG=... REFERENCE_IDS=golden_v1
```

Result: Intrinsic metrics + `vs_reference` with ROUGE/BLEU/WER scores

## Related Documentation

- `src/podcast_scraper/evaluation/scorer.py` - Metrics computation implementation
- `src/podcast_scraper/evaluation/reporter.py` - Report generation
- `scripts/eval/run_experiment.py` - Experiment runner that calls scorer
