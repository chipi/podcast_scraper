# Baseline: {baseline_id}

## Purpose

This baseline represents a frozen reference run used for regression detection and comparison.

It is used for:

- Regression detection (experiments vs baseline)
- CI gating (block PRs that regress)
- Cost/latency tradeoff evaluation

## Dataset

- **Dataset ID:** `{dataset_id}`
- **Episode Count:** {episode_count}
- **Dataset Validated:** `curated_5feeds_smoke_v1` (and later benchmark id)

## Scope

- **Scope:** dev-smoke / prod-benchmark
  - `dev-smoke`: Development validation on smoke test dataset
  - `prod-benchmark`: Production validation on full benchmark dataset

## Invariants

- Outputs are immutable
- Must not be overwritten
- Comparisons must use the same `dataset_id`
- This artifact is immutable once published

## Replacement Policy

Replace only when:

- A new baseline has been explicitly approved
- A new `baseline_id` is created (e.g., `{next_version}`)

**Never update in place.**

## Promotion Process

**prod_candidate → prod_authority after benchmark run passes gates + qualitative check**

Production candidates (`baseline_ml_prod_candidate_*`) are validated on smoke test datasets. To promote to production authority (`baseline_ml_prod_authority_*`):

1. Run benchmark on full production dataset
2. Verify all quality gates pass (boilerplate leak, speaker leak, truncation, etc.)
3. Perform qualitative review of sample outputs
4. Use promotion process to copy run to baseline folder:

```bash
make run-promote \
  RUN_ID=<benchmark_run_id> \
  --as baseline \
  PROMOTED_ID=baseline_ml_prod_authority_v1 \
  REASON="Promoted after successful benchmark run with all gates passing"
```

This copies the run from `data/eval/runs/` to `data/eval/baselines/` and marks it as immutable.

## Contents

- `predictions.jsonl` - Model outputs for all episodes
- `metrics.json` - Aggregated performance and quality metrics
- `fingerprint.json` - Complete system fingerprint (reproducibility)
- `baseline.json` - Baseline metadata and statistics
- `config.yaml` - Experiment configuration used (if provided)

## Model Configuration

{model_description}

- **Preprocessing Profile:** {preprocessing_profile}
- **Generation Params:** Baseline-safe defaults (temperature={temperature}, seed={seed})

See `fingerprint.json` for complete configuration details.
