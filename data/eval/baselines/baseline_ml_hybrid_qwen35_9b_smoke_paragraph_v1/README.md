# Baseline: baseline_ml_hybrid_qwen35_9b_smoke_paragraph_v1

**Promoted from:** `hybrid_ml_tier2_qwen35_9b_smoke_paragraph_v1`
**Promoted at:** 2026-04-03T06:27:28.624773Z
**Role:** Baseline (used for experiment comparison)

## Purpose

ML/hybrid baseline vs silver_sonnet46_smoke_v1

## Scope

- **Scope:** dev-smoke / prod-benchmark
  - `dev-smoke`: Development validation on smoke test dataset
  - `prod-benchmark`: Production validation on full benchmark dataset

## Dataset Validation

- **Dataset Validated:** `curated_5feeds_smoke_v1` (and later benchmark id)

## Usage

This baseline is used as a comparison point for experiments. It is:

- Required for experiments (experiments must specify this baseline)
- Can block CI (regressions against this baseline can fail CI)
- Immutable (cannot be overwritten)

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

## Artifacts

- `predictions.jsonl` - Model outputs for all episodes
- `metrics.json` - Aggregate metrics
- `fingerprint.json` - System fingerprint (reproducibility)
- `baseline.json` - Baseline metadata
- `config.yaml` - Experiment config used (if provided)
