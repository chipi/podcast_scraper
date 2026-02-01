# Baselines

This directory contains frozen reference runs used for regression detection and comparison.

## Purpose

Baselines represent known-good system behavior at a specific point in time. They are used for:

- Regression detection (experiments vs baseline)
- CI gating (block PRs that regress)
- Cost/latency tradeoff evaluation
- Historical tracking of system performance

## Structure

Each baseline is stored in its own directory:

```text
baselines/
  {baseline_id}/
    predictions.jsonl
    metrics.json
    fingerprint.json
    baseline.json
    config.yaml
    README.md
```text

## Invariants

- Baselines are immutable once published
- Must not be overwritten
- Comparisons must use the same `dataset_id`
- This artifact is immutable once published

## Replacement Policy

Replace only when:

- A new baseline has been explicitly approved
- A new `baseline_id` is created (e.g., `baseline_prod_authority_v2`)

**Never update in place.**

## Baseline Naming

Baseline IDs should be descriptive and versioned:

- `baseline_prod_authority_v1` - Production authoritative baseline
- `bart_led_small_smoke_v1` - Smoke test baseline with small models
- `baseline_ci_smoke_v1` - CI smoke test baseline

## Contents

Each baseline contains:

- `predictions.jsonl` - Model outputs for all episodes
- `metrics.json` - Aggregated performance and quality metrics
- `fingerprint.json` - Complete system fingerprint (reproducibility)
- `baseline.json` - Baseline metadata and statistics
- `config.yaml` - Experiment configuration used
- `README.md` - Baseline-specific purpose and invariants

## Promotion

Baselines are created by promoting runs from `runs/`:

```bash
make run-promote \
  RUN_ID=run_2026-01-16_12-10-03 \
  --as baseline \
  PROMOTED_ID=baseline_prod_authority_v2 \
  REASON="New production baseline with improved preprocessing"
```

This moves the run to `baselines/` and marks it as immutable.
