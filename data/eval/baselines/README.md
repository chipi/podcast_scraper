# Baselines

This directory contains frozen reference runs used for regression detection and comparison.

## Purpose

Baselines represent known-good system behavior at a specific point in time. They are used for:

- Regression detection (experiments vs baseline)
- CI gating (block PRs that regress)
- Cost/latency tradeoff evaluation
- Historical tracking of system performance

## Task Types

Baselines can be created for different task types:

- **summarization** - Text summarization baselines (e.g., `baseline_ml_prod_authority_v1`)
- **ner_entities** - Named Entity Recognition baselines (e.g., `baseline_ner_prod_authority_v1`)

## Structure

Each baseline is stored in its own directory:

```text
baselines/
  {baseline_id}/
    run.log              # Execution log (what actually happened)
    predictions.jsonl
    metrics.json
    fingerprint.json     # System fingerprint (what should have happened)
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

- `baseline_ml_prod_authority_v1` - Production authoritative baseline (summarization)
- `baseline_ner_prod_authority_v1` - Production authoritative baseline (NER)
- `baseline_ner_dev_authority_v1` - Development authoritative baseline (NER)
- `baseline_ml_dev_authority_smoke_v1` - Development smoke test baseline (summarization)
- `bart_led_small_smoke_v1` - Smoke test baseline with small models (summarization)
- `baseline_ci_smoke_v1` - CI smoke test baseline

## Contents

Each baseline contains:

- `run.log` - Execution log capturing what actually happened (models loaded, params used, warnings)
- `predictions.jsonl` - Model outputs for all episodes
- `metrics.json` - Aggregated performance and quality metrics
- `fingerprint.json` - Complete system fingerprint (reproducibility - what should have happened)
- `baseline.json` - Baseline metadata and statistics
- `config.yaml` - Experiment configuration used
- `README.md` - Baseline-specific purpose and invariants

**Note**: `run.log` is copied from the original run during promotion. It provides diagnostic evidence of execution, complementing `fingerprint.json` which defines the intended configuration.

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
