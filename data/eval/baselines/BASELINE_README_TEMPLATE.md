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
