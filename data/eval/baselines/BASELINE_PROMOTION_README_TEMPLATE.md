# Baseline: {promoted_id}

**Promoted from:** `{run_id}`
**Promoted at:** {promoted_at}Z
**Role:** Baseline (used for experiment comparison)

## Purpose

{reason}

## Usage

This baseline is used as a comparison point for experiments. It is:

- Required for experiments (experiments must specify this baseline)
- Can block CI (regressions against this baseline can fail CI)
- Immutable (cannot be overwritten)

## Artifacts

- `predictions.jsonl` - Model outputs for all episodes
- `metrics.json` - Aggregate metrics
- `fingerprint.json` - System fingerprint (reproducibility)
- `baseline.json` - Baseline metadata
- `config.yaml` - Experiment config used (if provided)
