# Baseline: baseline_ml_dev_authority_smoke_v1

**Promoted from:** `baseline_bart_v7_cleaning_v4`
**Promoted at:** 2026-02-01T13:58:58.529160Z
**Role:** Baseline (used for experiment comparison)

## Purpose

First baseline using cleaning_v4 preprocessing profile with speaker anonymization and header stripping. Demonstrates improved speaker leak prevention compared to v3.

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
