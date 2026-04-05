# Reference: silver_sonnet46_benchmark_bullets_v1

**Promoted from:** `experiment_sonnet46_benchmark_bullets_v1`
**Promoted at:** 2026-04-05T13:55:44.259627Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** N/A

## Purpose

Sonnet-4.6 benchmark bullets silver

## Usage

This reference is used as "truth" for evaluation metrics (e.g., ROUGE).
It is:

- Not required for experiments (experiments can run without references)
- Cannot block CI (references are informational)
- Rarely updated (only when truth changes)
- Used for absolute quality assessment

## Artifacts

- `predictions.jsonl` - Reference outputs for all episodes
- `metrics.json` - Aggregate metrics
- `fingerprint.json` - System fingerprint (reproducibility)
- `baseline.json` - Metadata (kept for compatibility)
- `config.yaml` - Experiment config used (if provided)
