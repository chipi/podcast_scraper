# Reference: silver_sonnet46_benchmark_v2_paragraph

**Promoted from:** `silver_candidate_sonnet46_benchmark_v2_paragraph`
**Promoted at:** 2026-04-14T14:31:45.779360Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** N/A
**Reference Quality:** silver

## Purpose

Held-out benchmark silver (v2): Sonnet 4.6 paragraph on benchmark_v2 (5 ep e03)

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
