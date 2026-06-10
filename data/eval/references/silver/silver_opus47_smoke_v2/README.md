# Reference: silver_opus47_smoke_v2

**Promoted from:** `silver_candidate_anthropic_opus47_smoke_v2_paragraph`
**Promoted at:** 2026-06-09T13:21:14.421323Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** N/A
**Reference Quality:** silver

## Purpose

Opus 4.7 silver on v2 dataset — companion to silver_opus47_smoke_v1 (#939)

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
