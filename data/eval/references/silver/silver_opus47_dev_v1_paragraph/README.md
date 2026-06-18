# Reference: silver_opus47_dev_v1_paragraph

**Promoted from:** `silver_candidate_anthropic_opus47_dev_v1_paragraph`
**Promoted at:** 2026-06-16T18:14:24.521045Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** N/A
**Reference Quality:** silver

## Purpose

Cross-vendor counterweight to silver_sonnet46_dev_v1_paragraph for #1016 Phase 2b

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
