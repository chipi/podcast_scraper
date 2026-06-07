# Reference: silver_sonnet46_smoke_v2

**Promoted from:** `silver_candidate_anthropic_claudesonnet46_smoke_v2_paragraph`
**Promoted at:** 2026-06-07T11:00:17.050586Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** curated_5feeds_smoke_v2
**Reference Quality:** silver

## Purpose

v2 silver re-selection (#903) - Sonnet 4.6 sweeps GPT-4o 5-0 and GPT-5.4 4-0-1 on v2 paragraph smoke

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
