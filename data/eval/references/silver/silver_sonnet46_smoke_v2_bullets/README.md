# Reference: silver_sonnet46_smoke_v2_bullets

**Promoted from:** `silver_candidate_anthropic_claudesonnet46_smoke_v2_bullets`
**Promoted at:** 2026-06-07T12:01:50.841223Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** curated_5feeds_smoke_v2
**Reference Quality:** silver

## Purpose

v2 silver re-selection (#903) on regenerated v2 - Sonnet 4.6 wins

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
