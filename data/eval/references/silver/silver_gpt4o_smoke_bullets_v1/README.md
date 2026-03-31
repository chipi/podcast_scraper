# Reference: silver_gpt4o_smoke_bullets_v1

**Promoted from:** `experiment_openai_gpt4o_smoke_bullets_v1`
**Promoted at:** 2026-03-31T09:00:52.581646Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** N/A
**Reference Quality:** silver

## Purpose

GPT-4o JSON bullet summaries on curated_5feeds_smoke_v1 bullet-aligned ROUGE for autoresearch

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
