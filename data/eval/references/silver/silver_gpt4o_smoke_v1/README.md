# Reference: silver_gpt4o_smoke_v1

**Promoted from:** `silver_openai_gpt4o_smoke_v1`
**Promoted at:** 2026-02-13T11:05:58.282770Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** N/A
**Reference Quality:** silver

## Purpose

Silver on smoke for ROUGE (5 episodes)

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
