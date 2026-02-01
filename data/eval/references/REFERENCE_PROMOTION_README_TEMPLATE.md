# Reference: {promoted_id}

**Promoted from:** `{run_id}`
**Promoted at:** {promoted_at}Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** {dataset_id}{quality_section}

## Purpose

{reason}

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
