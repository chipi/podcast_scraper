# Reference: silver_sonnet46_kg_v2_bullets

**Promoted from:** `silver_candidate_anthropic_claudesonnet46_benchmark_v2_bullets`
**Promoted at:** 2026-06-07T12:24:25.953090Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** curated_5feeds_kg_v2
**Reference Quality:** silver

## Purpose

v2-content benchmark silver (#903) - Sonnet 4.6 sweeps GPT-4o 15-0 and beats/ties GPT-5.4. Suffixed _kg_v2 to avoid collision with the autoresearch-framework silver_sonnet46_benchmark_v2_* refs (which point at v1 sources via curated_5feeds_benchmark_v2).

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
