# Reference: silver_gpt4o_benchmark_v1

**Promoted from:** `silver_openai_gpt4o_benchmark_v1`
**Promoted at:** 2026-02-01T20:33:39.981129Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** curated_5feeds_benchmark_v1

## Purpose

Silver reference using GPT-4o for benchmark dataset. High-quality LLM-generated summaries for measuring distance-to-target metrics (ROUGE, similarity) in all future ML model experiments.

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
