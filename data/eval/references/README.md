# References

This directory contains frozen quality targets for evaluation.

## Structure

References are organized by quality level and task type:

```text
references/
  silver/                  # Silver references (machine-generated)
    {reference_id}/
      predictions.jsonl
      fingerprint.json
      baseline.json
      config.yaml
      README.md
  gold/                    # Gold references (human-verified)
    ner_entities/          # Gold NER references
      {reference_id}/
        index.json
        {episode_id}.json
        README.md
    summarization/         # Gold summarization references
      {reference_id}/
        predictions.jsonl
        README.md
```

## Reference Types

### Silver References

- **Type:** LLM-generated, high-quality targets
- **Task:** Currently used for summarization only
- **Not human-verified**
- **Purpose:** Measure distance-to-target metrics (ROUGE, similarity, coverage ratio)
- **Usage:** Not used for CI blocking
- **Example:** `silver_gpt4o_benchmark_v1`

### Gold References

- **Type:** Human-verified ground truth
- **Task:** Used for both summarization and NER
- **Purpose:** Absolute quality judgment
- **Usage:** Can be used for CI blocking
- **Structure:** Organized by task type:
  - `gold/summarization/{reference_id}/` - Summarization gold references
  - `gold/ner_entities/{reference_id}/` - NER gold references
- **Example:** `gold/ner_entities/ner_entities_smoke_gold_v1`

## Invariants

- References are immutable once published
- Must not be overwritten
- Task type is auto-detected from run when promoting
- This artifact is immutable once published

## Replacement Policy

Replace only when:

- A new reference has been explicitly approved
- A new `reference_id` is created (e.g., `silver_gpt52_v2`)

**Never update in place.**

## Contents

Each reference contains:

- `run.log` - Execution log capturing what actually happened (models loaded, params used, warnings)
- `fingerprint.json` - Complete system fingerprint (reproducibility - what should have happened)
- `baseline.json` - Reference metadata and statistics
- `config.yaml` - Experiment configuration used
- `README.md` - Reference-specific purpose and quality level

**Task-specific files:**

- **Summarization references:**
  - `predictions.jsonl` - Model outputs (summaries) for all episodes

- **NER gold references:**
  - `index.json` - Reference index with episode list and metadata
  - `{episode_id}.json` - Per-episode gold entities with text fingerprints
  - No `predictions.jsonl` (gold references are ground truth, not model outputs)

**Note**: `run.log` is copied from the original run during promotion. It provides diagnostic evidence of execution, complementing `fingerprint.json` which defines the intended configuration.
