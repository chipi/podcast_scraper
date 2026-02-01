# References

This directory contains frozen quality targets for evaluation.

## Structure

References are organized by dataset:

```text
references/
  {dataset_id}/
    {reference_id}/
      predictions.jsonl
      fingerprint.json
      baseline.json
      config.yaml
      README.md
```text

## Reference Types

### Silver References

- **Type:** LLM-generated, high-quality targets
- **Not human-verified**
- **Purpose:** Measure distance-to-target metrics (ROUGE, similarity)
- **Usage:** Not used for CI blocking
- **Example:** `silver_gpt52_v1`

### Gold References

- **Type:** Human-verified ground truth
- **Purpose:** Absolute quality judgment
- **Usage:** Can be used for CI blocking
- **Example:** `gold_human_v1`

## Invariants

- References are immutable once published
- Must not be overwritten
- Comparisons must use the same `dataset_id`
- This artifact is immutable once published

## Replacement Policy

Replace only when:

- A new reference has been explicitly approved
- A new `reference_id` is created (e.g., `silver_gpt52_v2`)

**Never update in place.**
