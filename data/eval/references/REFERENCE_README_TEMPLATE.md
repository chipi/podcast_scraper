# Reference: {reference_id}

## Reference Quality

- **Type:** {silver|gold}
- **Human-verified:** {yes|no}

## Purpose

This reference represents {description of what this reference is used for}.

It is used to measure distance-to-target metrics (ROUGE, similarity).

## Generation

- **Model:** {model_name}
- **Temperature:** {temperature}
- **Fingerprint:** see `fingerprint.json`

## Dataset

- **Dataset ID:** {dataset_id}
- **Episode Count:** {count}

## Invariants

- Outputs are immutable
- Must not be overwritten
- Comparisons must use the same `dataset_id`
- This artifact is immutable once published

## Replacement Policy

Replace only when:

- A new reference has been explicitly approved
- A new `reference_id` is created (e.g., `{reference_id}_v2`)

**Never update in place.**

## Notes

{Additional notes about this reference, if any}

## Usage

This reference is used for:

- {List of use cases, e.g., "ROUGE computation", "Similarity metrics", "CI blocking" (gold only)}
