# Datasets

This directory contains canonical, frozen sets of episodes defined in JSON.

## Purpose

Datasets define which episodes from sources are included in evaluation. Each dataset:

- References episodes from `sources/`
- Defines episode selection criteria
- Is versioned (e.g., `curated_5feeds_smoke_v1`, `curated_5feeds_benchmark_v1`)

## Structure

Each dataset is a single JSON file:

```text
datasets/
  {dataset_id}.json
```

## Dataset JSON Format

Each dataset JSON contains:

- `dataset_id` - Unique identifier
- `version` - Dataset version
- `description` - Human-readable description
- `episodes` - Array of episode definitions with:
  - `episode_id` - Unique episode identifier
  - `transcript_path` - Path to transcript in sources
  - `transcript_hash` - SHA256 hash for validation
  - `preprocessing_profile` - Preprocessing profile ID
  - Additional metadata (title, duration, etc.)

## Invariants

- Datasets are immutable once published
- Must not be modified after they're used in baselines or experiments
- Episode paths must reference valid files in `sources/`
- Hashes must match actual file contents
- This artifact is immutable once published

## Do Not

- Modify dataset JSON files after they're used
- Change episode IDs
- Update hashes without updating source files
- Add episodes without versioning the dataset

## Versioning

When creating a new version of a dataset:

1. Create a new JSON file with incremented version (e.g., `curated_5feeds_smoke_v2.json`)
2. Update the `dataset_id` to match the new version
3. Document changes in the description

## Usage

Datasets are used by:

- Baseline creation (`make baseline-create DATASET_ID=...`)
- Experiment runs (`make experiment-run CONFIG=...`)
- Dataset materialization (`make dataset-materialize DATASET_ID=...`)
