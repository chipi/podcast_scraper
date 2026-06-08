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

### Layer-focused naming (#903 convention)

A version suffix marks the source content; a layer suffix marks the eval angle:

| Suffix | Meaning | Example |
| --- | --- | --- |
| `_kg_v2` | Selection optimized for KG entity / cross-episode recurrence signal | `curated_5feeds_kg_v2` |
| `_cil_v2` | Selection optimized for CIL person/topic/org bridging signal | `curated_5feeds_cil_v2` |
| `_cleaning_v2` | Selection optimized for sponsor-block cleaning evaluation | `curated_5feeds_cleaning_v2` |
| `_smoke_v2` | First episode per feed; fast iteration | `curated_5feeds_smoke_v2` |
| `_benchmark_v2` | Held-out scale for autoresearch v2 framework (currently points at v1 sources — see PR #903 note) | `curated_5feeds_benchmark_v2` |
| `_raw_v2` | All episodes from the v2 source set (currently exposed only at the `sources/curated_5feeds_raw_v2/` directory; no dataset JSON yet) | n/a |

Pick the layer suffix that matches the metric you want to baseline against;
all four `_v2` files reference the same 15-episode set, so the difference is
intent + description, not contents.

## Usage

Datasets are used by:

- Baseline creation (`make baseline-create DATASET_ID=...`)
- Experiment runs (`make experiment-run CONFIG=...`)
- Dataset materialization (`make dataset-materialize DATASET_ID=...`)
