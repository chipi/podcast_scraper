# Materialized Dataset: {dataset_id}

This directory contains materialized inputs derived from the dataset definition and raw sources.

## Notes

- Contents are generated automatically
- Safe to delete at any time
- Must be byte-for-byte reproducible from:
  - Source dataset: `{source_path_rel}`
  - Preprocessing and chunking configs

## Do Not

- Commit manual edits
- Treat this as authoritative data
- Modify files directly

## Regeneration

To regenerate this materialized dataset:

```bash
make dataset-materialize DATASET_ID={dataset_id}
```

This will:

1. Validate all source paths and hashes
2. Copy transcripts to this directory
3. Generate per-episode `meta.json` files
4. Verify byte-for-byte reproducibility

## Contents

- Transcript files (`.txt`) - Copied from sources with hash validation
- Episode metadata (`{{episode_id}}.meta.json`) - Per-episode validation metadata
- Dataset metadata (`meta.json`) - Dataset-level validation metadata

All files are hash-verified against the source dataset definition.
