# Sources

This directory contains raw, immutable inputs collected from podcast feeds and other sources.

## Purpose

Sources are the ground truth inputs for all evaluation work. They contain:

- Original transcripts as produced by the ingestion pipeline
- RSS XML feeds with episode metadata
- Episode-level factual metadata (no labels, no annotations)

## Structure

Each source is stored in its own directory:

```text
sources/
  {source_id}/
    {feed_name}/
      {episode_id}.txt
      {episode_id}.metadata.json
      {feed_name}.xml
    index.json
    README.md
```text

## Invariants

- Sources are immutable once published
- Files are unmodified after ingestion
- No preprocessing, cleanup, or normalization is applied
- Content hashes are recorded in `index.json` for drift detection
- This artifact is immutable once published

## Do Not

- Edit transcripts
- Add derived data
- Store model outputs here
- Modify metadata files

## Metadata Schema

Episode metadata files (`.metadata.json`) follow a JSON schema defined in `../schemas/episode_metadata.schema.json`.

Current version: **1.0**

The schema separates:
- **Facts** (`speakers`): Who participated in the episode
- **Expectations** (`expectations`): What should/shouldn't appear in generated outputs

See `../schemas/episode_metadata.schema.json` for the complete schema definition.

## Source Index

Each source directory contains an `index.json` file that provides:

- Complete inventory of all episodes
- SHA256 hashes for integrity verification
- Source paths and metadata references

Use this index for:

- Programmatic dataset generation
- Drift detection
- Validation

## Validation

Before using a source:

1. Verify `index.json` exists and is valid
2. Check that all referenced files exist
3. Validate hashes match actual file contents
