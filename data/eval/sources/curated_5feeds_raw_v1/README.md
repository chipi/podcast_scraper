# Source: curated_5feeds_raw_v1

This directory contains raw, immutable inputs collected from five curated podcast feeds.

## Contents

- Original transcripts as produced by the ingestion pipeline (`.txt` files)
- RSS XML feeds (`.xml` files)
- Episode-level factual metadata (`.metadata.json` files)
- Source inventory (`index.json` with content hashes)

## Guarantees

- Files are unmodified after ingestion
- No preprocessing, cleanup, or normalization is applied
- Content hashes are recorded in `index.json` for drift detection
- This artifact is immutable once published

## Do Not

- Edit transcripts
- Add derived data
- Store model outputs here
- Modify metadata files

## Source Feeds

This source contains episodes from five podcast feeds:

- Feed P01 (MTB/trail building)
- Feed P02 (Software development)
- Feed P03 (Scuba diving)
- Feed P04 (Photography)
- Feed P05 (Investing)

Each feed directory contains:

- Transcript files (`.txt`)
- Episode metadata (`.metadata.json`)
- RSS XML feed (`.xml`)

## Validation

The `index.json` file provides:

- Complete inventory of all episodes
- SHA256 hashes for integrity verification
- Source paths and metadata references

Use this index for programmatic dataset generation and drift detection.
