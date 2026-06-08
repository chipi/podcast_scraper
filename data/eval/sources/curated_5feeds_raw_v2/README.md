# Source: curated_5feeds_raw_v2

This directory contains raw, immutable inputs collected from five curated podcast feeds — v2 content.

## v1 vs v2

v2 is the second-generation fixture rebuild (PR #902, closes #109/#111/#900). Transcripts are byte-identical to `tests/fixtures/transcripts/v2/` for the 5-feed × 3-episode subset (p01–p05), the same scope `curated_5feeds_raw_v1` covers.

`curated_5feeds_raw_v1` stays frozen. v1 silvers, baselines, and provider comparisons remain valid for their original questions. v2 lives alongside it for the v2-specific eval-track in issue #903.

## What v2 changes vs v1

- **Per-speaker TTS voice mapping** (#111) — speaker labels in the transcript correspond to distinct synthetic voices when re-rendered to audio.
- **Commercial segments** (#109) — every episode carries opening / mid-roll / closing sponsor blocks (~3 per episode) drawn from `SPONSOR_PATTERNS`. The mid-roll uses an `Ad:` speaker label.
- **KG/GIL/CIL content rewrite** (#900) — recurring guests within a podcast, cross-feed topic spans, position arcs, edge-case name ambiguity ("two Marcos" test, Fischer canonicalization test).

See `docs/guides/eval-reports/EVAL_FIXTURES_V2_2026_06_06.md` for the text-derived v1→v2 deltas.

## Contents

- Original v2 transcripts (`.txt` files), byte-identical to `tests/fixtures/transcripts/v2/`
- RSS XML feeds (`.xml` files)
- Episode-level factual metadata (`.metadata.json` files) — speakers + titles unchanged from v1 (v2 preserves the same hosts/guests for the 5-feed subset)
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
