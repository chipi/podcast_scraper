# Synthetic Tier-3 viewer validation corpus

**GH #774 / RFC-086 Tier-3 CI automation.** Self-contained, version-pinned
corpus generated from the in-repo text fixtures (`tests/fixtures/rss/` +
`tests/fixtures/transcripts/`). Lives here so CI / scheduled GHA can run
the Tier-3 validation walk against a stable corpus without external
dependencies or operator setup.

## Regenerate

```bash
python scripts/build_synthetic_validation_corpus.py
```

Deterministic + idempotent (same fixtures → byte-identical output, modulo
the "publish_date relative to today" bit). Re-run when the text fixtures
evolve or when the viewer schema changes.

## Composition

- **9 podcasts** — one canonical RSS each (p01_mtb, p02_software, p03_scuba,
  p04_photo, p05_investing, p06_edge_cases, p07_sustainability, p08_solar,
  p09_biohacking). Variants (`_fast`, `_multi`, `_selection`,
  `_with_transcript`) are skipped.
- **23 episodes** total (up to 3 per podcast, limited by available
  transcript files).
- **GI + KG artifacts** per episode with Episode / Topic / Insight / Quote
  / Entity nodes. Episode nodes carry `metadata_relative_path` +
  `episode_id` properties so the viewer's resolver can map back to the
  Library row.
- **Recent publish dates** (within last 7 days from generation time) so
  the default graph-lens window captures everything.
- **~240 KB total** — well under any size budget; checked into the repo.

## Run Tier-3 against it

The dev server's `--output-dir` must be a parent of (or equal to) the
fixture path. The repo provides `make serve-for-validation` which starts
the API rooted at the repo root, so this fixture is reachable:

```bash
# Terminal 1
make serve-for-validation

# Terminal 2
make ci-ui-validation CORPUS=$PWD/tests/fixtures/viewer-validation-corpus
```

## What works against the synthetic corpus

- **V1 — Library row "Open in graph"** ✓ Episode resolves, camera centers,
  Library handoff full L0+L1+L2+L3+L5+L6 contract holds.
- **V3 — Search "Show on graph"** — skipped (no vector index, as
  expected per RFC-086).

## Known gaps

- **V2 — Digest topic pill** ✗ Synthetic corpus has no `topic_bands`
  with cross-episode hits (text fixtures have low topic intermingling —
  each transcript covers a distinct hobby topic with no overlap). Needs
  richer mock data; tracked as a sub-issue under #774.
- **V4 — Dashboard topic-cluster chip** ✗ Same reason — no multi-member
  topic clusters in the synthetic data.
- **V5 — Hot-state Library → Library** ✗ Triggers the separate
  `EpisodeDetailPanel.openInGraph` timing bug (filteredArtifact lag after
  `appendRelativeArtifacts` — tracked in #775).

## When to use which corpus

| Use case | Corpus | Notes |
| --- | --- | --- |
| CI Tier-3 smoke (schema + V1 contract) | this synthetic | Self-contained, no external deps |
| Pre-push local validation | operator-supplied real corpus | Catches V2/V4/V5 + drift |
| Investigation of cross-episode bugs | operator real corpus | Synthetic doesn't reproduce |

The institutional rule from ADR-095 still applies: bugs surfaced by
Tier-3 against ANY corpus must land a Tier-2 matrix-row reproducer before
the fix PR merges.
