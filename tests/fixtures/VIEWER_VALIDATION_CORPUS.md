# Synthetic Tier-3 viewer validation corpus

**GH #774 / RFC-086 Tier-3 CI automation.** Self-contained,
version-pinned corpus generated from the in-repo text fixtures
(`tests/fixtures/rss/` + `tests/fixtures/transcripts/`). The generated
output lives at `tests/fixtures/viewer-validation-corpus/` and gets
**wiped + regenerated** by the script, so this README lives one level
up to survive regeneration.

## Regenerate

```bash
python scripts/build_synthetic_validation_corpus.py
```

Deterministic + idempotent (same fixtures → byte-identical output, modulo
`publish_date` which is relative to "today" so the 7-day graph lens
catches everything). Re-run when text fixtures evolve or when viewer
schema changes.

## Composition

- **9 podcasts** — one canonical RSS each (p01_mtb, p02_software,
  p03_scuba, p04_photo, p05_investing, p06_edge_cases,
  p07_sustainability, p08_solar, p09_biohacking). Variants (`_fast`,
  `_multi`, `_selection`, `_with_transcript`) are skipped.
- **23 episodes** total (up to 3 per podcast, limited by available
  transcripts).
- **GI + KG artifacts** per episode with Episode / Topic / Insight /
  Quote / Entity nodes. Episode nodes carry `metadata_relative_path` +
  `episode_id` properties so the viewer's resolver can map back.
- **5 cross-cutting umbrella topics** (`technology`, `outdoor
  activities`, `gear`, `environment`, `health`) injected into each
  episode's topics so multiple podcasts share topic ids — this enables
  multi-member topic-clusters and would enable digest topic-bands if
  the API ever reads pre-built data.
- **Recent publish dates** (within last 7 days from generation time)
  so the default graph-lens window captures everything.
- **~300 KB total** — well under any size budget.

## Run Tier-3 against it

The API's `--output-dir` must be a parent of (or equal to) the fixture
path. The default `make serve` uses `.test_outputs/` as the root; use
the dedicated target instead.

For V1 / V5 only (no vector index required):

```bash
# Terminal 1
make serve-for-validation

# Terminal 2
make ci-ui-validation CORPUS=$PWD/tests/fixtures/viewer-validation-corpus
```

For the full set (V1, V2, V4, V5 — V3 still cleanly skips), build the
LanceDB index + topic clusters first:

```bash
# One-time prereq (downloads ~80MB MiniLM model on first run):
make preload-ml-models

# Build the in-corpus LanceDB index + topic_clusters.json
make build-validation-index

# Then terminal 1 + terminal 2 as above.
```

## What works against the synthetic corpus

- **V1 — Library row "Open in graph"** ✓ Episode resolves, camera
  centers, full 6-point contract holds.
- **V2 — Digest topic pill "Open in graph"** ✓ (after
  `make build-validation-index`) — the LanceDB index gives
  `/api/corpus/digest` real `kg_topic` + `insight` embeddings to
  derive topic-bands from.
- **V3 — Search "Show on graph"** ✓ (after
  `make build-validation-index`) — the LanceDB index lets semantic search
  return focusable hits with a "Show on graph" affordance; the FSM
  resolves them like any other handoff.
- **V4 — Dashboard topic-cluster chip** ✓ (after
  `make build-validation-index`) — `topic_clusters.json` is built at
  threshold 0.35, calibrated for this corpus's umbrella-topic structure
  (typically yields 2 multi-member clusters — e.g. "outdoor activities"
  with "environment", "technology" with variants).
- **V5 — Hot-state Library → Library** ✓ (FIXED in #775 via
  `EpisodeDetailPanel.openInGraph` microtask retry).

## Why the prereq instead of a checked-in index

The `lance_index/` directory contents are binary / hash-keyed by embedding
model id, so they would churn on every model-version bump and bloat
the repo. Building the index at CI step 0 (or via
`make build-validation-index` locally) is ~30s on a warm runner and
exercises the same indexer code that runs against operator corpora —
making the Tier-3 walk a real end-to-end probe of the indexing path,
not just a viewer walk.

## When to use which corpus

| Use case | Corpus | Notes |
| --- | --- | --- |
| CI Tier-3 smoke (V1, V3-skip, V5, filters) | this synthetic | Self-contained, no external deps |
| Pre-push local validation (V2 + V4 too) | operator-supplied real corpus | Needs vector index |
| Investigation of cross-episode bugs | operator real corpus | Synthetic doesn't reproduce |

The institutional rule from ADR-095 still applies: bugs surfaced by
Tier-3 against ANY corpus must land a Tier-2 matrix-row reproducer
before the fix PR merges.
