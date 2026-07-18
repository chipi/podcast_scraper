# graph-v3 — corpus artifact reproducibility

Notes on regenerating the enrichment artifacts the graph-v3 lenses consume,
so a fresh clone of `podcast_scraper-FUTURE` doesn't rely on hand-copied
JSON from a sibling worktree.

## Current situation

During graph-v3 tier 4 (theme-cluster regions) and tier 5C (enricher-based
lenses), the artifacts under
`.test_outputs/manual/prod-v2/corpus/enrichments/` for prod-v2 were sourced
by `cp` from the sibling `podcast_scraper-ai-ml-improvements` worktree,
which had already run the full enrichment pipeline against the same episode
set. `.test_outputs/` is `.gitignore`d, so nothing about that copy is under
version control — a fresh clone will not have these files, and any lens
that depends on them (theme regions, velocity halo, credibility border,
insight-sentiment tint, consensus edges) will silently disable (enricher
gating hides the toggle).

## Regenerating from scratch

To regenerate any of the deterministic enrichers against a live corpus:

```bash
# All 7 deterministic enrichers, corpus-scope only (skip per-episode passes):
make enrich CORPUS=.test_outputs/manual/prod-v2/corpus CORPUS_ONLY=1

# Just theme clusters (tier 4):
make enrich CORPUS=.test_outputs/manual/prod-v2/corpus \
    ONLY=topic_theme_clusters CORPUS_ONLY=1

# temporal_velocity + grounding_rate + guest_coappearance (tier 5C):
make enrich CORPUS=.test_outputs/manual/prod-v2/corpus \
    ONLY=temporal_velocity,grounding_rate,guest_coappearance CORPUS_ONLY=1

# Per-episode sentiment sidecars (tier 5C-3 — writes ONE file per episode
# under {run}/metadata/enrichments/{stem}.insight_sentiment.json):
make enrich CORPUS=.test_outputs/manual/prod-v2/corpus ONLY=insight_sentiment
```

The `topic_theme_clusters` enricher expects `>=2` episodes per co-occurring
pair (default `min_pair`), so it emits 0 clusters on a 3-episode pilot. On
prod-v2 (209 episodes) it emits 6 clusters (interest rates, ai agents,
future of work, quantum computing, employee engagement, tech industry) —
matches the sibling worktree's output, confirming determinism.

## Enricher inventory + graph-v3 lens dependencies

| Enricher | Artifact | Consuming lens |
|---|---|---|
| `topic_theme_clusters` | `enrichments/topic_theme_clusters.json` | `themeClusterRegions` (region tints), tier 4 |
| `topic_cooccurrence_corpus` | `enrichments/topic_cooccurrence_corpus.json` | node-detail co-occurrence table |
| `topic_similarity` | `enrichments/topic_similarity.json` | `aggregatedEdges` panel (search too) |
| `topic_consensus` | `enrichments/topic_consensus.json` | `consensusEdges` lens (tier 5D-1) |
| `temporal_velocity` | `enrichments/temporal_velocity.json` | `velocityHalo` lens (tier 5C-1); Digest trend arrows |
| `grounding_rate` | `enrichments/grounding_rate.json` | `personCredibility` lens (tier 5C-2); Person node detail |
| `guest_coappearance` | `enrichments/guest_coappearance.json` | `coGuestEdges` lens (tier 5D-2); Person node detail |
| `insight_density` | `metadata/enrichments/{stem}.insight_density.json` | Episode enrichment section — no graph lens |
| `insight_sentiment` | `metadata/enrichments/{stem}.insight_sentiment.json` | Player conversation-arc + position-arc tints (joined into arc responses server-side by `cil_queries._attach_sentiment`). Tier 5C-3 graph lens **descoped — closed**; sentiment belongs on the arc panels, not on hidden-at-low-zoom Insight graph nodes. See graph-v3/SUMMARY.md § "Not done" for the rationale. |

## Notes

- Every graph-v3 lens toggle is enricher-gated: if the underlying artifact
  is missing, the row is hidden from the Lenses popover so no dead
  controls surface. This means "did the enricher run?" and "can I use
  this lens?" are the same question — no separate feature flag needed.
- If new enricher artifacts are added to a corpus while `make serve` is
  running, the artifacts store rehydrates on corpus-path change (footer
  input) or full page reload — no restart needed.

## Longer-term escape

The right final state is: the enrichment pipeline runs against prod-v2 in
this worktree (not just the sibling) so the artifacts live at the same
path they'd land on any fresh clone.

## Tests-facing fixture — regenerable via `make`

The fixture graph-v3 lens tests + tier-3 real-corpus walks consume lives
at `tests/fixtures/viewer-validation-corpus/v3/enrichments/`. **These
files ARE committed to git** — a fresh clone works without running any
enrichers. But when an enricher's output schema changes, someone must
rerun the fixture:

```bash
make enrich-viewer-fixture
git diff tests/fixtures/viewer-validation-corpus/v3/enrichments/
# review the diff, then either commit or investigate the drift
```

The target is `.PHONY` and wraps `make enrich CORPUS=... WITH_ML=1
CORPUS_ONLY=1`. Requires the venv to have `.[ml,search]` extras
installed (for `topic_similarity` and `topic_consensus`).

**Verified 2026-07-17**: the bundle-discovery bug from the original
`handover-theme-clusters.md` is fixed here too. Direct check:

```bash
PYTHONPATH=src python3 -c "
from podcast_scraper.enrichment.paths import discover_episode_bundles
from pathlib import Path
print(len(discover_episode_bundles(Path('.test_outputs/manual/prod-v2/corpus'))))
"
# → 99
```

## End-to-end reproducibility run (2026-07-17)

Ran the full enrichment set locally to prove reproducibility.

```bash
make enrich CORPUS=.test_outputs/manual/prod-v2/corpus \
    WITH_ML=1 CORPUS_ONLY=1 PROFILE=cloud_balanced
```

**Result: 8 of 9 corpus-scope enrichers regenerated cleanly**, replacing the
sibling-worktree `cp`s under `.test_outputs/manual/prod-v2/corpus/enrichments/`.

| Enricher | Result | Notes |
|---|---|---|
| `topic_cooccurrence_corpus` | ✓ | 1.54 MB, deterministic |
| `topic_theme_clusters` | ✓ | 6 clusters — semantically identical to the sibling copy |
| `grounding_rate` | ✓ | 125 persons |
| `guest_coappearance` | ✓ | 87 pairs |
| `topic_similarity` | ✓ | 1 MB, ML tier (sentence-transformers `all-MiniLM-L6-v2` local) |
| `topic_consensus` | ✓ | 22 rows, ML tier (`all-MiniLM-L6-v2` + `deberta-v3-small` NLI local, ~2 min) |
| `temporal_velocity` | ✓ | v1.2.0 datetime-tz bug fixed on this branch (2026-07-17); re-verified locally against prod-v2 → `status=ok records=833`. Framework's `enrichment.enricher.completed` event now carries `error` / `error_class` when status ≠ ok. |
| `insight_density` | (skipped) | Episode-scope, filtered out by `--corpus-only` |
| `insight_sentiment` | (skipped) | Same |

**Delta from sibling `cp`**: nothing structural.
`nli_contradiction.json` (deprecated by ADR-108, superseded by `topic_consensus`)
is left untouched.

**Fixed on this branch (2026-07-17):** `temporal_velocity` v1.2.0 was
raising `TypeError: can't compare offset-naive and offset-aware datetimes`
inside `_full_week_axis` on prod-v2 because publish-date strings mix
date-only ISO (`2026-06-27` → naive) with datetime-plus-Z (`…Z` → aware).
Fix: coerce naive → UTC before the `min`/`max`. Re-verified against prod-v2
locally: `status=ok records=833`. Alongside, the enrichment framework's
`build_enricher_completed` event builder now propagates `error` /
`error_class` from `EnricherResult` on non-ok terminal outcomes so future
silent regressions surface with a diagnostic payload (was: framework
dropped them on the floor).
