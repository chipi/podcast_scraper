# Design note — D8: incremental (unchanged-skip) reindex for the two-tier indexer

**Status:** APPROVED + IMPLEMENTED (2026-08-11) on branch `feat/incremental-processing-and-o11y`.
Companion to `docs/wip/PROD-DEPLOY-ANOMALIES-2026-08-11.md` (§D8 RCA + Implementation status).
Implemented exactly as designed below: fingerprint = sha256 of the collected `(doc_id, text)` rows +
model_id + `_FP_SCHEMA_VERSION`, persisted to `search/episode_fingerprints.json`, skip when matched,
full rebuild ignores fingerprints. Code: `search/two_tier_indexer.py` (`_episode_fingerprint`,
`_episode_scope_key`, `_fingerprint_skip_unchanged`, `_load/_write_episode_fingerprints`,
`_finalize_index_build`) + `search/indexer.py` (wires the real `episodes_skipped_unchanged`). Tests:
`tests/integration/search/test_two_tier_indexer_incremental.py` + the multi-run composition test.
Promote to a numbered ADR when the branch merges.

## Problem (RCA recap)

`build_two_tier_index` (`src/podcast_scraper/search/two_tier_indexer.py`) re-embeds the **entire
corpus** on every reindex. It loops `discover_metadata_files(out)` and calls `_embed()` for every
doc of every episode; there is **no unchanged-skip**. `index_corpus` sets
`episodes_reindexed = tt.episodes` and never populates `episodes_skipped_unchanged` (a dead metric
from the retired FAISS path). `episode_fingerprints.json` does not exist on prod. Upsert is
idempotent for **storage** (merge on id) but not for **compute** — it re-embeds, then merges.

Measured: adding ONE episode to a 107-episode corpus = `vector_index_seconds=635` (~10.6 min),
`episodes_skipped_unchanged=0`. Complexity is **O(corpus)** per add (~50 min at ~500 eps). This
blocks Step-2 volume scaling as hard as D7 (skip-existing) does.

## Goals / Non-goals

**Goals**
- An incremental reindex embeds only **new or changed** episodes → O(changed), not O(corpus).
- Correctness first: a changed episode MUST be re-embedded (never a stale index from a false skip).
- Zero behavior change for a **full** reindex (`rebuild=true` / schema/model change) — still rebuilds
  everything from a clean slate.
- Populate the real `episodes_skipped_unchanged` metric + a log line (feeds D9).

**Non-goals**
- Orphan/deletion sweep on the incremental path (rollback still uses full rebuild — unchanged from
  today; documented limitation).
- Changing the embedding model, chunking, or the tier schema.
- Cross-episode work (topic clusters still rebuild after; that is a separate, cheaper step).

## Design

### Fingerprint = hash of the collected rows (not source mtimes)

`_collect_docs_for_episode` (parse metadata + chunk transcript + read gi/kg) is **cheap**; only
`_embed` is expensive. So we still *collect* every episode's rows each build, then decide whether to
embed:

1. For each episode, run `_collect_docs_for_episode` (as today) to get the `(doc_id, text, meta)`
   rows.
2. Compute `fp = sha256(canonical(sorted (doc_id, text) pairs) + model_id + FP_SCHEMA_VERSION)`.
   Hashing the **collected rows** (the exact text that would be embedded) captures every input that
   affects the index output — transcript, insights, quotes, kg, titles/summaries — in one hash,
   without guessing which source files matter. `model_id` + `FP_SCHEMA_VERSION` make a model or
   indexer change invalidate every fingerprint.
3. Key by the STABLE `index_fingerprint_scope_key(feed_id, episode_id)` (already in
   `corpus_scope.py`).

### Skip decision

- Load `episode_fingerprints.json` (next to the lance index, `search/`) at build start:
  `{scope_key: {"fp": "<sha>", "model_id": "<id>"}}`.
- Per episode: if `not clear_requested` AND stored `fp` == current `fp` AND stored `model_id` ==
  current `model_id` → **skip** embed+upsert for this episode; `stats.episodes_skipped_unchanged += 1`.
  Else embed+upsert (as today) and record the new `fp`.
- After a successful build, write the updated fingerprints file atomically (temp + rename).

### Interactions

- **Full / stale reindex** (`drop_existing` or schema bump → `clear_requested`): ignore fingerprints,
  rebuild all, then write a fresh fingerprints file. Guarantees a clean rebuild always fixes drift.
- **D2 (stats cache):** after any build that changed the index, clear the index-stats perf_cache (or
  bump the lance dir mtime) so `/api/index/stats` reflects the new state. Handled in the D2 change;
  D8 just makes "changed" precise.
- **Incremental vs upsert correctness:** a skipped episode's rows are already in the tables (prior
  build upserted them by the same stable `doc_id`), so skipping embed+upsert leaves them intact.

### Persistence & atomicity

- File: `search/episode_fingerprints.json` (co-located with `lance_index/`, same dir the stats +
  clusters live in). Small (one row per episode).
- Written atomically at the end of a successful build only (a failed build must not poison
  fingerprints → next build re-embeds).

## Edge cases & risks

- **False skip = stale index (the one real risk).** Mitigated by hashing the *collected rows'
  text*, which is exactly the embedded content — if any embedded text changes, the hash changes.
  A re-run that only rewrites unrelated metadata (not embedded) correctly skips (no index impact).
- **model_id / chunking / schema change:** `model_id` is in the hash; `FP_SCHEMA_VERSION` bumps on
  any change to chunking or row construction → global invalidation. A stale-schema reindex already
  sets `clear_requested` → full rebuild.
- **Missing/corrupt fingerprints file:** treated as "no prior fingerprints" → full embed (safe,
  slow) → file rewritten. Self-healing.
- **Deleted episode (rollback):** incremental leaves its rows; same as today. Full rebuild sweeps.
  Documented; unchanged behavior.
- **Concurrent builds:** the existing per-corpus rebuild gate already serializes builds.

## Testing (Tier-2, added with the change)

- Build index for N episodes → assert `episodes_skipped_unchanged=0`, all embedded.
- Re-run with no changes → `episodes_skipped_unchanged=N`, 0 embedded, vectors unchanged, wall time
  bounded (no `_embed` calls — assert via a spy/counter).
- Add 1 episode → exactly 1 embedded, `episodes_skipped_unchanged=N`, new episode searchable.
- Change 1 episode's transcript → exactly that 1 re-embedded.
- `rebuild=true` → ignores fingerprints, embeds all, rewrites file.
- Model-id change in config → all re-embedded.

## Rollout

Ships in the same image as D7/D2/D9. After deploy, re-run Step 0: expect the reindex to embed only
the new episode (seconds, not ~10 min) and `episodes_skipped_unchanged=106`. First reindex after
deploy re-embeds all once (no fingerprints yet) — expected, one-time.
