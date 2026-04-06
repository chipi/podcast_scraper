# Idea: Index rebuild / staleness indicator from UI

**Date:** 2026-04-04
**Status:** Parked (not scheduled)
**Context:** Identified during functional gap analysis of GI/KG viewer v2.

## Problem

When users add new episodes or re-run GI extraction, the vector index becomes stale.
The UI has no way to detect or communicate this — search silently returns incomplete
results. Users must leave the browser, run `podcast index` in a terminal, and return.

## Approaches (incremental)

### 1. Staleness indicator (low effort, high value)

Add `mtime` to `ArtifactItem` in `/api/artifacts`. Compare newest artifact mtime
against `index.stats.last_updated`. Show a warning badge on the Dashboard:
"Index may be stale — N artifacts modified since last build."

Read-only, no long-running operations, no risk.

### 2. Trigger-and-forget rebuild button (medium effort, very high value)

`POST /api/index/rebuild` spawns indexing in a background task. UI polls
`/api/index/stats` until `last_updated` changes. Needs:

- Mutex guard (reject if already indexing)
- Timeout / cancellation
- Memory consideration (embedding model loaded in-process)

### 3. Rebuild with SSE progress (higher effort, best UX)

Same as #2 but streams progress via Server-Sent Events:
"Embedding episode 3/12…", "Building FAISS index…", "Done: 1,847 vectors."

Requires progress hooks in the indexing pipeline (not currently present).

### 4. Incremental re-index (high effort, roadmap)

Only index new/changed artifacts instead of full rebuild. Current FAISS pipeline
rebuilds from scratch — incremental adds are easy but updates/deletes need ID tracking.

## Recommendation

Start with **approach 1** (staleness indicator), then **approach 2** (rebuild button).
Approaches 3–4 are future roadmap items.

## Related

- Gap 5 from functional gap analysis (intentionally deferred)
- `src/podcast_scraper/search/corpus_search.py` — indexing pipeline
- `GET /api/index/stats` — current stats endpoint
- `DashboardView.vue` — where the indicator/button would live
