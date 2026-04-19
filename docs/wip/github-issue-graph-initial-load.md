# GitHub issue: Graph initial load (copy for new issue)

**Title:** Viewer: graph initial load — graphLens, episode cap, status line, default types

## Problem

- With a corpus path set and API healthy, the viewer loads **all** GI/KG artifacts into the merged graph → hairball on non-trivial corpora.
- There is no **graph-owned** time window; sharing Digest/Library `corpusLens` would tie graph load to “all time” when the user only wanted a wide Library browse.
- No compact **status** for what slice is loaded or how to change the window / expand.

## Scope

- **`graphLens` Pinia store:** seeded once from `corpusLens` on first Graph tab open per session; independent thereafter; if Library is all-time, seed graph to **7d**; reset on corpus path change.
- **Episode cap (15)** on merged graph load for **API sync** and **local multi-file** pickers; shared selection helper; **`(capped)`** when the in-window pool exceeds N.
- **`GET /api/artifacts`:** `publish_date` (YYYY-MM-DD) per row — metadata first, **mtime ingested surrogate** so the field is populated for GI/KG listing.
- **`GraphStatusLine.vue`:** counts, lens control, `data-testid`s per UXS; lens change clears RFC-076 expansion and reloads graph.
- **Default node types:** Quote, Speaker, Episode off by default; “filters active” chip + reset.

## Out of scope

- Cytoscape stylesheet-only work; gesture overlay spec; **app-wide** unified lens/cap for Digest/Library/Dashboard (`graphLens` is Graph-only per WIP).

## Acceptance

- Checkpoints in `docs/wip/GRAPH-INITIAL-LOAD.md` §10.
- `data-testid`s in §5; `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` updated.

## Canonical spec

[`docs/wip/GRAPH-INITIAL-LOAD.md`](../wip/GRAPH-INITIAL-LOAD.md)
