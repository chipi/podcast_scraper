# WIP: Viewer corpus load and graph interaction stability

**Status:** Draft
**Created:** 2026-04-15
**Context:** Large multi-feed corpora (many `*.gi.json` / `*.kg.json`), DevTools MCP session on `localhost:5173`.

> **See also:** [Viewer Frontend Architecture](../architecture/VIEWER_FRONTEND_ARCHITECTURE.md)
> for the stable component tree and store map.

## Symptoms observed

1. After loading a large corpus and a few UI actions, the app feels unresponsive.
2. **Corpus** panel: **Load into graph** shows **Loading…** and stays disabled for a long time or appears stuck.
3. Console: **Unhandled error during execution of watcher callback** on **`GraphCanvas`**, followed by **Uncaught (in promise)**.
4. Console: **404** on some resources; **Vue** warnings about missing properties on **`NodeDetail`** that do not appear in current source (suggests stale HMR or mismatched bundle).
5. Many **cytoscape** wheel-sensitivity warnings (expected given current `wheelSensitivity`).

## Root cause hypotheses

| Area | Hypothesis |
| ---- | ---------- |
| Artifact load | `loadSelected()` in `artifacts.ts` clears state and **sequentially** `await`s each file. Corpus auto-sync selects **all** GI/KG paths; tens of episodes imply **tens of round-trips** and a long window where `artifacts.loading` is true. |
| Concurrency | `syncMergedGraphFromCorpusApi()` (watcher on corpus path + health) and manual **Load into graph** can both call `loadSelected()` without a **single-flight** guard; overlapping runs can clear `parsedList` mid-flight and confuse `loading` / graph state. |
| Graph | A **`watch`** callback in `GraphCanvas.vue` throws or triggers a **rejected promise** (e.g. during `redraw()`, focus, degree filter, minimap). Vue reports watcher failure; graph may be left inconsistent. |
| Client bundle | **NodeDetail** property warnings for names absent from repo → treat as **stale tab** until reproduced after hard refresh on a clean dev session. |
| Network | Hung or very slow **`GET /api/artifacts/...`** would keep `loading` true until timeout (if any) or forever (if no timeout). |

## Goals

1. **Predictable loading:** User always sees progress, can cancel or scope work, and does not assume a freeze.
2. **No overlapping loads:** At most one logical “load artifacts into store” operation at a time; stale async work bails out cleanly.
3. **Resilient graph:** Watchers and `redraw()` paths do not throw uncaught; failures log once and degrade gracefully.
4. **Verify:** Reproduce on acceptance-sized corpus (40+ episodes, two feeds) with Network + console clean aside from known noise.

## Phase A — Hardening (P0)

**A1. Single-flight / generation token for `loadSelected`**

- In `web/gi-kg-viewer/src/stores/artifacts.ts`, add an internal monotonic **generation** (or `AbortController` per load).
- At start of `loadSelected`, bump generation; after each `await`, if generation changed, **abort** (stop fetching, optionally discard partial `out`).
- Ensure **`loading`** is cleared only by the **winning** run’s `finally`, or use a ref counting / “last started wins” pattern so a completed stale run cannot clear `loading` while a newer run is active.

**A2. Serialize corpus auto-sync with manual load**

- In `web/gi-kg-viewer/src/App.vue`, `syncMergedGraphFromCorpusApi` already uses `corpusGraphSyncGen`; extend the same idea so **manual** `onLoadIntoGraphClick` either:
  - bumps the same generation before calling `loadSelected`, or
  - shares one **`loadArtifactsFromCorpus(reason: 'sync' | 'manual')`** helper that owns the mutex.

**A3. GraphCanvas watchers: try/catch + log**

- Wrap bodies of watchers that call **`redraw()`**, **`tryApplyPendingFocus`**, **`applyDegreeVisibility`**, **`setupNavigator`**, etc., in **`try/catch`** (and for `nextTick` / async paths, **`.catch()`** on promises).
- Log with a single prefix (e.g. `[GraphCanvas]`) at **`console.warn`** or **`console.error`** with a short message; avoid rethrowing from watchers.

**A4. Optional: guard `redraw()` entry**

- If `container.value` is missing or unmounted, return early (some paths may already do this; audit all callers).

## Phase B — UX and perceived performance (P1)

**B1. Progress surface**

- Expose **`loadedCount` / `totalCount`** (or indeterminate **spinner + label**) while `loadSelected` runs so **Loading…** is not silent.
- Consider **cap** or **warning** when selection exceeds N files (e.g. “Loading 80 files; consider Digest/Library for browsing”), without blocking power users.

**B2. Concurrency on the server (optional, larger change)**

- Add a **batch** or **multi-get** API for artifact JSON (single request for many relative paths) to cut round-trips. Requires FastAPI route design, size limits, and tests under `tests/unit/podcast_scraper/server/` and `tests/integration/server/`. Defer if Phase A is enough.

**B3. Timeouts**

- Ensure `fetchArtifactJson` (or `fetch` wrapper) uses a **timeout** and surfaces **`loadError`** so `loading` always clears.

## Phase C — Cleanup and verification (P2)

**C1. Reproduce NodeDetail warnings**

- Hard refresh (cache bust). If warnings persist, grep built output / source map; fix any template/bindings that reference removed computeds.

**C2. 404 inventory**

- From DevTools Network, list failing URLs; fix asset paths or proxy if they are app bugs; ignore if extension or font noise.

**C3. Tests**

- **Unit:** artifacts store: overlapping `loadSelected` calls → final state matches last run; `loading` false after rejection.
- **Unit or component:** GraphCanvas watcher with mocked `cy` / `gf` throwing → no uncaught rejection (spy on `console.warn`).
- **E2E (if feasible):** small fixture corpus, assert **Load into graph** enables after load; optional Playwright step for large corpus marked slow.

## Suggested order of work

1. **A1 + A2** (store + App sync) — highest impact for “stuck UI” and races.  
2. **A3** — stops watcher errors from destabilizing the graph.  
3. **B1 + B3** — user-visible feedback and hung-request recovery.  
4. **B2** — only if load time remains unacceptable after batching on client logic.  
5. **C1–C3** — polish and regression gates.

## Out of scope (for this plan)

- Changing **cytoscape** `wheelSensitivity` (product decision; warnings are informational).
- Full **RFC** unless batch API or security model for multi-file read changes materially.

## References

- `docs/wip/wip-viewer-holistic-http-stability.md` — shared timeouts, `StaleGeneration`, full viewer async inventory  
- `web/gi-kg-viewer/src/stores/artifacts.ts` — `loadSelected`, `loading`  
- `web/gi-kg-viewer/src/App.vue` — `syncMergedGraphFromCorpusApi`, `onLoadIntoGraphClick`  
- `web/gi-kg-viewer/src/components/graph/GraphCanvas.vue` — `watch`, `redraw`  
- `docs/guides/DEVELOPMENT_GUIDE.md` — viewer / serve workflow  
- `e2e/E2E_SURFACE_MAP.md` — update if load UI or testids change
