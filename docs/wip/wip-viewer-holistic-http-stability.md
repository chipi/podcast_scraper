# WIP: GI/KG viewer holistic HTTP timeout and stale-run guards

Inventory of client-side fetch stability (timeout + generation / single-flight) for the
`web/gi-kg-viewer` app. Use this when adding new API calls or watchers.

> **See also:** [Viewer Frontend Architecture](../architecture/VIEWER_FRONTEND_ARCHITECTURE.md)
> for the stable component tree, store map, and API layer overview. This WIP doc
> focuses on the evolving gate inventory and implementation details.

## Convention (naming and flow)

1. One **`StaleGeneration`** instance per independent async pipeline (e.g. `searchRunGate`,
   `digestLoadGate`). Instantiate at module scope in `<script setup>` or inside a Pinia
   `defineStore` factory so it is shared across calls.
2. At the start of each logical run: **`const seq = gate.bump()`** (returns the new
   monotonic id).
3. After **every** `await` (or before writing derived UI state): if **`gate.isStale(seq)`**,
   return without mutating user-visible state.
4. In **`finally`**: clear loading flags only when **`gate.isCurrent(seq)`**.
5. To drop in-flight work without starting a new numbered run (e.g. corpus path cleared):
   **`gate.invalidate()`** (same as bumping without capturing `seq`).

Prefer **`gate.isStale(seq)`** / **`gate.isCurrent(seq)`** over comparing to a raw counter so
the idiom stays consistent.

## Shared helper

| Item | Detail |
| ---- | ------ |
| Module | `web/gi-kg-viewer/src/utils/staleGeneration.ts` |
| Class | `StaleGeneration` — `bump()`, `invalidate()`, `isCurrent(seq)`, `isStale(seq)` |
| Tests | `web/gi-kg-viewer/src/utils/staleGeneration.test.ts` |

## Shared timeout

| Item | Detail |
| ---- | ------ |
| Module | `web/gi-kg-viewer/src/api/httpClient.ts` |
| Default | `DEFAULT_VIEWER_FETCH_TIMEOUT_MS` (120s); `fetchWithTimeout` maps abort to a clear `Error` |

API modules under `src/api/*.ts`, shell `/api/health` and `/api/artifacts`, transcript
fetches, and artifact JSON reads go through `fetchWithTimeout` (or the same
`AbortSignal.timeout` pattern in one place).

### Health fetch and optimistic flags

`fetchHealth` does **not** reset route availability flags to optimistic “on” at the start of
each call. Overlapping `/api/health` requests therefore avoid a brief incorrect “everything
available” flash; the UI keeps the previous flags until the **winning** response applies
state (or an error clears flags when that run is still current).

## API-level dedupe (cost, not correctness)

**What this is:** Some GET helpers share **one in-flight HTTP request** per logical key (e.g. corpus path) so that **at the same moment** two callers do not open duplicate TCP/API work. It does **not** replace **StaleGeneration**: UI state is still owned by each surface’s gates. Dedupe is only an optimization for **bandwidth, server load, and client connection churn**.

**Implementation:** `web/gi-kg-viewer/src/api/inFlightDedupe.ts` exports **`dedupeInFlight(key, run)`**. Keys are **`GET|<url-with-query>`** so identical URLs share one in-flight promise.

**What is deduped (concurrent identical URL):**

| API module | Functions |
| ---------- | --------- |
| `corpusMetricsApi.ts` | `fetchCorpusStats`, `fetchCorpusRunsSummary`, `fetchCorpusManifest` |
| `corpusLibraryApi.ts` | `fetchCorpusFeeds`, `fetchCorpusEpisodes`, `fetchCorpusEpisodeDetail`, `fetchCorpusSimilarEpisodes` |
| `indexStatsApi.ts` | `fetchIndexStats` |
| `digestApi.ts` | `fetchCorpusDigest` |

**Not deduped:** `POST` / mutations (`postIndexRebuild`), `searchApi`, `exploreApi`, `cilApi`, `artifactsApi` (per-file URLs differ). Add **`dedupeInFlight`** there only if profiling shows hot duplicate concurrent URLs.

**Sequential calls:** If the first request **already finished**, a second call starts a **new** request. Dedupe applies to **concurrency**, not caching.

## Stale-run / single-flight guards

| Surface | Gate(s) | Notes |
| ------- | ------- | ----- |
| Artifacts store | `loadGate` | `loadSelected` / `loadFromLocalFiles` |
| App corpus graph sync | `corpusGraphSyncGate` | `syncMergedGraphFromCorpusApi` |
| Graph canvas | `graphEpisodeOpenGate`, `graphLayoutGate` | Episode open + layout callbacks; catalog metadata resolution passes `shouldCancel` into `resolveEpisodeMetadataViaCorpusCatalog` |
| Dashboard (main) | `dashRefreshGate` | `refreshDashboardMetrics` |
| Dashboard overview | `corpusCatalogStatsGate`, `graphPanelRefreshGate` | `refreshCorpusCatalogStats`; manual **Refresh graph** (`fetchArtifactList` / `loadSelected`) |
| Episode detail | `detailLoadGate` | `loadDetail`, `loadFeedsAndIndex(seq)`, `loadSimilarEpisodes`; watchers `invalidate()` |
| Library | `libraryFeedsGate`, `libraryEpisodesGate` | `loadFeeds` / `loadEpisodes` |
| Digest | `digestLoadGate`, `digestCatalogGate`, `digestGraphOpenGate` | `loadDigest` / `loadFeedsCatalog`; **Open in graph** on topic hits. Corpus/health watch `digestGraphOpenGate.invalidate()` |
| Shell | `healthFetchGate`, `artifactListFetchGate` | `fetchHealth`, `fetchArtifactList` |
| Search store | `searchRunGate` | `runSearch` (bump after validation) |
| Explore store | `exploreRunGate` | Shared for filtered + NL runs |
| Index stats store | `indexStatsRefreshGate`, `indexRebuildGate` | `refreshIndexStats`, `requestIndexRebuild` |
| Transcript dialog | `transcriptOpenGate` | Main transcript + segments sidecar |
| Topic timeline dialog | `timelineLoadGate` | CIL topic timeline fetch |

Also documented historically: `DashboardView` dashboard refresh, `GraphCanvas` safe watchers,
`artifacts` single-flight (now via `StaleGeneration`).

## Tests

| File | Intent |
| ---- | ------ |
| `src/api/inFlightDedupe.test.ts` | `dedupeInFlight` concurrent vs sequential behavior |
| `src/api/httpClient.test.ts` | Timeout / abort error shaping |
| `src/utils/staleGeneration.test.ts` | `StaleGeneration` behavior |
| `src/stores/artifacts.loadSelected.test.ts` | Artifact load single-flight |
| `src/stores/shell.fetchArtifactList.overlap.test.ts` | Overlapping artifact list fetches |
| `src/stores/search.runSearch.overlap.test.ts` | Overlapping search runs |
| `src/api/corpusMetricsApi.test.ts` | Concurrent `fetchCorpusStats` dedupes to one HTTP call |

## Related

See also `docs/wip/wip-viewer-corpus-load-graph-stability.md` for corpus path / graph load
behavior.
