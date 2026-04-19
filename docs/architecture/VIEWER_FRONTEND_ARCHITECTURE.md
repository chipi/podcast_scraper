# Viewer Frontend Architecture

> **Scope:** Internal architecture of the GI/KG browser viewer SPA
> (`web/gi-kg-viewer/`). For the FastAPI backend see the
> [Server Guide](../guides/SERVER_GUIDE.md); for visual/UX contracts see
> [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) and the
> [UXS index](../uxs/index.md); for the original design rationale see
> [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md).

## Stack

| Layer | Technology | Notes |
| ----- | ---------- | ----- |
| Framework | Vue 3 (Composition API, `<script setup>`) | No Options API |
| Build | Vite | Dev server on port 5173; Playwright uses 5174 |
| State | Pinia | One store per domain; no Vuex |
| Styling | Tailwind CSS + semantic `--ps-*` tokens | Tokens defined in UXS-001 |
| Graph | Cytoscape.js | Single canvas, custom stylesheet in `cyGraphStylesheet.ts` |
| Charts | Chart.js | Dashboard only; registered once via `chartRegister.ts` |
| Language | TypeScript (strict) | Vitest for unit tests |
| E2E | Playwright (Firefox) | Surface map in `e2e/E2E_SURFACE_MAP.md` |

No Vue Router is used. Navigation is tab-state driven (see Shell below).

## Component tree

High-level layout matches **`App.vue`** (no Vue Router): **header** → **collapsible left column**
(**`LeftPanel`**) → **center** (main tab roots) → **collapsible right column** (**`SubjectRail`**)
→ **footer** **`StatusBar`**.

```text
App.vue
+-- [header]  Main views (Digest | Library | Graph | Dashboard); theme cycle; kbd hints (/ search, Esc clear)
+-- [banners]  Optional alerts (e.g. sibling merge)
|
+-- [main row]
|   +-- [left column]  Collapsible w-72 / w-8; chevron toggle; when collapsed, vertical "Search" affordance
|   |   +-- LeftPanel
|   |       +-- SearchPanel  (#search-q, results, advanced modal, ResultCard, …)
|   |       +-- ExplorePanel (section under Search)
|   |
|   +-- [center]  mainTab drives which root is mounted (keep-alive where used)
|   |   +-- DigestView
|   |   +-- LibraryView
|   |   +-- DashboardView
|   |   |   +-- CorpusDataWorkspace  (artifacts list, API health, Data cards — data-testid corpus-data-workspace)
|   |   |   +-- DashboardOverviewSection, chart components, …
|   |   +-- GraphTabPanel  (graph canvas + toolbar when Graph tab)
|   |
|   +-- [right column]  Collapsible w-96 / w-8; collapsed shortcuts Search / Explore / Details (graph)
|       +-- SubjectRail  (subject.kind: null | 'episode' | 'graph-node' | 'topic' | 'person')
|           +-- GraphNodeRailPanel     (subject.kind === 'graph-node')
|           |   +-- NodeDetail (embed-in-rail), GraphConnectionsSection, GraphNeighborhoodMiniMap, …
|           +-- Episode region + EpisodeDetailPanel  (subject.kind === 'episode')
|           |   +-- Optional Details / Neighbourhood tablist on Graph tab + GraphConnectionsSection slot
|           +-- Placeholder copy  (topic / person kinds when not implemented)
|
+-- [footer]  StatusBar  (corpus path, health, offline files — shell store)
```

### Shared components (`components/shared/`)

| Component | Used by |
| --------- | ------- |
| `HelpTip` | App, DigestView, LibraryView, ExplorePanel, SearchPanel, EpisodeDetailPanel, NodeDetail, TopicTimelineDialog |
| `PodcastCover` | DigestView, LibraryView, EpisodeDetailPanel, TopicTimelineDialog |
| `CollapsibleSection` | LibraryView |
| `TranscriptViewerDialog` | NodeDetail |
| `TopicTimelineDialog` | NodeDetail |

### Dialogs

All dialogs use the native `<dialog>` element. `SearchPanel` also contains an
inline `<dialog>` for advanced search options (not extracted to its own file).

## Shell and navigation

There is **no Vue Router**. The app is a single-page shell with a **main tab** plus
**subject** state:

| Axis | Ref / store field | Values |
| ---- | ----------------- | ------ |
| Main view | `App.vue` local `mainTab` | `digest`, `library`, `graph`, `dashboard` |
| Left query column | `LeftPanel.vue` (no tab switcher) | `SearchPanel` + `ExplorePanel` stacked |
| Right subject rail | `subject.kind` (`stores/subject.ts`) | `graph-node`, `episode`, `topic`, `person`, or empty |

The **shell store** owns `corpusPath`, health status, and feature-availability
flags. Changing `corpusPath` triggers cascading refreshes across stores
(artifacts, library, digest, dashboard, index stats).

## Pinia store map

```text
  shell  ─────────────>  (health, corpus path, feature flags, artifact list)
    |
    |  corpusPath drives
    v
  artifacts  ─────────>  (GI/KG JSON files, parsed graph, bridge doc)
    |
    |  displayArtifact drives
    v
  graphFilters  ───────>  (filter state, filtered artifact view)

  graphNavigation  ────>  (pending focus, library highlights, ego focus)
  graphExplorer  ──────>  (layout preference, minimap, degree bucket)

  search  ─────────────>  (query, results, filters)
       \__ uses graphNavigation (highlight resets)

  explore  ────────────>  (NL + filtered explore, insights, leaderboard)

  indexStats  ─────────>  (FAISS index envelope, rebuild polling)
       \__ uses shell (corpus path, health gating)

  subject      ────────>  (which subject kind, metadata path, graph node id, …)

  corpusLens  ─────────>  (date filter for corpus-wide views)
  theme  ──────────────>  (dark / light / system cycle)
```

### Cross-store dependencies

Most stores are **independent**. The intentional couplings are:

- `graphFilters` watches `artifacts.displayArtifact`
- `indexStats` reads `shell.corpusPath` and `shell.healthStatus`
- `search.clearResults` clears `graphNavigation` library highlights
- `App.vue` orchestrates corpus path changes across shell, artifacts, and
  downstream stores

## API layer

All HTTP calls go through **`src/api/`** modules. No component or store calls
`fetch` directly.

### Infrastructure

| Module | Purpose |
| ------ | ------- |
| `httpClient.ts` | `fetchWithTimeout` wrapper (default 15 s); `isAbortOrTimeout` classifier. Only module that calls `fetch`. |
| `inFlightDedupe.ts` | `dedupeInFlight(key, run)` — shares one in-flight promise per identical `GET` URL. Concurrency optimization, not a cache. |

### Feature modules

| Module | Functions | Dedupe | Timeout |
| ------ | --------- | ------ | ------- |
| `artifactsApi` | `fetchArtifactJson` | No (per-file URLs differ) | Yes |
| `searchApi` | `searchCorpus` | No | Yes |
| `exploreApi` | `fetchExploreFiltered`, `fetchExploreNaturalLanguage` | No | Yes |
| `cilApi` | `fetchTopicTimeline`, `fetchTopicPersons` | No | Yes |
| `corpusLibraryApi` | `fetchCorpusFeeds`, `fetchCorpusEpisodes`, `fetchCorpusEpisodeDetail`, `fetchCorpusSimilarEpisodes` | Yes | Yes |
| `corpusMetricsApi` | `fetchCorpusStats`, `fetchCorpusRunsSummary`, `fetchCorpusManifest` | Yes | Yes |
| `indexStatsApi` | `fetchIndexStats`, `postIndexRebuild` | GET only | Yes |
| `digestApi` | `fetchCorpusDigest` | Yes | Yes |

`POST` / mutation endpoints (`postIndexRebuild`) are never deduped.

### Convention

- **Key format for dedupe:** `GET|/api/<route>?<query>` — same trimmed URL
  shares one promise.
- **Error shape:** API wrappers throw on non-2xx; `corpusLibraryApi` adds an
  upgrade hint on 404 for older servers.
- **AbortSignal:** `fetchWithTimeout` passes an `AbortSignal` to `fetch`;
  stores can cancel via `StaleGeneration`.

## Async correctness

### StaleGeneration pattern

Every async pipeline that updates UI state is guarded by a **`StaleGeneration`**
instance (defined in `src/utils/staleGeneration.ts`). The pattern:

1. **Bump** the gate before starting work (`gate.bump()` returns a new
   sequence number).
2. After each `await`, check **`gate.isStale(seq)`**. If true, abandon the
   result.
3. UI state is only written when the sequence is still current.

This prevents stale responses from overwriting fresher data when the user
changes context (corpus path, search query, selected episode) while a request
is in flight.

### Gate inventory

A full per-surface gate table is maintained in the
[WIP holistic HTTP stability doc](../wip/wip-viewer-holistic-http-stability.md).
Key surfaces:

| Surface | Gate(s) | Notes |
| ------- | ------- | ----- |
| Artifacts | `loadGate` | `loadSelected` / `loadFromLocalFiles` |
| Shell | `healthFetchGate`, `artifactListFetchGate` | Health does not optimistically flash flags |
| Graph canvas | `graphEpisodeOpenGate`, `graphLayoutGate` | Catalog metadata passes `shouldCancel` |
| Dashboard | `dashRefreshGate` | Main refresh; overview has separate gates |
| Library | `libraryFeedsGate`, `libraryEpisodesGate` | Per-section |
| Digest | `digestLoadGate`, `digestCatalogGate`, `digestGraphOpenGate` | Graph-open gate invalidated on corpus change |
| Search | `searchRunGate` | Bumped after query validation |
| Explore | `exploreRunGate` | Shared for filtered + NL |
| Index stats | `indexStatsRefreshGate`, `indexRebuildGate` | Rebuild polls with rebuild gate |
| Transcript dialog | `transcriptOpenGate` | Main + segments sidecar |
| Topic timeline | `timelineLoadGate` | CIL topic timeline |

### Relationship: gates vs dedupe

- **Gates** own **correctness** — they decide whether a response is still
  relevant to the current UI state.
- **Dedupe** owns **cost** — it prevents duplicate HTTP requests when multiple
  callers ask for the same URL at the same instant.

Both layers are independent. A deduped response still goes through the caller's
gate check before reaching the UI.

## Data flow examples

### Library tab: corpus path change

```text
User enters corpus path in shell
  -> shell.corpusPath updates
  -> LibraryView watcher fires
     -> libraryFeedsGate.bump()
     -> fetchCorpusFeeds(path)  [deduped if concurrent]
     -> gate.isStale(seq)? -> abandon
     -> feeds rendered
     -> user picks feed
     -> libraryEpisodesGate.bump()
     -> fetchCorpusEpisodes(path, feed)
     -> gate check -> episodes rendered
```

### Search to graph focus

```text
User types query, clicks Search
  -> searchRunGate.bump()
  -> searchCorpus(query, filters)
  -> gate check -> results rendered as ResultCards
  -> user clicks "Open in graph" on a ResultCard
  -> graphNavigation.requestFocusNode(cyId)
  -> App.vue switches mainTab to 'graph'
  -> GraphCanvas detects pendingFocusNodeId
  -> Cytoscape centers + highlights node
```

### Digest to graph

```text
DigestView loads
  -> digestLoadGate.bump()
  -> fetchCorpusDigest(path)
  -> gate check -> topics/episodes rendered
  -> user clicks topic hit "Open in graph"
  -> digestGraphOpenGate.bump()
  -> loadRelativeArtifacts(paths)
  -> gate check
  -> graphNavigation.requestFocusNode(topicNodeId)
  -> App switches to graph tab
```

## Types

| File | Domain |
| ---- | ------ |
| `types/artifact.ts` | Raw + parsed GI/KG JSON, `ParsedArtifact`, `GraphFilterState`, node/edge shapes |
| `types/bridge.ts` | RFC-072 `bridge.json`: `BridgeDocument`, `BridgeIdentity` |

API modules export their own request/response types co-located with the
functions that use them.

## Utility modules (selected)

| Module | Purpose |
| ------ | ------- |
| `parsing.ts` | Core artifact parsing, Cytoscape element generation, ego subgraph |
| `mergeGiKg.ts` | Merge GI + KG artifacts into a single display graph |
| `cyGraphStylesheet.ts` | Cytoscape stylesheet builder, label placement |
| `graphEpisodeMetadata.ts` | Map graph nodes to episode/metadata paths; corpus catalog resolution |
| `staleGeneration.ts` | `StaleGeneration` class for async cancel |
| `searchFocus.ts` | Search hit to Cytoscape node ID resolution |
| `corpusSearchHandoff.ts` | Build search query from library episode metadata |
| `transcriptViewerModel.ts` | Transcript fetch, size cap, highlight splitting |
| `localCalendarDate.ts` | Date helpers for corpus lens presets |
| `listRowArrowNav.ts` | Keyboard arrow navigation for vertical lists |

## Testing

| Layer | Tool | Location | Command |
| ----- | ---- | -------- | ------- |
| Unit (TS) | Vitest | `src/**/*.test.ts` | `make test-ui` |
| E2E (browser) | Playwright (Firefox) | `e2e/*.spec.ts` | `make test-ui-e2e` |
| API (Python) | pytest | `tests/unit/podcast_scraper/server/`, `tests/integration/server/` | `make test-fast` |

The E2E surface contract is documented in
`web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` (outside the docs tree).

## Related documents

| Document | What it covers |
| -------- | -------------- |
| [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md) | Design rationale, stack choices, folder layout |
| [Server Guide](../guides/SERVER_GUIDE.md) | FastAPI routes, API contract |
| [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) | Shared design system, tokens, typography |
| [UXS index](../uxs/index.md) | Per-feature UXS docs (Digest, Library, Graph, Search, Dashboard) |
| E2E Surface Map (`web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md`) | Playwright selectors, surface ownership |
| [WIP: HTTP stability](../wip/wip-viewer-holistic-http-stability.md) | Full gate inventory, dedupe table, health behavior |
| [WIP: Corpus load stability](../wip/wip-viewer-corpus-load-graph-stability.md) | Large-corpus load and graph hardening |
| [Architecture](ARCHITECTURE.md) | System-level architecture (viewer is one surface) |
| [Development Guide](../guides/DEVELOPMENT_GUIDE.md) | Dev workflow, `make serve`, debugging |

---

**Version:** 1.0
**Created:** 2026-04-15
