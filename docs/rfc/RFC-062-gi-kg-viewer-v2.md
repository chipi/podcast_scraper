# RFC-062: GI/KG Viewer v2 — Semantic Search UI & Web Application Architecture

- **Status**: Completed
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, GIL/KG consumers, viewer users
- **Related PRDs**:
  - `docs/prd/PRD-021-semantic-corpus-search.md` (semantic search — surfaced in viewer)
  - `docs/prd/PRD-017-grounded-insight-layer.md` (GIL — primary visualization content)
  - `docs/prd/PRD-019-knowledge-graph-layer.md` (KG — primary visualization content)
- **Related RFCs**:
  - `docs/rfc/RFC-061-semantic-corpus-search.md` (search backend this viewer consumes)
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` (GIL artifact format)
  - `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md` (UC4/UC5 — semantic QA, Insight Explorer)
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md` (KG artifact format)
  - `docs/rfc/RFC-056-knowledge-graph-layer-use-cases.md` (KG use cases)
  - `docs/rfc/RFC-051-database-projection-gil-kg.md` (future structured query backend)
- **Related UX specs**:
  - `docs/uxs/UXS-001-gi-kg-viewer.md` (visual and token contract for viewer v1/v2)
- **Related Documents**:
  - [GitHub #489](https://github.com/chipi/podcast_scraper/issues/489) — Implementation issue for this RFC
  - [GitHub #445](https://github.com/chipi/podcast_scraper/issues/445) — Viewer v1 implementation
  - [GitHub #484](https://github.com/chipi/podcast_scraper/issues/484) — Semantic Corpus Search Phase 1
  - [GitHub #466](https://github.com/chipi/podcast_scraper/issues/466) — GI + KG depth roadmap
  - [GitHub #50](https://github.com/chipi/podcast_scraper/issues/50) — Simple UI + server (v2.7)
  - [GitHub #347](https://github.com/chipi/podcast_scraper/issues/347) — UI to access output from DB (v2.7)
  - [GitHub #46](https://github.com/chipi/podcast_scraper/issues/46) — Docker architecture (v2.7)
  - `docs/architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md` — Platform architecture vision

## Abstract

This RFC proposes a full rebuild of the GI/KG viewer (`web/gi-kg-viz/`) as a proper
Vue 3 web application backed by the project's first FastAPI server layer. The viewer v2
replaces the current collection of vanilla-JS pages with a single-page application
featuring semantic search integration (RFC-061), a Cytoscape.js graph engine, typed
state management via Pinia, and a REST API layer wrapping the `VectorStore`,
`gi explore`, and artifact loading capabilities of the core Python library.

The FastAPI backend is intentionally placed in `src/podcast_scraper/server/` — not
`viewer/` — because it is the **seed of the platform API**. The viewer is the first
consumer of this server; the platform UI ([#50](https://github.com/chipi/podcast_scraper/issues/50),
[#347](https://github.com/chipi/podcast_scraper/issues/347), megasketch Part A) is the
second. One server, pluggable route groups, feature flags to activate platform routes
when ready.

**Architecture Alignment:** The server is a **consumption and coordination layer** — it
does not modify artifacts, pipeline stages, or CLI commands. It wraps existing Python
APIs (`VectorStore.search()`, `gi explore`, artifact loading) behind REST endpoints.
The frontend is a static SPA served by the same process, with a file-picker fallback
for offline use. This follows the megasketch constraint A.2: **"One pipeline core,
multiple shells"** — the CLI is one shell, the server is another.

## Problem Statement

The viewer v1 (`web/gi-kg-viz/`, [#445](https://github.com/chipi/podcast_scraper/issues/445))
was built as an exploratory prototype to visualize GI and KG artifacts:

- **No search capability** — no text search, no semantic search, no query interface.
  The only "filtering" is by node type or file name substring. With the semantic search
  layer landing in #484 (RFC-061), the viewer has no way to surface the most powerful
  new capability in the stack.
- **Duplicate graph engines** — both vis-network and Cytoscape.js render the same data
  in parallel pages (`graph-vis.html`, `graph-cyto.html`). This was useful for
  comparison during prototyping but doubles maintenance for no ongoing value.
- **No shared state** — each HTML page manages its own global state via `window.GiKgViz`
  IIFEs. There is no reactive data layer, no store, no event bus. Adding features
  requires duplicating logic across pages.
- **No framework** — vanilla JS with CDN script tags. No bundler, no type checking, no
  component model. The codebase is fragile at ~153 KB of hand-managed JavaScript.
- **No backend integration** — the viewer loads files via browser File API or an
  optional dev server (`gi_kg_viz_server.py`). It cannot call Python APIs (search,
  explore, index stats) without a proper backend layer.
- **Multiple entry points** — `index.html`, `graph-vis.html`, `graph-cyto.html`,
  `json-only.html` — fragmenting the experience instead of offering a unified
  application.

Semantic Corpus Search (RFC-061) adds `podcast search`, `podcast index --stats`, and
upgraded `gi explore --topic` / `gi query` with semantic matching. These capabilities
need a visual consumption layer that preserves GIL provenance (grounding, quotes,
timestamps) and enables cross-feed discovery — the same use cases RFC-061 targets but
through a graphical interface.

**Use Cases:**

1. **Visual Semantic Search**: Type a natural-language query, see ranked results with
   type badges, scores, episode attribution, and grounding status — then click a result
   to focus the corresponding node in the graph
2. **Corpus Dashboard**: At-a-glance view of index health — vector counts, type
   breakdown, feed coverage, freshness, model info — equivalent to `podcast index --stats`
   but always visible
3. **Semantic Explore/QA**: Ask "what do my podcasts say about X?" and get the UC4
   answer envelope rendered as a card with supporting evidence, not just terminal JSON
4. **Graph + Search Integration**: Search results and graph exploration in the same view
   — click a search result to see its neighborhood, double-click a node to search for
   similar content
5. **Offline Fallback**: Load `gi.json`/`kg.json` files from disk when no backend is
   running — degraded mode (no search, no explore) but graph and metrics still work

## Goals

1. **Single-page Vue 3 application** replacing the multi-page vanilla-JS prototype
2. **Cytoscape.js as sole graph engine** with proper Vue component wrapping
3. **Semantic search panel** consuming the RFC-061 `VectorStore.search()` API via
   a FastAPI endpoint
4. **Index health dashboard** surfacing `IndexStats` visually
5. **Explore/QA integration** exposing `gi explore --topic` and `gi query` through the
   viewer
6. **FastAPI server** in `src/podcast_scraper/server/` — the project's canonical server
   layer, with viewer routes as the first route group and platform routes (#50, #347)
   added later via feature flags
7. **Clean frontend architecture** with Pinia stores, typed API client, and composables
8. **File-picker fallback** for offline/no-backend usage (graph and metrics only)
9. **Playwright E2E test layer** for browser-based regression testing of viewer
   functionality
10. **Platform-ready design** — server, frontend, and API contracts designed to grow
    into the full platform vision (megasketch Part A/B) without architectural rewrites

## Constraints & Assumptions

**Constraints:**

- Must consume existing artifacts and APIs — no changes to `gi.json`, `kg.json`, or
  pipeline output formats
- Backend must wrap existing Python functions, not reimplement search or explore logic
- Must work without a backend (degraded mode) for quick file inspection
- Must run with `podcast serve` CLI command and `make serve` Makefile target — single
  command to start both frontend and backend
- No authentication required (single-user local tool; auth deferred to platform mode)
- Server module naming must be `server/` not `viewer/` — it is the project's server
  layer, not just the viewer's backend

**Assumptions:**

- RFC-061 (Semantic Corpus Search) will be implemented before or in parallel with the
  search panel (M4). The viewer backend can be built first; the search endpoint activates
  when a vector index exists.
- The v1 viewer (`web/gi-kg-viz/`) has been removed (Phase 4 complete).
- The target user is a developer or power user running the tool locally.

## Design & Implementation

### 1. Tech Stack

**Frontend:**

| Layer | Technology | Rationale |
| ----- | ---------- | --------- |
| Framework | Vue 3 (Composition API, `<script setup>`) | Lightweight reactivity, SFCs, no JSX overhead. Right weight for a data visualization tool. |
| Build | Vite | Fast HMR, native ESM, zero-config Vue support. Static output for production. |
| Graph | Cytoscape.js | Programmatic API for filtering/styling/selection. Compound nodes for clustering. Extensions for layouts (cola, dagre) and interaction (popper, cxtmenu). |
| Charts | Chart.js 4 (via vue-chartjs) | Familiar from v1, lightweight, covers distribution and stats charts. |
| CSS | Tailwind CSS | Utility-first, built-in dark mode, replaces 600+ line custom stylesheet. |
| State | Pinia | Vue's official store. Replaces `window.GiKgViz` globals with reactive typed stores. |
| HTTP | Native fetch (or ofetch) | No need for axios at this scale. Typed wrappers in `api/` layer. |
| Types | TypeScript | Type safety for artifact shapes, search results, API contracts. |

**Backend:**

| Layer | Technology | Rationale |
| ----- | ---------- | --------- |
| Server | FastAPI | Async, typed, auto-docs. Extends `gi_kg_viz_server.py` pattern. Aligns with platform megasketch (Part A). |
| API style | REST + JSON | Matches CLI `--format json` contracts. Same shapes frontend consumes. |
| Search | Thin wrapper over `VectorStore.search()` | No new search logic — exposes what RFC-061 builds. |
| Artifacts | File-system reads | `gi.json`, `kg.json` loaded from `output_dir`. |
| Static files | FastAPI `StaticFiles` mount | Production: FastAPI serves Vite build output. Dev: Vite proxy. |

**Why Vue 3 over React:**

- **Reactivity without ceremony** — filter toggles, search state, graph selection all
  naturally reactive without `useCallback`/`useMemo` patterns
- **Single-file components** — template + script + scoped style in one file, cleaner for
  a small codebase
- **Lighter bundle** — Vue 3 core ~16 KB gzipped vs React + ReactDOM ~42 KB; matters when
  main payload is Cytoscape + Chart.js
- **Composition API** — same composable/hooks pattern as React but without rules-of-hooks
  constraints

**Why Cytoscape.js (dropping vis-network):**

- **Programmatic API** — richer API for computed styles, filtering, batch operations.
  Critical for search-result-to-node highlighting.
- **Compound nodes** — native parent/child grouping for feed clusters, semantic clusters
- **Extension ecosystem** — cola/dagre layouts, popper tooltips, cxtmenu context menus
- **WebGL escape hatch** — `cytoscape-canvas` renderer for large corpora
- **Active maintenance** — stronger community; better TypeScript support

### 2. Frontend Architecture

```text
web/gi-kg-viewer/
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
├── index.html
├── src/
│   ├── main.ts                       # Vue app bootstrap
│   ├── App.vue                       # Root component + router-view
│   ├── router/
│   │   └── index.ts                  # vue-router: explore, dashboard, detail
│   ├── stores/
│   │   ├── artifacts.ts              # loaded GI/KG artifacts, merge logic
│   │   ├── search.ts                 # query, results, filters, loading state
│   │   ├── indexHealth.ts            # IndexStats, feed coverage, freshness
│   │   └── graph.ts                  # focus node, layout config, visual state
│   ├── api/
│   │   ├── client.ts                 # base fetch wrapper, error handling
│   │   ├── artifacts.ts              # GET /api/artifacts, /api/artifacts/:name
│   │   ├── search.ts                 # GET /api/search
│   │   └── index.ts                  # GET /api/index/stats
│   ├── composables/
│   │   ├── useSearch.ts              # search + debounce + result mapping
│   │   ├── useGraph.ts              # cytoscape instance lifecycle
│   │   ├── useFilters.ts             # filter state + apply to graph/search
│   │   └── useArtifacts.ts           # artifact loading + file-picker fallback
│   ├── components/
│   │   ├── graph/
│   │   │   ├── GraphCanvas.vue       # cytoscape wrapper
│   │   │   ├── GraphLegend.vue       # color legend with click-to-solo
│   │   │   ├── GraphControls.vue     # fit, re-layout, export PNG/SVG
│   │   │   └── NodeDetail.vue        # click-to-inspect detail panel
│   │   ├── search/
│   │   │   ├── SearchBar.vue         # query input + submit
│   │   │   ├── SearchResults.vue     # scrollable result list
│   │   │   ├── SearchFilters.vue     # type, feed, date, speaker, grounded
│   │   │   └── ResultCard.vue        # typed result card (insight/quote/chunk)
│   │   ├── dashboard/
│   │   │   ├── IndexStats.vue        # vector counts, model info
│   │   │   ├── CorpusTimeline.vue    # episodes on time axis by feed
│   │   │   └── FeedCoverage.vue      # feed × indexed status matrix
│   │   ├── common/
│   │   │   ├── MetricsPanel.vue      # reusable key-value metrics display
│   │   │   ├── FilterChips.vue       # active filter badges
│   │   │   └── LoadingState.vue      # spinner / skeleton
│   │   └── layout/
│   │       ├── AppShell.vue          # sidebar + main + optional right panel
│   │       ├── Sidebar.vue           # navigation + artifact list
│   │       └── TopBar.vue            # search bar + mode toggle
│   ├── views/
│   │   ├── ExploreView.vue           # graph + search + filters (main view)
│   │   ├── DashboardView.vue         # index health + corpus overview
│   │   └── DetailView.vue            # single episode/insight deep dive
│   ├── types/
│   │   ├── artifact.ts              # GI/KG TypeScript types
│   │   ├── search.ts                # SearchResult, IndexStats, SearchFilters
│   │   └── graph.ts                 # CyNode, CyEdge, VisualConfig
│   ├── theme/
│   │   ├── tokens.css               # CSS custom-property definitions (UXS-001 tokens)
│   │   ├── theme.ts                 # runtime helpers: resolve token, apply preset
│   │   └── presets/                  # optional alternate value sets for experimentation
│   │       ├── default.css           # production token values (= UXS-001 table)
│   │       └── compact.css           # denser spacing / smaller font experiment
│   └── utils/
│       ├── colors.ts                # node type → color mapping (from shared.js)
│       ├── parsing.ts               # artifact parsing (ported from shared.js)
│       └── formatting.ts            # dates, scores, text truncation
└── public/
    └── favicon.svg
```

**Key architectural decisions:**

- **Stores** own the data and derived state. Components read from stores reactively.
  API calls are made via the `api/` layer and results are written to stores.
- **Composables** encapsulate reusable logic (graph lifecycle, debounced search,
  filter application) without coupling to specific components.
- **Types** mirror the Python Pydantic models (`SearchResult`, `IndexStats`,
  `ParsedArtifact`) and provide compile-time safety.
- **API client** is a thin typed wrapper over fetch. Each endpoint module exports
  functions like `searchCorpus(query, filters)` → `Promise<SearchResponse>`.

### 3. Server Architecture (Platform Seed)

The server lives in `src/podcast_scraper/server/` — not `viewer/`. This is the
project's **canonical server layer**. The viewer is the first consumer; the platform
UI (#50, #347, megasketch) is the second. Route groups are pluggable via feature flags.

```text
src/podcast_scraper/server/
├── __init__.py
├── app.py                            # FastAPI app factory (feature-flagged routers)
├── dependencies.py                   # shared: config, output_dir, vector_store
├── schemas.py                        # Pydantic response models (shared across routers)
├── middleware/
│   ├── __init__.py
│   └── cors.py                       # CORS config; future: auth, request logging
├── routes/
│   ├── __init__.py
│   ├── health.py                     # GET /api/health (always mounted)
│   │
│   │   # --- Viewer routes (v2.6, this RFC) ---
│   ├── artifacts.py                  # GET /api/artifacts, /api/artifacts/:name
│   ├── search.py                     # GET /api/search (semantic search)
│   ├── explore.py                    # GET /api/explore (gi explore / gi query)
│   ├── index_stats.py                # GET /api/index/stats
│   │
│   │   # --- Platform routes (v2.7, added later via #50, #347) ---
│   ├── feeds.py                      # CRUD /api/feeds (catalog, placeholder)
│   ├── episodes.py                   # GET /api/episodes (browsing, placeholder)
│   ├── jobs.py                       # POST/GET /api/jobs (pipeline runs, placeholder)
│   └── status.py                     # GET /api/status (pipeline status, placeholder)
```

**FastAPI app factory (`app.py`):**

```python
def create_app(
    output_dir: Path,
    *,
    enable_viewer: bool = True,
    enable_platform: bool = False,
    vector_index_path: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="podcast_scraper")

    # Always available
    app.include_router(health_router)

    # Viewer routes (v2.6 — this RFC)
    if enable_viewer:
        app.include_router(artifacts_router, prefix="/api")
        app.include_router(search_router, prefix="/api")
        app.include_router(explore_router, prefix="/api")
        app.include_router(index_stats_router, prefix="/api")
        app.mount("/", StaticFiles(...), name="viewer")

    # Platform routes (v2.7, future — #50, #347, megasketch)
    if enable_platform:
        app.include_router(feeds_router, prefix="/api")
        app.include_router(episodes_router, prefix="/api")
        app.include_router(jobs_router, prefix="/api")

    return app
```

- Accepts `output_dir` (path to corpus outputs) and optional `vector_index_path`
- Lazily loads `FaissVectorStore` on first search request
- Mounts Vite build output as static files at `/` (viewer mode)
- CORS enabled for dev mode (Vite dev server on different port)
- Platform routes are placeholder files until v2.7 work begins — they exist in the
  tree to make the growth path visible but are not mounted by default

**CLI entry point (`podcast serve`):**

```bash
# v2.6 — viewer mode (default)
podcast serve --output-dir ./output

# v2.7 — full platform mode (future)
podcast serve --output-dir ./output --platform

# Development shorthand
make serve        # Start backend + frontend dev servers
```

The `podcast serve` command is a new CLI subcommand that starts the FastAPI server.
It replaces the ad-hoc `scripts/gi_kg_viz_server.py` script with a proper entry point.

**Route contracts (viewer routes — implemented in this RFC):**

```python
# GET /api/health
@router.get("/api/health")
async def health() -> dict:
    """Server health check. Always mounted."""
    ...

# GET /api/artifacts?path=<output_dir>
@router.get("/api/artifacts")
async def list_artifacts(path: str) -> ArtifactListResponse:
    """List all gi.json and kg.json files in the output directory."""
    ...

# GET /api/artifacts/{name}
@router.get("/api/artifacts/{name}")
async def get_artifact(name: str) -> dict:
    """Load and return a parsed artifact by filename."""
    ...

# GET /api/search?q=<query>&type=insight,quote&top_k=20&feed=...
@router.get("/api/search")
async def search_corpus(
    q: str,
    type: list[str] | None = None,
    feed: str | None = None,
    since: str | None = None,
    speaker: str | None = None,
    grounded_only: bool = False,
    top_k: int = 10,
) -> SearchResponse:
    """Semantic search over the vector index."""
    ...

# GET /api/index/stats
@router.get("/api/index/stats")
async def index_stats() -> IndexStatsResponse:
    """Return vector index statistics."""
    ...

# GET /api/explore?topic=<topic>&speaker=<speaker>&limit=20
@router.get("/api/explore")
async def explore_insights(
    topic: str | None = None,
    speaker: str | None = None,
    limit: int = 20,
    grounded_only: bool = False,
) -> ExploreResponse:
    """Run gi explore and return ExploreOutput."""
    ...
```

**Platform route contracts (v2.7, placeholder stubs only):**

```python
# Future: CRUD /api/feeds — catalog management (#50)
# Future: GET /api/episodes — episode browsing (#347)
# Future: POST/GET /api/jobs — pipeline job management (#50)
# Future: GET /api/status — pipeline status monitoring (#50)
```

**Integration with existing code:**

The server imports and calls existing functions — it does not reimplement logic:

- `VectorStore.search()` for `/api/search`
- `run_uc5_insight_explorer()` for `/api/explore`
- `IndexStats` from the vector store for `/api/index/stats`
- Direct file reads for `/api/artifacts`

**Relationship to existing `service.py`:**

The existing `podcast_scraper/service.py` provides `service.run()` — a one-shot
pipeline execution entry point. The new `server/` module is the **long-lived server**
that wraps pipeline capabilities behind HTTP. When platform mode (#50) lands, the
server will use `service.run()` internally for job execution. They are complementary:
`service.py` = "run once," `server/` = "serve continuously."

### 4. Graph ↔ Search Integration

The key UX innovation of v2 is **bidirectional linking between search results and
graph nodes:**

**Search → Graph:**

1. User types a query in `SearchBar`
2. `useSearch` composable calls `/api/search`, writes results to `search` store
3. `SearchResults` renders ranked `ResultCard` components
4. Clicking a `ResultCard` dispatches `graph.focusNode(docId)` to the graph store
5. `GraphCanvas` receives the focus event, highlights the target node (glow + zoom),
   and dims non-neighborhood nodes

**Graph → Search:**

1. User double-clicks a node (e.g., an Insight node) in `GraphCanvas`
2. `useGraph` composable reads the node's text content
3. Dispatches `search.searchSimilar(nodeText)` — pre-fills the search bar and executes
4. Search results show content semantically similar to the clicked node

**Focus + Context:**

- `NodeDetail` panel shows full node properties: text, grounding status, supporting
  quotes, episode attribution, confidence score
- For Insight nodes: expandable supporting quotes with transcript references
- For Transcript Chunk results: timestamp + "jump to transcript" link

### 5. Offline / No-Backend Fallback

The viewer must degrade gracefully when no backend is running:

| Feature | With backend | Without backend |
| ------- | ------------ | --------------- |
| Load artifacts | `/api/artifacts` | File picker / `showDirectoryPicker()` |
| Graph rendering | Full | Full |
| Metrics panel | Full | Full (computed from loaded artifacts) |
| Semantic search | Full | Disabled (message: "Run `podcast serve` to enable search") |
| Explore/QA | Full | Disabled (same message) |
| Index stats | Full | Disabled |

**Detection:** On mount, `api/client.ts` pings `GET /api/health`. If it fails, the
`artifacts` store switches to file-picker mode and search/explore components show a
disabled state with an instructional message.

### 6. Dev Server & Build

**Development:**

```bash
# Terminal 1: FastAPI backend
make serve-api
# Runs: uvicorn podcast_scraper.server.app:create_app --reload --port 8000

# Terminal 2: Vite dev server
make serve-ui
# Runs: cd web/gi-kg-viewer && npm run dev
# Vite proxies /api/* to localhost:8000

# Or combined:
make serve
# Runs both via a process manager or background + fg
```

**CLI entry point (production):**

```bash
# Viewer mode (serves built frontend + API)
podcast serve --output-dir ./output --port 8000

# With platform routes enabled (v2.7, future)
podcast serve --output-dir ./output --platform
```

**Production build:**

```bash
cd web/gi-kg-viewer && npm run build
# Production build to dist/
# Output: web/gi-kg-viewer/dist/
# FastAPI serves dist/ as static files
```

**Makefile targets:**

| Target | Action |
| ------ | ------ |
| `make serve` | Start backend + frontend dev servers |
| `make serve-api` | Start FastAPI backend only |
| `make serve-ui` | Start Vite dev server only |
| `cd web/gi-kg-viewer && npm run build` | Production build of frontend |
| `make test-ui` | Vitest unit tests for TS utils |
| `make test-ui-e2e` | Playwright browser E2E (Firefox) |
| `make test-ui-e2e` | Playwright E2E tests for viewer |

## Key Decisions

1. **Full rebuild vs incremental port**
   - **Decision**: Full rebuild in `web/gi-kg-viewer/` (new directory)
   - **Rationale**: The v1 architecture (vanilla JS, globals, multi-page, dual engines)
     is fundamentally incompatible with the reactive component model, typed stores,
     and search integration needed for v2. Porting would require rewriting every file
     while fighting the existing structure. A clean start with proper tooling (Vite,
     TypeScript, Vue) is faster and produces a better result. v1 has been removed.

2. **Cytoscape.js only (drop vis-network)**
   - **Decision**: Consolidate to Cytoscape.js as the sole graph engine
   - **Rationale**: The vis-network comparison served its purpose in #445 exploration.
     For a production viewer, maintaining two engines doubles code (graph-vis.js +
     graph-cyto.js) for no user value. Cytoscape has the stronger programmatic API for
     search-driven highlighting, compound nodes for clustering, and extension ecosystem
     for advanced layouts.

3. **Vue 3 over React**
   - **Decision**: Vue 3 with Composition API
   - **Rationale**: Lighter bundle (~16 KB vs ~42 KB gzipped), natural reactivity for
     filter/search state without `useCallback`/`useMemo` overhead, single-file
     components for a small codebase. The Composition API provides the same
     composable/hooks pattern. Either would work; Vue is the better fit for this scale.
     This decision also holds for future platform UI views (#50, #347) — Vue scales
     cleanly from data visualization to CRUD forms with the same component model.

4. **`src/podcast_scraper/server/` — platform-first naming**
   - **Decision**: Backend lives in `server/`, not `viewer/`, inside the Python package
   - **Rationale**: This is the project's **canonical server layer**, not just the
     viewer's backend. The viewer is the first consumer; platform routes (#50, #347,
     megasketch) are the second. Naming it `server/` avoids a rename when platform work
     starts. Feature flags (`enable_viewer`, `enable_platform`) control which route
     groups are mounted. This follows megasketch constraint A.2: "One pipeline core,
     multiple shells." The `podcast serve` CLI command starts this server. The existing
     `scripts/gi_kg_viz_server.py` has been replaced by this proper module.

5. **Tailwind CSS (no component library)**
   - **Decision**: Tailwind utility classes + custom components
   - **Rationale**: The v1 `styles.css` (600+ lines) would need rewriting regardless.
     A component library (PrimeVue, Vuetify) adds weight and opinion that fights a
     data-visualization UI. Tailwind gives dark mode, responsive, and consistent
     spacing out of the box with full design control.

6. **Token-based theming with preset support**
   - **Decision**: All visual tokens (colors, typography, spacing, radii) are defined as
     CSS custom properties in `src/theme/tokens.css` and consumed by Tailwind via
     `tailwind.config.js` `extend.colors` / `extend.fontFamily`. A `theme.ts` helper
     exposes runtime getters for use in Chart.js and Cytoscape (which cannot read CSS
     vars directly). Optional CSS preset files (`src/theme/presets/`) can override
     tunable token values (e.g. fonts, radii, spacing) for rapid experimentation.
   - **Rationale**: UXS-001 distinguishes **frozen** token names and conventions from
     **open** values (see "Tunable parameters" section). Separating token definitions
     from component code lets a developer swap font families, adjust spacing scales,
     or compare compact vs relaxed density by loading a single preset file -- without
     touching any Vue component. This is lightweight (no runtime theming library; just
     CSS cascade) and aligns with the Tailwind workflow.

7. **Pinia state management**
   - **Decision**: Pinia stores for artifacts, search, graph, and index state
   - **Rationale**: Replaces the `window.GiKgViz` / `window.GiKgVizShell` globals
     with typed reactive stores. Components subscribe to store state and dispatch
     actions. Cross-component coordination (search result → graph focus) happens
     through store watchers, not DOM events or callback chains. Scales cleanly when
     platform stores (`feeds.ts`, `jobs.ts`, `episodes.ts`) are added in v2.7.

8. **Playwright for UI E2E testing**
   - **Decision**: Playwright as the browser E2E test framework
   - **Rationale**: Official Vue recommendation for E2E. Headless by default (CI
     friendly). Multi-browser (Chromium, Firefox, WebKit). Lighter than Cypress (no
     Electron). Built-in web assertions. Test structure mirrors the component
     structure for maintainability.

9. **`podcast serve` CLI command**
   - **Decision**: New top-level CLI subcommand to start the server
   - **Rationale**: Replaces the ad-hoc `scripts/gi_kg_viz_server.py` with a proper
     entry point. Supports `--output-dir`, `--port`, and future `--platform` flag.
     Aligns with the "CLI stays first-class" constraint (megasketch A.2.1) — the
     server is started via CLI, not a separate tool.

## Alternatives Considered

1. **Incremental refactor of v1**
   - **Description**: Add Vue components to existing HTML pages, gradually replace
     vanilla JS
   - **Pros**: No big-bang rewrite; preserves working code
   - **Cons**: Mixing vanilla JS globals with Vue reactivity creates bugs; no
     TypeScript; can't introduce Vite without restructuring; "half-ported" state is
     worse than either extreme
   - **Why Rejected**: The v1 architecture fundamentally conflicts with the target.
     A clean rebuild is faster for this codebase size (~153 KB).

2. **React + Next.js**
   - **Description**: Full React SPA (or Next.js for SSR)
   - **Pros**: Larger ecosystem; more developer familiarity broadly
   - **Cons**: Heavier bundle; SSR unnecessary for a local tool; more boilerplate for
     state management
   - **Why Rejected**: Vue is a better fit at this scale. No SSR or SEO requirements.
     Either would work; Vue is lighter and more ergonomic for reactive data visualization.

3. **Svelte + SvelteKit**
   - **Description**: Svelte for minimal bundle, SvelteKit for routing
   - **Pros**: Smallest bundle; no virtual DOM; compiler-based reactivity
   - **Cons**: Smaller ecosystem for Cytoscape/Chart.js integration; fewer community
     examples for the specific integration needed
   - **Why Rejected**: Vue has better Cytoscape integration examples and a more mature
     ecosystem for this specific use case. Svelte is a valid alternative.

4. **Keep vis-network alongside Cytoscape**
   - **Description**: Maintain both graph engines, let user choose
   - **Pros**: Comparison; user preference
   - **Cons**: Doubles graph component code; doubles layout logic; no user has
     requested the comparison post-#445
   - **Why Rejected**: The comparison served its #445 purpose. Consolidation halves
     the graph code and focuses effort on search integration.

5. **No backend (WASM embeddings in browser)**
   - **Description**: Run `all-MiniLM-L6-v2` via ONNX Runtime Web; load FAISS index
     client-side
   - **Pros**: Fully offline; no Python server needed
   - **Cons**: ~80 MB model download in browser; slow inference on CPU; complex build;
     can't reuse Python `VectorStore` code; limits future API growth
   - **Why Rejected**: Heavy engineering for limited gain. The FastAPI backend is
     simpler, faster, and aligns with the platform vision.

## Testing Strategy

### Test Layers

| Layer | Tool | Scope | Speed |
| ----- | ---- | ----- | ----- |
| Frontend unit | Vitest | Store logic, composables, utilities, component rendering | Fast (no browser) |
| Frontend component | Vitest + Vue Test Utils | Component behavior with mocked stores/API | Fast (no browser) |
| API integration | pytest + httpx | Backend routes return correct shapes, error handling | Fast (no browser) |
| UI E2E | Playwright | Full browser: load → search → click → verify graph | Slow (~30s per test) |
| Visual regression | Playwright screenshots | Graph renders consistently, dark mode, layout | Optional, in CI |

### Frontend Unit + Component Tests (Vitest)

- **Store logic**: artifact parsing, search result mapping, filter application, merge
  logic. Mock API calls via `vi.mock`.
- **Composables**: debounced search, graph instance lifecycle, filter state management.
- **Utility functions**: color mapping, date formatting, text truncation, parsing.
- **Component rendering**: key components render correctly with given props/store state.
  Search bar emits correct events. Result cards render all fields for each doc type.
  Graph canvas initializes Cytoscape instance.

### Backend Unit + Integration Tests (pytest)

- **Unit tests**: Route handlers return correct shapes. Error cases (missing index,
  invalid query, nonexistent artifact) return proper HTTP status codes. Artifact
  listing matches file system state.
- **Integration tests**: Full round-trip with httpx `TestClient`: load test artifacts →
  call `/api/artifacts` → verify response shape. Search with mock `VectorStore` →
  verify result mapping and filtering. Health endpoint returns expected payload.

### UI E2E Tests (Playwright)

Playwright provides the browser automation layer for end-to-end regression testing.

**Test structure:**

```text
web/gi-kg-viewer/
├── e2e/
│   ├── playwright.config.ts          # Playwright config (headless Chromium)
│   ├── fixtures/
│   │   ├── sample-gi.json            # small test GI artifact (~5 insights)
│   │   ├── sample-kg.json            # small test KG artifact (~10 entities)
│   │   └── sample-index/             # pre-built small FAISS index for search
│   │       ├── vectors.faiss
│   │       ├── metadata.json
│   │       └── index_meta.json
│   ├── helpers/
│   │   ├── server.ts                 # programmatic FastAPI start/stop for tests
│   │   └── test-data.ts              # artifact loading + expectation helpers
│   └── tests/
│       ├── graph-load.spec.ts        # load artifacts → graph renders nodes/edges
│       ├── graph-filters.spec.ts     # toggle type filters → node count changes
│       ├── graph-focus.spec.ts       # double-click node → 1-hop ego focus
│       ├── search.spec.ts            # type query → results appear → score ordering
│       ├── search-to-graph.spec.ts   # click result card → graph highlights node
│       ├── dashboard.spec.ts         # navigate to dashboard → index stats render
│       ├── offline-mode.spec.ts      # no backend → file picker fallback works
│       └── dark-mode.spec.ts         # toggle theme → verify no visual breaks
```

**Key E2E scenarios for regression testing:**

1. **Graph renders** — load sample GI+KG → verify node count matches artifact →
   verify edges visible → verify legend populates
2. **Search works** — type query → verify result cards appear → verify score ordering
   → verify doc type badges render
3. **Search → Graph focus** — click result card → verify node highlighted with glow →
   verify non-neighborhood dimmed → verify detail panel opens
4. **Filters apply** — toggle "hide ungrounded" → verify node count decreases → reset
   → verify original count restored
5. **Dashboard loads** — navigate to dashboard → verify index stats display → verify
   Chart.js distribution renders
6. **Offline fallback** — no backend → verify file picker appears → load files →
   verify graph renders → verify search shows "start server" message
7. **Dark mode** — toggle theme → take screenshot → compare with baseline

**Playwright configuration:**

See `web/gi-kg-viewer/playwright.config.ts` for the current configuration
(Firefox, Vite on port 5174, `testDir: ./e2e`).

### Test Organization

- `web/gi-kg-viewer/e2e/` — browser E2E tests (Playwright)
- `tests/unit/podcast_scraper/server/` — backend unit tests (pytest)
- `tests/integration/test_server_api.py` — backend API integration (pytest)

### Test Execution

| Command | What Runs | When |
| ------- | --------- | ---- |
| `make test-ui-e2e` | Playwright (browser E2E) | Pre-merge, CI |
| `make ci-fast` | pytest (includes server unit + integration) | Pre-commit |
| `make ci` | Full suite (pytest + Playwright) | CI pipeline |

## Rollout & Monitoring

**Rollout Plan:**

- **M1 — Scaffold + Server Shell** (done): Vite + Vue + Tailwind + Pinia project
  in `web/gi-kg-viewer/`. Token-based theming layer in `src/theme/` — `tokens.css`
  (UXS-001 semantic tokens as CSS custom properties), `theme.ts` runtime helper,
  default preset, Tailwind config wired to tokens. FastAPI server skeleton in
  `src/podcast_scraper/server/` with `/api/health` and `/api/artifacts` endpoints.
  `podcast serve` CLI command. `make serve` target. Verify Vite dev proxy works.
  Platform route placeholder stubs (empty files, not mounted).
- **M2 — Graph Port** (done): `GraphCanvas.vue` wrapping Cytoscape.js with all v1
  capabilities: load artifacts, render nodes/edges, filter by type, grounded-only
  toggle, legend with click-to-solo, 1-hop ego focus on double-click, node detail panel.
  Parsing logic ported from `shared.js` to typed TypeScript (`parsing.ts`, `colors.ts`).
  Merge logic (same-layer, GI+KG cross-layer) ported to `artifacts` store.
- **M3 — Metrics + Dashboard** (done): `MetricsPanel.vue` (key-value display),
  `IndexStats.vue` (vector counts, model info, freshness), distribution chart
  (Chart.js bar). File-picker fallback for no-backend mode. `DashboardView` with
  corpus overview.
- **M4 — Search Panel + Backend** (done): `SearchBar`, `SearchResults`,
  `SearchFilters`, `ResultCard`. `/api/search` wrapping `VectorStore.search()`.
  Click result → graph focus. Search filters (type, feed, date, speaker, grounded-only,
  top-k). Depends on RFC-061 `VectorStore` being available.
- **M5 — Explore/QA Integration** (done): `/api/explore` endpoint wrapping
  `run_uc5_insight_explorer` and `run_uc4_semantic_qa`. Topic explorer and QA view in
  frontend. Semantic matching when vector index available, substring fallback when not.
- **M6 — Polish** (done): Dark mode (Tailwind, driven by `tokens.css`
  `prefers-color-scheme`), finalize open tunable parameters in UXS-001 (typography,
  radii, spacing), keyboard shortcuts (/ for search, Escape to clear focus), graph
  export (PNG/SVG via Cytoscape), responsive layout, loading states, error handling,
  documentation update.
- **M7 — E2E Test Layer** (done): Playwright setup and configuration. Test
  fixtures (sample artifacts, pre-built small FAISS index). Core E2E scenarios: graph
  load, search, search-to-graph focus, filters, dashboard, offline fallback, dark mode.
  `make test-ui-e2e` target. CI integration.

**Total estimate:** ~18-23 days of focused work across 7 milestones.

**Monitoring:**

- API response times logged (FastAPI middleware)
- Frontend performance: graph render time, search latency (console metrics)
- Bundle size tracked per build

**Success Criteria:**

1. All v1 graph + metrics capabilities available in v2 (feature parity)
2. Semantic search returns results and highlights graph nodes
3. Index health dashboard shows vector counts, feed coverage, freshness
4. File-picker fallback works without backend
5. `podcast serve` and `make serve` start the full application in one command
6. `make ci-fast` passes with server backend tests included
7. Dark mode works correctly, driven by UXS-001 semantic tokens in `tokens.css`
8. Theme presets load correctly; swapping a preset changes typography/spacing/radii
   without touching component code
9. Playwright E2E tests pass in CI (headless Firefox)
10. Server architecture supports adding platform routes (#50, #347) without restructuring
11. v1 `web/gi-kg-viz/` removed (Phase 4 complete)
12. All documentation deliverables complete (see Definition of Done below)

**Definition of Done — Documentation Deliverables:**

The following documentation must be created or updated before this RFC is considered
complete. These are tracked as part of the implementation issue.

| Deliverable | Action | Description | Status |
| ----------- | ------ | ----------- | ------ |
| `docs/guides/SERVER_GUIDE.md` (new) | Create | Comprehensive server guide: tech stack rationale, architecture overview, route conventions, how to add new routes, how to add platform routes, configuration, `podcast serve` usage, dev workflow, extension patterns. This is the go-to reference for anyone extending the server. | Created |
| `docs/architecture/ARCHITECTURE.md` | Update | Add the server module (`src/podcast_scraper/server/`) to the module map. Document the "one pipeline core, multiple shells" pattern. Add the CLI → Server → Platform evolution diagram. | Updated |
| `docs/architecture/TESTING_STRATEGY.md` | Update | Add the UI testing layers (Playwright E2E, visual regression). Document Makefile targets (`make test-ui-e2e`). Document test fixture strategy for viewer E2E. | Updated |
| `docs/guides/DEVELOPMENT_GUIDE.md` | Update | Add `make serve`, `make serve-api`, `make serve-ui`, `make test-ui`, `make test-ui-e2e` to the developer commands section. Document the Vite dev proxy setup. | Updated |
| `README.md` | Update | Add `podcast serve` command to the CLI reference. Mention the viewer and link to `SERVER_GUIDE.md`. | Updated |
| `web/gi-kg-viewer/README.md` (new) | Create | Frontend-specific README: how to install, dev, build, test, lint. Component architecture overview. How to add new views/stores. | Created |
| `docs/guides/TESTING_GUIDE.md` | Update | Add Playwright E2E section: setup, running, writing new tests, CI integration, fixture management. | Updated |

## Platform Evolution Path

This section documents how the server and frontend architecture introduced by this RFC
grows into the full platform vision described in the
[megasketch](../architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md) and tracked by
[#50](https://github.com/chipi/podcast_scraper/issues/50) (simple UI + server),
[#347](https://github.com/chipi/podcast_scraper/issues/347) (UI for DB output), and
[#46](https://github.com/chipi/podcast_scraper/issues/46) (Docker architecture).

### What This RFC Establishes (v2.6)

| Component | What Gets Built | Where |
| --------- | --------------- | ----- |
| Server module | `src/podcast_scraper/server/` with app factory, feature flags, viewer routes | Python package |
| CLI command | `podcast serve --output-dir` | CLI layer |
| Viewer routes | `/api/health`, `/api/artifacts`, `/api/search`, `/api/explore`, `/api/index/stats` | Server routes |
| Frontend SPA | Vue 3 + Cytoscape.js viewer with search, dashboard, explore views | `web/gi-kg-viewer/` |
| E2E tests | Playwright test suite for viewer regression testing | `web/gi-kg-viewer/e2e/` |

### What Platform Work Adds (v2.7 — #50, #347)

| Component | What Gets Added | How It Fits |
| --------- | --------------- | ----------- |
| Platform routes | `/api/feeds` (catalog CRUD), `/api/episodes` (browsing), `/api/jobs` (pipeline runs), `/api/status` (monitoring) | New route files in `server/routes/`, mounted when `enable_platform=True` |
| Platform stores | `feeds.ts`, `jobs.ts`, `episodes.ts` in Pinia | New stores alongside viewer stores |
| Platform views | Feed management, episode browser, job dashboard, config editor | New `vue-router` routes: `/feeds`, `/episodes`, `/jobs`, `/settings` |
| DB integration | Postgres reads for episodes/summaries (#347, RFC-051) | New dependency in `server/dependencies.py` |
| `podcast serve --platform` | Activates platform routes + DB connection | Flag in CLI + app factory |

### What Docker/Deployment Adds (v2.7+ — #46)

| Component | What Gets Added | How It Fits |
| --------- | --------------- | ----------- |
| Docker Compose | `postgres` + `api` + `worker` + `caddy` services | Megasketch Part B |
| Worker process | Same image, different `command` — consumes job queue | Uses `service.run()` internally |
| Static build | Vite output baked into Docker image | `npm run build` in Dockerfile |

### Growth Path (No Architectural Rewrites)

```text
v2.6 (this RFC)
├── src/podcast_scraper/server/
│   ├── app.py (enable_viewer=True)
│   └── routes/ (health, artifacts, search, explore, index_stats)
├── web/gi-kg-viewer/ (Vue 3 SPA)
└── podcast serve --output-dir

v2.7 (#50, #347)
├── src/podcast_scraper/server/
│   ├── app.py (enable_viewer=True, enable_platform=True)
│   └── routes/ (+ feeds, episodes, jobs, status)
├── web/gi-kg-viewer/
│   └── src/views/ (+ FeedsView, EpisodesView, JobsView, SettingsView)
└── podcast serve --output-dir --platform

v2.7+ (#46, megasketch)
├── docker-compose.yml (postgres, api, worker, caddy)
├── src/podcast_scraper/server/ (same module, running in container)
└── podcast serve --platform --db-url postgres://...
```

The key property: **nothing gets restructured between v2.6 and v2.7+.** The server
module, the frontend SPA, the route pattern, the Pinia stores, and the Playwright
test structure all extend additively. This is the payoff of naming it `server/`
instead of `viewer/` and using feature flags instead of separate entry points.

## Relationship to Other RFCs

This RFC (RFC-062) is both a **consumption layer** for the GIL/KG depth initiative
([#466](https://github.com/chipi/podcast_scraper/issues/466)) and the **seed of the
platform server** ([#50](https://github.com/chipi/podcast_scraper/issues/50),
[#347](https://github.com/chipi/podcast_scraper/issues/347), megasketch):

```text
RFC-049 (GIL Core)             → artifacts visualized in graph
RFC-050 (GIL Use Cases)        → UC4/UC5 exposed via explore/search UI
RFC-055 (KG Core)              → KG artifacts visualized in graph
    ↓
RFC-061 (Semantic Search)      → search backend consumed by viewer
    ↓
RFC-062 (this RFC)             → server + viewer SPA + E2E tests
    ↓
RFC-051 (DB Projection)        → structured query backend (platform reads)
    ↓
#50 / #347 / megasketch        → platform routes + UI views on same server
    ↓
#46 / megasketch Part B        → Docker Compose deployment
```

**Key Distinction:**

- **RFC-061**: Defines the search *engine* — `VectorStore`, FAISS, embed-and-index
  pipeline, CLI commands
- **RFC-062**: Defines the *server layer* and the search *UI* — FastAPI server with
  pluggable route groups, Vue 3 frontend, graph integration, dashboard, E2E tests.
  The server is the first building block of the platform; the viewer is its first UI.

Together, RFC-061 provides the retrieval engine, RFC-062 provides the server and
visual interface, and the platform work (#50, #347, megasketch) extends both with
CRUD routes, job management, and Postgres integration — all on the same
`src/podcast_scraper/server/` foundation.

## Benefits

1. **Surfaces semantic search visually**: The most powerful new capability (RFC-061)
   becomes accessible through a graphical interface, not just CLI
2. **Proper web architecture**: Vue 3 + Pinia + TypeScript replaces fragile vanilla JS
   with a maintainable, type-safe component model
3. **Single graph engine**: Halves the graph code, focuses maintenance on Cytoscape.js
4. **Platform foundation**: The `server/` module is designed to grow into the full
   platform API (megasketch Part A) — viewer routes land first, platform routes (#50,
   #347) extend additively without restructuring
5. **Graceful degradation**: File-picker fallback preserves offline use; search features
   activate when backend + index are available
6. **Developer experience**: Vite HMR, TypeScript, ESLint, Vitest — standard modern
   tooling instead of CDN script tags and browser refresh
7. **Regression safety**: Playwright E2E test layer catches visual and functional
   regressions before they reach users. Test structure grows with platform views.
8. **One server, one CLI**: `podcast serve` is the single entry point for all server
   functionality — viewer today, platform tomorrow. No fragmented scripts.

## Migration Path

1. **Phase 1 — Build v2** : New `web/gi-kg-viewer/` directory and
   `src/podcast_scraper/server/` module.
2. **Phase 2 — Validate parity** : v1 capabilities work in v2. Playwright
   E2E tests green.
3. **Phase 3 — Switch default** : `make serve` points to v2. `podcast serve` is the
   canonical server command.
4. **Phase 4 — Remove v1** : `web/gi-kg-viz/` and `scripts/gi_kg_viz_server.py`
   deleted. v2 is the sole viewer.
5. **Phase 5 — Platform extension** (v2.7): Add platform routes, views, and stores to
   the same server and frontend. No structural migration needed.

## Open Questions

1. **Graph export formats**: Should the viewer support exporting the graph as data
   (JSON, GraphML) in addition to image (PNG, SVG)? Useful for external tools.
   Recommendation: defer to M6 polish; image export first.
2. **Platform route stubs**: Should the v2.7 platform route files (`feeds.py`,
   `episodes.py`, `jobs.py`) ship as empty stubs with `pass` handlers in v2.6, or
   should they be added only when platform work starts? Recommendation: empty stubs
   in v2.6 to make the growth path visible and validate the feature-flag pattern.
3. **Frontend directory rename**: When platform views are added (v2.7), should
   `web/gi-kg-viewer/` be renamed to `web/app/` to reflect the broader scope?
   Recommendation: defer — rename is trivial and can happen when platform views
   actually land. Viewer naming is accurate for v2.6.

**Resolved Questions (from earlier draft):**

- **Vue router vs tabs** — Resolved: use `vue-router` for clean view separation, URL
  deep linking, and alignment with platform views that will be added as routes.
- **E2E test tooling** — Resolved: Playwright. Official Vue recommendation, headless
  CI, multi-browser, lighter than Cypress. See Testing Strategy section.
- **Bundle vendoring** — Resolved: `npm install` via `package.json`. Vite bundles
  everything; vendoring is a v1 concern solved by proper tooling.
- **Backend naming** — Resolved: `src/podcast_scraper/server/`, not `viewer/`. This
  is the project's server layer, not just the viewer's backend.

## References

- **Related PRD**: `docs/prd/PRD-021-semantic-corpus-search.md`
- **Related RFC**: `docs/rfc/RFC-061-semantic-corpus-search.md`
- **Related RFC**: `docs/rfc/RFC-049-grounded-insight-layer-core.md`
- **Related RFC**: `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md`
- **Related RFC**: `docs/rfc/RFC-055-knowledge-graph-layer-core.md`
- **Related RFC**: `docs/rfc/RFC-056-knowledge-graph-layer-use-cases.md`
- **Related RFC**: `docs/rfc/RFC-051-database-projection-gil-kg.md`
- **Viewer v1**: `web/gi-kg-viz/` (removed — was the #445 prototype)
- **Platform Vision**: `docs/architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md`
- **Platform Issues**: [#50](https://github.com/chipi/podcast_scraper/issues/50) (UI + server),
  [#347](https://github.com/chipi/podcast_scraper/issues/347) (UI for DB),
  [#46](https://github.com/chipi/podcast_scraper/issues/46) (Docker architecture)
- **Source Code**: `scripts/gi_kg_viz_server.py` (removed — replaced by `server/`)
- **Source Code**: `podcast_scraper/gi/explore.py` (explore/query logic)
- **Source Code**: `podcast_scraper/service.py` (one-shot pipeline execution)
