# RFC-062: GI/KG Viewer v2 — Semantic Search UI & Web Application Architecture

- **Status**: Completed (v2.6.0) — Vue 3 SPA **`web/gi-kg-viewer/`**, FastAPI
  **`src/podcast_scraper/server/`**, **`podcast serve`** / **`make serve`**, Playwright E2E.
  The **shell** (nav, **status bar** corpus path, **Dashboard**, graph host, **left** query column
  for semantic search + explore, **right** subject column, health matrix) lives here; **Library**, **Digest**, extended **corpus** and **index rebuild**
  APIs, and **graph exploration chrome** are specified in sibling RFCs but **ship inside the same
  app** (see **Delivered scope** below).
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, GIL/KG consumers, corpus operators, viewer users
- **Related PRDs**:
  - [PRD-017: Grounded Insight Layer](../prd/PRD-017-grounded-insight-layer.md) — GI graph content
  - [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md) — KG graph content
  - [PRD-021: Semantic Corpus Search](../prd/PRD-021-semantic-corpus-search.md) — semantic search UI
  - [PRD-022: Corpus Library & Episode Browser](../prd/PRD-022-corpus-library-episode-browser.md) —
    **Library** tab (with [RFC-067](RFC-067-corpus-library-api-viewer.md))
  - [PRD-023: Corpus Digest & Library Glance](../prd/PRD-023-corpus-digest-recap.md) — **Digest** tab
    (with [RFC-068](RFC-068-corpus-digest-api-viewer.md))
  - [PRD-024: Graph exploration toolkit](../prd/PRD-024-graph-exploration-toolkit.md) — graph chrome
    (with [RFC-069](RFC-069-graph-exploration-toolkit.md))
- **Related ADRs**:
  - [ADR-064: Canonical server layer with feature-flagged routes](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)
  - [ADR-066: Playwright for UI E2E testing](../adr/ADR-066-playwright-for-ui-e2e-testing.md)
- **Related RFCs**:
  - [RFC-061: Semantic corpus search](RFC-061-semantic-corpus-search.md) — search backend
  - [RFC-049: GIL core](RFC-049-grounded-insight-layer-core.md) — GI artifact shape
  - [RFC-050: GIL use cases](RFC-050-grounded-insight-layer-use-cases.md) — explore / QA patterns
  - [RFC-055: KG core](RFC-055-knowledge-graph-layer-core.md) — KG artifact shape
  - [RFC-056: KG use cases](RFC-056-knowledge-graph-layer-use-cases.md)
  - [RFC-067: Corpus library API & viewer](RFC-067-corpus-library-api-viewer.md) — **`/api/corpus/feeds`**
    **`/episodes`**, Library UX, episode subject-rail handoffs
  - [RFC-068: Corpus digest API & viewer](RFC-068-corpus-digest-api-viewer.md) — **`/api/corpus/digest`**
    **Digest** tab, discovery flows
  - [RFC-069: Graph exploration toolkit](RFC-069-graph-exploration-toolkit.md) — minimap, degree
    filter, layouts, zoom affordances on the graph card
  - [RFC-077: Viewer feeds, operator YAML, pipeline jobs on `serve`](RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) —
    opt-in **`/api/feeds`**, **`/api/operator-config`**, **`/api/jobs`** (not `enable_platform`)
  - [RFC-051: Database projection GIL/KG](RFC-051-database-projection-gil-kg.md) — future structured
    query backend
- **Related UX specs**:
  - [VIEWER_IA: Viewer information architecture](../uxs/VIEWER_IA.md) — **canonical** shell IA (regions, axes, persistence, clearing, first-run)
  - [UXS-001: GI/KG viewer](../uxs/UXS-001-gi-kg-viewer.md) — shared design system (tokens, typography, **visual** chrome, accessibility)
  - [UXS-002 Corpus Digest](../uxs/UXS-002-corpus-digest.md), [UXS-003 Corpus Library](../uxs/UXS-003-corpus-library.md),
    [UXS-004 Graph Exploration](../uxs/UXS-004-graph-exploration.md), [UXS-005 Semantic Search](../uxs/UXS-005-semantic-search.md),
    [UXS-006 Dashboard](../uxs/UXS-006-dashboard.md) — per-surface visual contracts (see [UXS index](../uxs/index.md))
- **Related Documents**:
  - [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) —
    Playwright contract (selectors, surfaces); **update when UI or E2E changes** (repo path:
    `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md`)
  - [Platform architecture blueprint](../architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md)
  - [GitHub #489](https://github.com/chipi/podcast_scraper/issues/489) — Viewer v2 implementation
  - [GitHub #509](https://github.com/chipi/podcast_scraper/issues/509) — E2E / surface map hygiene
  - [GitHub #445](https://github.com/chipi/podcast_scraper/issues/445) — Viewer v1 (removed)
  - [GitHub #484](https://github.com/chipi/podcast_scraper/issues/484) — Semantic search phase 1
  - [GitHub #466](https://github.com/chipi/podcast_scraper/issues/466) — GI + KG depth roadmap
  - [GitHub #50](https://github.com/chipi/podcast_scraper/issues/50) — Platform UI + server (v2.7)
  - [GitHub #347](https://github.com/chipi/podcast_scraper/issues/347) — UI for DB-backed corpus (v2.7)
  - [GitHub #46](https://github.com/chipi/podcast_scraper/issues/46) — Docker architecture (v2.7)
  - [GitHub #606](https://github.com/chipi/podcast_scraper/issues/606) — Viewer shell restructure (query left, subject right, status bar)
- **Updated**: 2026-04-19 — shell IA #606 narrative; **RFC-077** server flags, route tree, and `enable_platform` vs opt-in feeds/operator/jobs

## Abstract

This RFC defines the **v2 GI/KG viewer**: a **Vue 3 + Vite** single-page app (**`web/gi-kg-viewer/`**)
backed by the canonical FastAPI layer (**`src/podcast_scraper/server/`**). It replaces the removed
vanilla-JS prototype (**`web/gi-kg-viz/`**, #445) with one **“Podcast Intelligence Platform”** shell:
main navigation (**Digest**, **Library**, **Graph**, **Dashboard**), a **left** query column (**Semantic search** default; **Explore** as a slide-in mode from the same column), a **right** subject column (episode / graph-node / future topic-person), semantic search (**RFC-061**), a **Cytoscape.js** graph host, and a **Dashboard** with
**Coverage** / **Intelligence** / **Pipeline** sub-tabs (see **UXS-006**). **Pinia**, **TypeScript**, and **Tailwind**
(UXS-001 tokens) carry state and styling; **`podcast serve`** serves the built SPA plus **`/api/*`**.

**Shipped layout (#606, 2026-04):** **Shell IA** is normative in **[VIEWER_IA](../uxs/VIEWER_IA.md)**. Corpus path and offline **Files** live on the **status bar**; **List** opens **`artifact-list-dialog`** (load / graph handoffs). **Dashboard** ships as **briefing** plus **Coverage** / **Intelligence** / **Pipeline** (index rebuild and coverage charts on **Coverage** — see **UXS-006**), not a legacy left **API · Data** workspace component. The **Delivered scope** table below reflects this; see
[#606](https://github.com/chipi/podcast_scraper/issues/606), **VIEWER_IA**, and **Active** UXSs.

The same release expanded the **server** beyond the original search + artifacts + explore + index
stats sketch: **corpus metrics**, **library**, **digest**, **binary/covers**, **`POST /api/index/rebuild`**
and **`GET /api/corpus/runs/summary`** (and related routes) power the **Dashboard** workspace, **Library**, **Digest**,
and related graph surfaces. Behavioral and API detail for those areas lives in **RFC-067**, **RFC-068**,
and **RFC-069**; graph chrome specifics (minimap, degree buckets, layout cycle + **Re-layout**, filters popover, export PNG) are
**RFC-069** on top of this RFC’s graph integration patterns.

The FastAPI package remains the **seed of the platform API** ([#50](https://github.com/chipi/podcast_scraper/issues/50),
[#347](https://github.com/chipi/podcast_scraper/issues/347)): **`enable_platform`** in **`app.py`**
is still reserved for future megasketch **catalog / DB** routers under **`routes/platform/`**.
**RFC-077** adds separate flags (**`enable_feeds_api`**, **`enable_operator_config_api`**, **`enable_jobs_api`**) for corpus RSS list file, viewer-safe operator YAML, and HTTP pipeline jobs — see [RFC-077](RFC-077-viewer-feeds-and-serve-pipeline-jobs.md). **v2.6+** mounts **viewer + corpus** routes unconditionally; RFC-077 routes are **opt-in**.

**Architecture alignment:** The server is a **consumption and coordination layer** — it does not mutate
artifacts or pipeline outputs. It wraps **`VectorStore`**, **`gi explore`**, filesystem artifacts, and
corpus helpers behind REST. The SPA uses a **file-picker fallback** when **`/api/health`** fails.
This follows megasketch **A.2** — **one pipeline core, multiple shells** (CLI vs server).

## Delivered scope (v2.6 viewer shell)

| Surface | What shipped (summary) | Normative detail |
| ------- | ---------------------- | ---------------- |
| **App shell** | Header (**Podcast Intelligence Platform** + v2), **Main views** tabs, **left** query column (**Semantic search** / **Explore** mode), **status bar** (corpus path, health, offline files), **right** subject column (episode / graph-node / …) | [VIEWER_IA](../uxs/VIEWER_IA.md) + [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) |
| **Digest** | Default entry tab for online mode; rolling window, topic bands, **Recent** list, **Episode** subject rail, **Search topic** handoff | [UXS-002](../uxs/UXS-002-corpus-digest.md) + [RFC-068](RFC-068-corpus-digest-api-viewer.md) |
| **Library** | Feed list + cursor-paginated episodes, filters, **Episode** subject rail (**Open in graph**, **Prefill semantic search**, similar episodes) | [UXS-003](../uxs/UXS-003-corpus-library.md) + [RFC-067](RFC-067-corpus-library-api-viewer.md) |
| **Graph** | Merged GI/KG load, **Sources** / **Types** / **Edges**, search-hit focus, **Episode** subject rail on graph when metadata resolves; **toolbar** zoom/fit/export; **RFC-069** overlays (layout, degree filter, minimap, Shift+box zoom) | [UXS-004](../uxs/UXS-004-graph-exploration.md) + this RFC + [RFC-069](RFC-069-graph-exploration-toolkit.md) |
| **Semantic search** | **`#search-q`**, since / top‑k, **Advanced search** modal, **Search result insights** modal, **G** / **L** actions, merge duplicate KG surfaces | [UXS-005](../uxs/UXS-005-semantic-search.md) + [RFC-061](RFC-061-semantic-corpus-search.md) |
| **Dashboard** | **Briefing** card; **Coverage** / **Intelligence** / **Pipeline** sub-tabs; Chart.js surfaces per **UXS-006**; corpus **List** / **Load into graph** via **status bar** (not a legacy **`CorpusDataWorkspace`** on the tab body) | [UXS-006](../uxs/UXS-006-dashboard.md) + [RFC-071](RFC-071-corpus-intelligence-dashboard-viewer.md) + [VIEWER_IA](../uxs/VIEWER_IA.md) (status bar) |
| **Server** | **`app.py`** mounts **health**, **artifacts**, **search**, **explore**, **index_stats**, **index_rebuild**, **corpus_library**, **corpus_binary**, **corpus_metrics**, **corpus_digest**, …; optional **RFC-077** **`/api/feeds`**, **`/api/operator-config`**, **`/api/jobs`** | [ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md), [RFC-077](RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) |
| **E2E** | Playwright under **`web/gi-kg-viewer/e2e/`** (Firefox; dedicated Vite port in CI) | [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md), [ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md) |

The table documents **shipped** viewer chrome including the **#606** shell IA update.

## Shell restructure (#606, shipped)

**Information architecture:** query column left (**Semantic search** default; **Explore** mode), main view tabs,
**subject** column right, **status bar** for corpus path / health / offline files / **List** (artifacts dialog). **Normative** shell IA: **[VIEWER_IA](../uxs/VIEWER_IA.md)**. **Normative** layout and tokens
stay in **Active** UXSs (see
[UX specifications index — Living documents and ship boundary](../uxs/index.md#living-documents-and-ship-boundary)). Track history in
[GitHub #606](https://github.com/chipi/podcast_scraper/issues/606).

### Behavioral invariants

- **Corpus path change** clears current **subject** context and must keep the existing
  refresh cascade (artifacts list, library, digest, dashboard, index stats) wired from
  `shellStore` / `App.vue` watchers.
- **Subject** selection in the right **subject rail** **persists** across **main tab**
  switches (Digest / Library / Graph / Dashboard). Clear on explicit rail close, new
  subject selection, or corpus path change.
- With **main tab** **Graph** and subject kind episode / topic / person (where
  applicable), preserve graph **focus / center** on the resolved Cytoscape node (existing
  `graphNavigation` patterns).
- **StaleGeneration** and other async gates must **not** be removed or bypassed during
  shell-only refactors.
- **Offline** GI/KG load via folder / file picker remains supported on the **status bar**.

### Focus and keyboard (shell-specific)

Global shortcuts (**slash** for search focus, **Escape** on Graph, per-row **G** / **L** on semantic
search hits) are documented in [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) and implemented in
`useViewerKeyboard.ts`. Extend the
[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
when automation-visible strings change.

## Problem Statement

**v2.6 scope note:** The shipped app addresses the v1 gaps below **and** adds corpus-scale
navigation (**Digest**, **Library**), aggregate **Dashboard** analytics (briefing + sub-tabs; **index rebuild** on **Coverage**), **status bar** corpus/artifact flows, and deep graph chrome (**RFC-069**) — all in one SPA. The
following bullets remain the historical motivation; **Delivered scope** summarizes what landed.

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
11. **Corpus discovery tabs** — **Digest** and **Library** main views with shared **Episode** subject rail
    patterns and handoffs to graph and search (**RFC-068**, **RFC-067**)
12. **Corpus and runs APIs** — **`/api/corpus/*`** for stats, feeds, episodes, digest, runs summary,
    binary/covers; extended **`/api/health`** capability flags for the **Dashboard** API card
13. **Index lifecycle in UI** — **`GET /api/index/stats`** plus **`POST /api/index/rebuild`**
    (incremental vs full) from **Dashboard → Coverage** (see **UXS-006**), with disabled states while rebuild runs
14. **Graph exploration toolkit** — On-canvas and card chrome for fit/zoom/export, layouts, degree
    buckets, minimap, Shift+drag box zoom (**RFC-069**)
15. **Dashboard analytics** — **Coverage** / **Intelligence** / **Pipeline** sub-tabs driven by corpus +
    index + artifact aggregate endpoints (**UXS-006**)
16. **Playwright regression suite** — Specs and **E2E surface map** kept in lockstep with UI
    (**ADR-066**, [#509](https://github.com/chipi/podcast_scraper/issues/509))

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
│   │   │   ├── GraphCanvas.vue       # cytoscape wrapper + on-card type/edge tools
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
├── app.py                 # FastAPI factory: mounts routers below + optional StaticFiles → dist/
├── pathutil.py            # Corpus path validation / errors
├── routes/
│   ├── health.py          # GET /api/health (+ capability flags for UI matrix)
│   ├── artifacts.py       # GET /api/artifacts, artifact bodies, metadata paths
│   ├── search.py          # GET /api/search
│   ├── explore.py         # GET /api/explore
│   ├── index_stats.py     # GET /api/index/stats
│   ├── index_rebuild.py   # POST /api/index/rebuild
│   ├── corpus_library.py  # feeds, episodes, similar (RFC-067)
│   ├── corpus_digest.py   # digest window + topic bands (RFC-068)
│   ├── corpus_metrics.py  # corpus stats, runs summary, aggregates for Dashboard
│   ├── corpus_binary.py   # episode cover / binary helpers
│   ├── feeds.py           # optional RFC-077 — GET/PUT /feeds
│   ├── operator_config.py # optional RFC-077 — GET/PUT /operator-config
│   ├── jobs.py            # optional RFC-077 — POST/GET /jobs (+ cancel, reconcile)
│   └── platform/          # reserved (#50, #347); not used for RFC-077
├── dependencies.py        # shared wiring (output_dir, stores) as implemented
└── schemas.py             # Pydantic models shared by routes
```

**FastAPI app factory (`app.py`) — shipped behavior:**

- **`create_app(output_dir, static_dir=..., enable_platform=False, enable_feeds_api=False, …)`** — all **viewer + corpus**
  routers in the tree above (except the three optional RFC-077 files) are **`include_router(..., prefix="/api")`** unconditionally.
- **`enable_feeds_api` / `enable_operator_config_api` / `enable_jobs_api`**: when true, mount **RFC-077** routers (`feeds`, `operator_config`, `jobs`). Env aliases: `PODCAST_SERVE_ENABLE_*` (reload factory).
- **`enable_platform`**: reserved; **`routes/platform/`** exists for future **#50 / #347** work but does
  not mount routers yet.
- **`static_dir`**: when **`web/gi-kg-viewer/dist`** exists (or path passed), mounts **`StaticFiles`**
  at **`/`** with **`html=True`** for SPA fallback.
- **CORS**: allows local Vite ports **5173** / **5174** for dev.

Other modules (**middleware**, extra **schemas** split) may evolve; the list above is the **shipped**
route surface for the viewer.

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

**Corpus routes (v2.6, shipped — not “platform placeholders”):**

- **`/api/corpus/*`** — feeds, episodes, digest, aggregate stats, runs summary, binary/covers, etc.
  (**RFC-067**, **RFC-068**, metrics helpers). These are **filesystem-first** catalog/digest APIs for
  the viewer, distinct from future **#50 / #347** CRUD + job routes under **`enable_platform`**.

**Platform route contracts (v2.7, future):**

```python
# Future under routes/platform/ + enable_platform:
# CRUD /api/feeds, job runners, Postgres-backed corpus, … (#50, #347, megasketch)
```

**Integration with existing code:**

The server imports and calls existing library code — it does not reimplement search/explore logic:

- `VectorStore.search()` for `/api/search`
- `run_uc5_insight_explorer()` (or equivalent) for `/api/explore`
- `IndexStats` / rebuild orchestration for `/api/index/stats` and `/api/index/rebuild`
- Direct file reads for `/api/artifacts`
- Corpus filesystem scans and metadata joins for **`/api/corpus/*`** (see **RFC-067** / **RFC-068**)

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
| Load artifacts | `/api/artifacts` (online path can auto-list + merge GI/KG) | File picker / `showDirectoryPicker()` |
| Graph rendering | Full | Full |
| Metrics panel | Full | Full (computed from loaded artifacts) |
| Semantic search | Full | Disabled (instructional copy in UI) |
| Explore/QA | Full | Disabled |
| Index stats / rebuild | Full | Disabled |
| **Digest** / **Library** / **Dashboard** corpus charts | Full when **`/api/corpus/*`** healthy | Degraded (no catalog, digest, or run summaries) |

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
| `make test-ui-e2e` | Playwright browser E2E (Firefox; see **`web/gi-kg-viewer/e2e/`**) |

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
     starts. **v2.6** mounts **viewer + corpus** API routers by default; **`enable_platform`**
     in **`create_app`** is reserved for future platform-only routers (**ADR-064**). This
     follows megasketch constraint A.2: "One pipeline core, multiple shells." The
     **`podcast serve`** CLI starts this server (replacing **`scripts/gi_kg_viz_server.py`**).

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
   - **Decision**: Pinia stores for artifacts, search, graph, index, corpus path, and
     feature-specific slices (library/digest/dashboard wiring as implemented)
   - **Rationale**: Replaces the `window.GiKgViz` / `window.GiKgVizShell` globals
     with typed reactive stores. Components subscribe to store state and dispatch
     actions. Cross-component coordination (search result → graph focus, Library episode →
     rail) happens through store watchers, not DOM events or callback chains. Additional
     platform stores (**feeds**, **jobs**, …) can land under **`enable_platform`** in v2.7.

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
- `tests/unit/podcast_scraper/server/` — backend unit tests (pytest; `importorskip("fastapi")` when `[server]` not installed)
- `tests/integration/server/` — backend API integration (pytest): `test_server_api.py`, Corpus Library, index rebuild, index stats, etc.

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
RFC-062 (this RFC)             → FastAPI app + Vue SPA shell + graph + dashboard + E2E
    ├─ RFC-067 (Library)       → /api/corpus/feeds|episodes|… + Library tab
    ├─ RFC-068 (Digest)        → /api/corpus/digest + Digest tab
    └─ RFC-069 (Graph toolkit) → Cytoscape chrome (minimap, degree, layouts, zoom)
    ↓
RFC-051 (DB Projection)        → structured query backend (platform reads)
    ↓
#50 / #347 / megasketch        → platform routes + UI views on same server
    ↓
#46 / megasketch Part B        → Docker Compose deployment
```

**Key distinction:**

- **RFC-061**: Search *engine* — `VectorStore`, FAISS, embed-and-index pipeline, CLI
- **RFC-062**: *Server process* + *viewer shell* + graph/search/dashboard integration;
  **RFC-067** / **RFC-068** / **RFC-069** extend the **same** app with corpus and graph UX
- **Platform (#50, #347)**: CRUD, jobs, Postgres — **`enable_platform`**, same **`server/`**
  package (**ADR-064**)

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
7. **Regression safety**: Playwright E2E (**`web/gi-kg-viewer/e2e/`**) and the **E2E surface map**
   catch visual and functional regressions; keep map and specs aligned when changing UI.
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
2. **Platform route stubs**: **Resolved for v2.6** — **corpus** needs are covered by
   **`corpus_library.py`**, **`corpus_digest.py`**, **`corpus_metrics.py`**, etc. **v2.7**
   platform CRUD/jobs will live under **`routes/platform/`** when **`enable_platform`**
   mounts them; no duplicate stub **`feeds.py`** beside corpus routes.
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
- **Backend naming** — Resolved: `src/podcast_scraper/server/`, not `viewer/`. This is the
  project's server layer, not just the viewer's backend.

## References

- **PRDs**: [PRD-017](../prd/PRD-017-grounded-insight-layer.md), [PRD-019](../prd/PRD-019-knowledge-graph-layer.md),
  [PRD-021](../prd/PRD-021-semantic-corpus-search.md), [PRD-022](../prd/PRD-022-corpus-library-episode-browser.md),
  [PRD-023](../prd/PRD-023-corpus-digest-recap.md), [PRD-024](../prd/PRD-024-graph-exploration-toolkit.md)
- **RFCs**: [RFC-061](RFC-061-semantic-corpus-search.md), [RFC-067](RFC-067-corpus-library-api-viewer.md),
  [RFC-068](RFC-068-corpus-digest-api-viewer.md), [RFC-069](RFC-069-graph-exploration-toolkit.md),
  [RFC-049](RFC-049-grounded-insight-layer-core.md), [RFC-050](RFC-050-grounded-insight-layer-use-cases.md),
  [RFC-055](RFC-055-knowledge-graph-layer-core.md), [RFC-056](RFC-056-knowledge-graph-layer-use-cases.md),
  [RFC-051](RFC-051-database-projection-gil-kg.md)
- **UXS**: [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) (hub); [UXS index](../uxs/index.md) (feature specs)
- **E2E**: [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) (`web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md`)
- **ADRs**: [ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md),
  [ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md)
- **Platform vision**: [PLATFORM_ARCHITECTURE_BLUEPRINT.md](../architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md)
- **Platform issues**: [#50](https://github.com/chipi/podcast_scraper/issues/50),
  [#347](https://github.com/chipi/podcast_scraper/issues/347),
  [#46](https://github.com/chipi/podcast_scraper/issues/46)
- **Source**: `src/podcast_scraper/server/` (FastAPI), `web/gi-kg-viewer/` (Vue SPA),
  `podcast_scraper/gi/explore.py`, `podcast_scraper/service.py`
- **Removed**: `web/gi-kg-viz/` (v1), `scripts/gi_kg_viz_server.py` (replaced by **`podcast serve`**)
