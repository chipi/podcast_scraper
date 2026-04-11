# RFC-069: GI/KG Viewer — Graph Exploration Toolkit (Single Pass)

- **Status**: Completed (v2.6.0)
- **Authors**: (TBD)
- **Stakeholders**: Viewer maintainers, UX reviewer for [UXS-001](../uxs/UXS-001-gi-kg-viewer.md)
- **Related PRDs**:
  - [PRD-024](../prd/PRD-024-graph-exploration-toolkit.md)
- **Related RFCs**:
  - [RFC-062](RFC-062-gi-kg-viewer-v2.md) (viewer stack, Cytoscape, merge)
- **Related UX specs**:
  - [UXS-001](../uxs/UXS-001-gi-kg-viewer.md)
- **Related Documents**:
  - [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)

## Abstract

This RFC specifies a **single implementation pass** that adds graph-native exploration tooling to `web/gi-kg-viewer`: explicit zoom controls + zoom readout, **minimap (required in v1)**, node-type and degree summaries (histogram-style; **degree buckets act as a filter** when clicked), edge-type filtering, **built-in** Cytoscape layouts only (cose, breadthfirst, circle, grid—no fcose), and **Shift+drag** box zoom. It **extends** the current graph stack ([RFC-062](RFC-062-gi-kg-viewer-v2.md)) without changing artifact formats or server APIs.

## Problem Statement

The Graph tab already exposes substantial power—wheel zoom, pan, **Fit**, COSE **Re-layout**, **Export PNG**, **Sources** + type and edge filters on the graph card (`GraphCanvas.vue` + `graphFilters` store), ego neighborhood (shift+double-click), and search/library highlights—but those capabilities were **partly hidden** (zoom/pan are implicit) and **layout/analysis** options were narrow. Operators exploring **merged multi-episode** graphs need visible controls and quick **structural summaries** to iterate in one session.

## Current implementation inventory (baseline)

| Area | Location / behavior |
| ---- | -------------------- |
| Engine | `cytoscape` in `GraphCanvas.vue` |
| Zoom | `wheelSensitivity: 0.35`; debounced `center` after zoom-out |
| Pan | Default Cytoscape background drag |
| Fit / focus | `fitAnimated`, search/library focus helpers |
| Layout | `layout: { name: 'cose', ... }`; **Re-layout** = `redraw()` |
| Export | `exportGraphPng` (2× PNG) |
| Node / layer filters | `GraphCanvas.vue` chrome + `graphFilters` store (`allowedTypes`, `allowedEdgeTypes`, GI/KG, ungrounded) |
| Type visibility | Graph card **Types** row + `allowedTypes` (replaces removed click-to-solo legend) |
| Ego | `focusNodeId` + `filterArtifactEgoOneHop` via `viewWithEgo` |
| Highlights | `.search-hit` class from `search` + `graphNavigation` |

All proposed features must **compose** with this baseline (no breaking tap/dbltap semantics without migration notes).

## Goals

1. Ship **PRD-024** functional requirements in **one PR** (or tightly coupled follow-up within the same milestone).
2. Prefer **vanilla Cytoscape APIs** and **small Vue components**; minimap may use a **navigator extension** if maintained; otherwise a minimal second-instance approach—**minimap ships in v1**.
3. Keep **graph filter state** coherent: node filters, legend solo, new edge filters, and layout choice should have a clear precedence (see below).
4. Update **UXS-001**, **E2E_SURFACE_MAP.md**, and **Playwright** for new affordances.

## Non-goals

- New HTTP routes or Python changes.
- Persisting user preferences to backend (localStorage optional later).
- 3D graph or WebGL rewrite.

## Design & implementation

### 1. UI placement (stakeholder-approved)

- **Primary toolbar** (`GraphCanvas.vue` top row): existing **Fit**, **Re-layout**, **Export PNG**; add **Zoom −**, **Zoom +**, **Zoom reset (100%)** — calls `cy.zoom(1)` (true 100% scale); **Fit** remains the control that fits the graph to the viewport. Add **Zoom %** readout (`Math.round(cy.zoom() * 100)`).
- **Secondary strip**: **Yes** — extra thin row on the graph card **or** a collapsible **Graph tools** block under “Filters & sources” (implementer’s choice for density): minimap toggle, layout `<select>`, edge-filter chips or multi-select, **Histograms** (type + degree) collapsible.
- **Minimap**: absolutely positioned overlay **bottom-right** of `.graph-canvas` when enabled; must not steal clicks from main graph except on minimap surface. **In scope for v1** (not deferred).

### 2. Zoom & box zoom

- Wire **Zoom ±** to `cy.zoom(cy.zoom() * factor)` with clamp `[minZoom, maxZoom]` (Cytoscape defaults or explicit constants).
- **Zoom reset**: **`cy.zoom(1)`** only — label **100%** or **Reset zoom** in UI; do not duplicate **Fit**.
- **Zoom %**: display `Math.round(cy.zoom() * 100)` updated on `zoom` / `pan` events (throttled with `requestAnimationFrame`).
- **Box zoom** (**decided**): **Shift+drag** on **background** to draw a rectangle (overlay), then `cy.fit` on nodes/edges intersecting the rect; document in toolbar hint and [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md). Avoid extra npm packages unless the overlay approach is unworkable.

### 3. Minimap (v1 required)

- **Option A**: Community **navigator** extension compatible with Cytoscape 3.x (e.g. evaluate `cytoscape-navigator` maintenance and bundle size).
- **Option B**: Second small `cytoscape` instance / texture-on-viewport-style sync.
- **Pick A or B in the implementation PR**; minimap **must ship** in the same pass as the rest of the toolkit. Click-drag on minimap to pan the main viewport when the extension/API allows it; if not, ship viewport rectangle + click-to-center as minimum.

### 4. Histograms

- **Node by type**: derive from `cy.nodes()` using `data('type')` or mapped visual type consistent with `graphNodeLegendLabel` keys; show horizontal bars; counts respect elements that pass upstream filters (use the same visibility pipeline as the main graph).
- **Degree distribution** (**decided — acts as filter**): compute total degree per node on the **current filtered** subgraph; fixed buckets (e.g. 0, 1, 2–5, 6–10, 11+). **Clicking a bucket sets an active degree filter**: only nodes whose degree falls in that bucket remain visible (implement via `display: 'none'` / `display: 'element'` on `cy` nodes or equivalent); **clicking the active bucket again or an explicit “Clear degree filter”** clears it. Histogram bars update when other filters/layout change; when a degree filter is active, counts can reflect “eligible” vs “visible” per implementer clarity (document in UI microcopy).

### 5. Edge filtering

- Collect **edge types** from `data('type')` or equivalent on `ParsedArtifact` edges when building elements; store `Record<string, boolean> allowedEdgeTypes` in `graphFilters` (or `graphExploration` store) default all **true**.
- Applying filters: either rebuild elements in `toCytoElements` path (cleaner for data layer) or `cy.edges().style('display', 'none')` for visible toggle (faster iteration). Prefer **data-layer** filter alongside existing `applyGraphFilters` if it stays readable.

### 6. Layout menu (**keep simple — decided**)

- Dropdown only: **`cose`** (default, match today), **`breadthfirst`**, **`circle`**, **`grid`** — all Cytoscape **built-ins**, **no fcose** / cose-bilkent / extra layout packages in this pass.
- On change: run layout on the **visible** element collection (see §7), then `cy.fit` padding—**without** full `destroyCy()` unless required.
- **Re-layout** button: re-run **currently selected** layout algorithm (not always COSE).

### 7. State precedence (normative)

1. Artifact merge + base `graphFilters` (types, layers, ungrounded) produce **filtered** graph data.
2. **Legend solo** further restricts visible types (existing).
3. **Ego** (`focusNodeId`) subsets to 1-hop (existing).
4. **Edge filters** apply on the resulting subgraph.
5. **Degree bucket filter** (from histogram interaction): when active, hides nodes outside the selected degree range (after 1–4).
6. **Layout** (**normative**): the layout dropdown and **Re-layout** must run on the **currently visible** subgraph only — i.e. `cy.elements(':visible')` (or equivalent), including after degree-bucket hiding — so positions reflect what the user sees and hidden nodes do not reserve space. Edges incident to hidden endpoints are excluded from layout naturally via visibility. Initial graph build after `redraw()` uses full filtered data until any visibility-only filter applies.

### 8. Performance & limits

- Histograms and edge-type enumeration: O(N+E) per layout stop or filter change; debounce if needed.
- Minimap: throttle sync to `layoutstop` + `viewport` events.
- Document in HelpTip: graphs **>5k** nodes may need disabling minimap or reducing layout options.

### 9. Testing

- **Unit**: pure functions for degree buckets and edge-type set extraction (if extracted).
- **E2E**: extend `e2e/*.spec.ts` or add `graph-tools.spec.ts`: mock health + minimal artifact list + GI JSON; assert **Zoom +** changes zoom label, layout dropdown switches and **Fit** still works; box zoom optional if stable in CI.
- Update [E2E_SURFACE_MAP.md](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) for every new **accessible name** or role.

### 10. Documentation

- [UXS-001](../uxs/UXS-001-gi-kg-viewer.md): graph toolbar density, minimap overlay, keyboard modifiers for box zoom.
- [PRD-024](../prd/PRD-024-graph-exploration-toolkit.md): marked **complete** (v2.6.0).

## Resolved decisions (stakeholder input)

| Topic | Decision |
| ----- | -------- |
| Minimap in v1 | **Yes** — ship in the same pass; extension vs second instance chosen in PR. |
| Zoom reset | **`cy.zoom(1)`** — **100%** scale; **Fit** stays separate. |
| Extra layouts | **Simple only** — cose, breadthfirst, circle, grid; **no fcose**. |
| Degree histogram | **Acts as filter** — click bucket toggles filter; clear control or second click clears. |
| Toolbar density | **Yes** — second row **or** collapsible **Graph tools** under Filters & sources. |
| Box zoom | **Shift+drag** on background; rectangle + `fit` intersecting elements. |
| Layout vs degree filter | **Layout uses visible elements only** — matches user perception; see §7 step 6. |

## References

- Cytoscape.js docs: layouts, `zoom`, `fit`, `viewport` events.
- Existing merge/filter: `utils/parsing.ts`, `stores/graphFilters.ts`, `GraphCanvas.vue`.
