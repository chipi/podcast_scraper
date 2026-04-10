# PRD-024: GI/KG Graph Exploration Toolkit (Viewer)

## Summary

Expand the **Graph** tab in the GI/KG viewer (`web/gi-kg-viewer`) with a **single delivery pass** of graph-native controls: clearer zoom/pan affordances, **minimap (in v1)**, type/degree summaries (degree buckets **act as a filter** when clicked), edge visibility controls, **simple** built-in layouts only (cose, breadthfirst, circle, grid), **Shift+drag** box zoom, and zoom **%** readout. **Zoom reset = 100%** (`zoom(1)`); **Fit** remains fit-to-viewport. Goal is to make large merged corpora explorable without leaving the graph surface, while preserving existing behavior (filters, search highlights, ego neighborhood, export).

## Background & Context

Today the graph (Cytoscape.js, see [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md)) already supports wheel zoom, background pan, **Fit**, **Re-layout** (COSE), **Export PNG**, node-type and GI/KG layer filters, shift+double-click 1-hop ego, and semantic-search / library highlights. Power users still ask for **visible** zoom controls, **orientation** in huge graphs (minimap), **distribution** views (counts / degree), and **layout experimentation**—none of which are first-class UI yet.

## Goals

- **G1**: One coordinated implementation slice (“toolkit pass”) so interactions can be evaluated together.
- **G2**: All new controls are **graph-local** (toolbar / collapsible panel / overlay), consistent with [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) tokens.
- **G3**: **No regression** on existing E2E contracts; update [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) and Playwright as needed.
- **G4**: Acceptable performance on **medium** corpora (hundreds of nodes); document limits for **very large** graphs.

## Non-Goals

- Server-side graph analytics APIs (client-only derivation from the current Cytoscape graph).
- Replacing Cytoscape or rewriting the merge pipeline ([RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md)).
- Time-based scrubbing (no timeline in artifact schema for this PRD).

## Personas

- **Analyst**: Explores merged multi-episode graphs; needs zoom UI, minimap, and layout switching.
- **Engineer**: Validates GI vs KG structure; needs edge-type toggles and degree histogram.
- **Operator**: Captures screenshots; keeps **Export PNG** and **Fit** reliable.

## User Stories

- _As an analyst, I can zoom with buttons and see zoom level so that I don’t rely only on the wheel._
- _As an analyst, I can open a minimap so that I know where I am in a large layout._
- _As an engineer, I can see node-type and degree distributions, **click a degree bucket to filter** the graph to that range, and filter edges so that I can sanity-check the artifact._
- _As an operator, I can pick an alternative layout so that dense regions become readable._

## Functional Requirements

### FR1: Zoom & view

- **FR1.1**: Zoom in, zoom out, **Reset zoom (100%)** — `cy.zoom(1)`; distinct from **Fit** (fit graph to viewport).
- **FR1.2**: Display current **zoom %** in the graph toolbar.
- **FR1.3**: **Box zoom**: **Shift+drag** on canvas background to draw a rectangle, then zoom/fit to that region ([RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md)).

### FR2: Minimap / overview

- **FR2.1**: **Required in v1**: toggleable minimap showing full graph extent and current viewport rectangle.
- **FR2.2**: Interact on minimap to pan (or at minimum click-to-center) per [RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md).

### FR3: Histograms & summaries

- **FR3.1**: Panel or collapsible section: **node count by type** (bars or compact bars) synced with visible elements after filters.
- **FR3.2**: **Degree distribution** (binned buckets, e.g. 0, 1, 2–5, …): **clicking a bucket filters** the graph to nodes in that degree range; toggle off or explicit clear restores all (per [RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md)).

### FR4: Edge & layout controls

- **FR4.1**: Filter edges by **edge type** (or grouped types) without breaking node filters; state lives in graph filter store or sibling store.
- **FR4.2**: Layout dropdown: **cose** (default), **breadthfirst**, **circle**, **grid** only — Cytoscape built-ins; **no fcose** in this pass.

### FR5: Quality & docs

- **FR5.1**: Update **UXS-001** for new visible strings and density rules.
- **FR5.2**: Extend Playwright coverage for at least one path through new controls (see RFC).

### FR6: Chrome density

- **FR6.1**: **Second toolbar row** on the graph card **or** collapsible **Graph tools** under “Filters & sources” for minimap toggle, layout, edges, histograms ([RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md)).

## Success Criteria

- New controls usable on **Graph** tab with **keyboard + pointer** without conflicting with existing double-click / shift+dbl behaviors.
- Lighthouse of manual test: analyst can answer “how many Entity vs Topic” and “what zoom am I at” in &lt; 30 seconds on a 200-node fixture.
- `make test-ui-e2e` green after map + spec updates.

## Related Documents

- **RFC**: [RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md) (technical design).
- **Viewer**: [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md), [UXS-001](../uxs/UXS-001-gi-kg-viewer.md).
- **Automation**: [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md).
