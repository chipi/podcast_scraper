# UXS-004: Graph Exploration

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-024: Graph Exploration Toolkit](../prd/PRD-024-graph-exploration-toolkit.md)
- **Related RFCs**:
  - [RFC-069: Graph Exploration Toolkit](../rfc/RFC-069-graph-exploration-toolkit.md)
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Implementation paths**:
  - `web/gi-kg-viewer/src/components/graph/GraphCanvas.vue`
  - `web/gi-kg-viewer/src/components/graph/GraphNodeRailPanel.vue`
  - `web/gi-kg-viewer/src/components/graph/NodeDetail.vue`
  - `web/gi-kg-viewer/src/components/graph/GraphConnectionsSection.vue`
  - `web/gi-kg-viewer/src/components/graph/GraphNeighborhoodMiniMap.vue`
  - `web/gi-kg-viewer/src/components/explore/ExplorePanel.vue`
  - `web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts`
  - `web/gi-kg-viewer/src/stores/graphExplorer.ts`
  - `web/gi-kg-viewer/src/stores/graphFilters.ts`
  - `web/gi-kg-viewer/src/stores/graphNavigation.ts`

---

## Summary

The Graph tab provides a Cytoscape-powered interactive graph canvas for exploring
merged GI/KG artifacts. This UXS defines the visual contract for graph chrome:
toolbar, canvas overlays, minimap, degree filter, node detail rail, and
neighborhood visualization. All tokens reference
[UXS-001](UXS-001-gi-kg-viewer.md).

---

## Toolbar (primary row)

A short hint for **Shift+dbl-click** (1-hop / neighborhood) and **Shift+drag** box
zoom (and search highlight chip when applicable).

---

## Canvas overlay (bottom-right)

**Fit** (`primary`), zoom **-** / **+** / **100%** (100% = `zoom(1)`, pan unchanged),
and **Export PNG** (rightmost, 2x full graph) in one `toolbar` with `aria-label`
"Graph fit, zoom, and export" (`role="toolbar"`); aria **Zoom out** / **Zoom in** on
**-** / **+**. Sits above the graph surface (`.graph-canvas`), not in the top chrome
row.

---

## Canvas overlay (upper-right)

`role="region"` `aria-label` "Graph layout, re-layout, and degree filter"
(`.graph-layout-controls`) -- tight vertical column (~6.75rem wide): full-width
**Re-layout**, **Layout** label stacked above select (cose, breadthfirst, circle,
grid; "Graph layout algorithm" combobox), **Degree** buckets in a 2-column grid,
compact **Clear** (degree) with `aria-label` "Clear degree filter" when active.
Node detail lives in the App right rail, not over the canvas.

---

## Toolbar (chrome below primary)

Optional **Sources** row first (merged GI / KG, **Hide ungrounded**, **filters
active** when relevant); **Minimap** checkbox row; then **Edges** and **Types**
(per-type checkboxes + all / none, swatches match node fills, counts). No separate
panel above the graph.

---

## Node detail (right rail)

Graph node title uses up to two lines (`line-clamp`) plus a native `title` with the
uncapped primary label when longer than the graph short label; a body paragraph is
omitted when it would only repeat that full title. `entity_kind: episode` is not
shown as a subtitle under Insight (and similar) nodes -- the rail header already gives
the graph type. Graph node detail in-rail has no close button; type avatars reuse
graph fill/border colors.

---

## Minimap

Fixed footprint (~7.5rem tall x ~10.5rem wide, capped vs short viewports) in the
lower-left of the graph canvas host (same `overflow-hidden` region as the main
Cytoscape surface), not a viewport-fixed tile and not over the app's right rail.

---

## Density

Use existing `text-[10px]` / `border-border` patterns so the extra row stays compact
and scannable; minimap is a fixed footprint in the lower-left of the graph canvas host.

---

## E2E contract

[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) --
graph shell row.

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-10 | Initial content (in UXS-001)                                   |
| 2026-04-13 | Extracted from UXS-001 into standalone UXS-004                 |
