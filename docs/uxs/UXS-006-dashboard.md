# UXS-006: Dashboard

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-025: Corpus Intelligence Dashboard](../prd/PRD-025-corpus-intelligence-dashboard-viewer.md)
- **Related RFCs**:
  - [RFC-071: Corpus Intelligence Dashboard](../rfc/RFC-071-corpus-intelligence-dashboard-viewer.md)
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Implementation paths**:
  - `web/gi-kg-viewer/src/components/dashboard/DashboardView.vue`
  - `web/gi-kg-viewer/src/components/dashboard/CorpusDataWorkspace.vue`
  - `web/gi-kg-viewer/src/components/dashboard/DashboardOverviewSection.vue`
  - `web/gi-kg-viewer/src/components/dashboard/MetricsPanel.vue`
  - `web/gi-kg-viewer/src/components/dashboard/CategoryLineChart.vue`
  - `web/gi-kg-viewer/src/components/dashboard/MultiSeriesLineChart.vue`
  - `web/gi-kg-viewer/src/components/dashboard/SimpleDoughnutChart.vue`
  - `web/gi-kg-viewer/src/components/dashboard/StackedStageBarChart.vue`
  - `web/gi-kg-viewer/src/components/dashboard/TypeCountBarChart.vue`
  - `web/gi-kg-viewer/src/components/dashboard/VerticalBarChart.vue`
  - `web/gi-kg-viewer/src/stores/indexStats.ts`

---

## Summary

The **Dashboard** tab provides the **Corpus artifacts** workspace (API capability card,
artifact list, **Load into graph**), then Chart.js panels grouped under **Pipeline**
vs **Content intelligence**. The left shell column stays **query-only** (Search +
Explore); corpus root, health, catalog, graph metrics, topic clusters, and vector
index tooling live on **Dashboard** inside **`CorpusDataWorkspace`** (`data-testid="corpus-data-workspace"`).
All tokens reference [UXS-001](UXS-001-gi-kg-viewer.md).

---

## Dashboard tab (charts)

- **Layout:** **`CorpusDataWorkspace`** first (full main-column width, scrolls with the
  tab): **Corpus artifacts** (List, All / None, Load into graph, hints), **API** health
  card, **`DashboardOverviewSection`** (**Data** heading: Corpus root, Corpus catalog,
  Graph, Vector index, topic clusters status). Then corpus summary counts strip (when
  API + path), then an in-dashboard tablist (**Dashboard sections**): **Pipeline** vs
  **Content intelligence** (charts as before).
- **Copy:** Intro blurb points to the **Corpus artifacts** workspace for corpus root,
  catalog snapshot, graph metrics, and index tooling; one-line hint under the section
  tabs explains the active chart panel; optional **Loading corpus charts...** while
  dashboard fetches aggregate APIs.
- Multi-series lines use end-of-line labels instead of legends; optional insight line
  under each chart when the data supports a clear takeaway.
- Full vector index actions remain on the **Vector index** card inside **Data** (same
  elevated-card pattern as before).

---

## Corpus data workspace (Dashboard)

### Corpus artifacts block

Same controls as the former left-rail data tab: **List**, **All** / **None**,
**Load into graph**, checkbox list, corpus path hints, load errors. **Load into graph**
switches the main view to **Graph** when a merged artifact is ready (emit from
`CorpusDataWorkspace`).

### API section

Neutral **API** heading + muted blurb (`GET /api/health` capability overview), then one
elevated card: **Health** (label + value), then Yes/No rows for Artifacts (graph),
Semantic search, Graph explore, Index routes, Corpus metrics, Library API, Digest API,
Binary (covers). **Retry health**; when health fails, offline copy points to **Files**
on the **status bar** (not inside this card).

### Data section

Rendered by **`DashboardOverviewSection`**: **Data** `h2`, blurb, **Topic clusters**
status, then elevated cards — **Corpus root**, **Corpus catalog** (+ Refresh), **Graph**
(+ Refresh), **Vector index** (`GET /api/index/stats` + **Update index** / **Full rebuild**).

Metric tables use the shared **MetricsPanel** pattern inside cards where a titled
sub-block helps (e.g. **Index statistics** under Vector index).

### Density

Same `surface` / `border` cards; workspace typography uses slightly smaller type
(`text-[10px]` / `text-xs`) so the block stays scannable inside the scrollable Dashboard
column.

### Intent colors

**Reindex recommended** uses warning panels; informational notes use muted framing;
**Last rebuild error** uses danger text (consistent with shell errors).

### Actions

Corpus catalog and Graph each have **Refresh** (independent of Vector index).
**Update index** and **Full rebuild** sit next to Refresh on Vector index; disabled
while `rebuild_in_progress` or `faiss_unavailable`.

---

## E2E contract

[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) --
**Dashboard** tab, **`data-testid="corpus-data-workspace"`**, **Data** heading, corpus +
graph + vector index controls (`openCorpusDataWorkspace` helper).

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-06 | Initial content (in UXS-001)                                   |
| 2026-04-13 | Extracted from UXS-001 into standalone UXS-006                 |
| 2026-04-19 | Corpus workspace on Dashboard; left rail query-only IA         |
