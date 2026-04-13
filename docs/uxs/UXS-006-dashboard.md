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

The **Dashboard** tab provides Chart.js panels grouped under Pipeline vs Content
intelligence. This UXS defines the visual contract for the dashboard layout, chart
composition, and the API / Data left panel. All tokens reference
[UXS-001](UXS-001-gi-kg-viewer.md).

---

## Dashboard tab (charts)

- **Layout:** Corpus summary counts strip (when API + path), then an in-dashboard
  tablist (Dashboard sections): **Pipeline** (runs, manifest throughput, cumulative
  growth from `run.json`, latest run stage times + episode outcomes) vs **Content
  intelligence** (Vector index and digest glance region, then GI mtime line,
  publish-month catalog bars with gap insight, GI vs KG cumulative-by-write-day,
  graph node-type and vector index doc-type bars).
- **Copy:** Short blurb points to API / Data for corpus root, catalog snapshot, graph
  metrics, and index tooling; a one-line hint under the tabs explains the active
  section; optional "Loading corpus charts..." while dashboard fetches aggregate APIs.
- Multi-series lines use end-of-line labels instead of legends; optional insight line
  under each chart when the data supports a clear takeaway.
- Full vector index actions remain in API / Data -> Data (elevated cards).

---

## API / Data (left panel)

### API section

Neutral section title + muted blurb, then one elevated card: **Health** (label +
value, e.g. OK from `/api/health`), then Yes/No rows for Artifacts (graph), Semantic
search, Graph explore, Index routes, Corpus metrics, Library API, Digest API,
Binary (covers) -- from the same health JSON (omit a flag on older servers -> treated
as advertised except catalog flags). **Retry health**; when health fails, the offline
Choose files affordance stays here.

### Data section

Neutral section title + blurb; then sibling elevated cards (same depth as the API
card):

- **Corpus root** (Path / Resolved)
- **Corpus catalog** (snapshot from `GET /api/corpus/stats` -- feeds, episodes,
  digest topic bands, publish-month histogram rollups, optional GI/KG list counts
  when the artifact list is loaded) + Refresh
- **Graph** (merged GI/KG node/edge metrics from the loaded graph) + Refresh
  (re-list `GET /api/artifacts` and reload selected GI/KG JSON)
- **Vector index** (`GET /api/index/stats` + rebuild actions)

Metric tables use the shared MetricsPanel pattern inside cards where a titled
sub-block helps (e.g. "Index statistics" under Vector index).

### Density

Same `surface` / `border` cards; sidebar uses slightly smaller type (`text-[10px]` /
`text-xs`) so the panel stays scannable at `w-72`.

### Intent colors

**Reindex recommended** uses warning panels; informational notes use muted framing;
**Last rebuild error** uses danger text (consistent with shell errors).

### Actions

Corpus catalog and Graph each have Refresh (independent of Vector index).
**Update index** and **Full rebuild** sit next to Refresh on Vector index; disabled
while `rebuild_in_progress` or `faiss_unavailable`.

---

## E2E contract

[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) --
API / Data tab, Data heading, corpus + graph + vector index controls.

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-06 | Initial content (in UXS-001)                                   |
| 2026-04-13 | Extracted from UXS-001 into standalone UXS-006                 |
