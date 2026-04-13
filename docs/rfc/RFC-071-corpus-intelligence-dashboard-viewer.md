# RFC-071: Corpus Intelligence Dashboard (GI/KG Viewer)

- **Status**: Completed (v2.6.0) — **retrospective** RFC; **Dashboard** tab and **`/api/corpus/*`**
  metrics routes shipped before this document was added.
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Maintainers, operators using **`podcast serve`** + viewer, Playwright E2E owners
- **Related PRDs**:
  - [PRD-025: Corpus intelligence dashboard (viewer)](../prd/PRD-025-corpus-intelligence-dashboard-viewer.md)
  - [PRD-016: Operational observability](../prd/PRD-016-operational-observability-pipeline-intelligence.md) —
    **adjacent** (CI/GitHub/monitor/profiles); this RFC is **viewer-local** corpus analytics only
  - [PRD-017](../prd/PRD-017-grounded-insight-layer.md), [PRD-019](../prd/PRD-019-knowledge-graph-layer.md),
    [PRD-021](../prd/PRD-021-semantic-corpus-search.md), [PRD-022](../prd/PRD-022-corpus-library-episode-browser.md),
    [PRD-023](../prd/PRD-023-corpus-digest-recap.md)
- **Related ADRs**:
  - [ADR-074: Multi-feed corpus parent layout and manifest](../adr/ADR-074-multi-feed-corpus-parent-layout-and-manifest.md) — manifest and parent-root semantics for **Pipeline** charts
  - [ADR-064: Canonical server layer](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)
  - [ADR-065: Vue 3 + Vite + Cytoscape](../adr/ADR-065-vue3-vite-cytoscape-frontend-stack.md)
  - [ADR-066: Playwright for UI E2E](../adr/ADR-066-playwright-for-ui-e2e-testing.md)
- **Related RFCs**:
  - [RFC-062: GI/KG viewer v2](RFC-062-gi-kg-viewer-v2.md) — SPA shell; **Dashboard** is a first-class
    main tab (see **Delivered scope** table there + this RFC for detail)
  - [RFC-063: Multi-feed corpus](RFC-063-multi-feed-corpus-append-resume.md) — **`corpus_manifest.json`**,
    **`run.json`** layout consumed for **Pipeline** charts
  - [RFC-061: Semantic corpus search](RFC-061-semantic-corpus-search.md) — **`/api/index/stats`** shape
  - [RFC-067](RFC-067-corpus-library-api-viewer.md) — **`GET /api/corpus/feeds`** (feeds-in-index vs catalog)
  - [RFC-068](RFC-068-corpus-digest-api-viewer.md) — **`GET /api/corpus/digest?compact=true`** glance line
- **Related UX specs**:
  - [UXS-006: Dashboard](../uxs/UXS-006-dashboard.md) — **Dashboard tab (charts)**
  - [UXS-001: GI/KG viewer](../uxs/UXS-001-gi-kg-viewer.md) — shared tokens and shell conventions
- **Related Documents**:
  - [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
  - [Server guide](../guides/SERVER_GUIDE.md) (HTTP overview)
- **Updated**: 2026-04-11 (authored retrospectively)

## Abstract

The **Dashboard** view in **`web/gi-kg-viewer`** aggregates **pipeline execution** signals (corpus
manifest, discovered **`run.json`** files, stage timings, episode outcomes) and **content
intelligence** signals (FAISS index stats, optional digest snapshot, GI/KG artifact mtimes, catalog
publish-month histogram vs list counts, loaded-graph node types vs vector **doc_type** counts). The
browser composes Chart.js panels inside two **tabpanels** — **Pipeline** and **Content intelligence** —
calling FastAPI **`corpus_metrics`** routes under **`/api/`** plus existing **index**, **digest**, and
**library** endpoints. This RFC records the **as-built** architecture and boundaries relative to
**RFC-062** (shell) and **PRD-025** (product intent).

## Problem Statement

Operators needed **corpus-scale** answers (runs, feeds, index health, artifact freshness) without
exporting data to separate BI tools or reading raw JSON trees. **API · Data** exposes the same facts as
**cards**; the **Dashboard** adds **time-series and distribution** views and ties **pipeline** vs
**content** mental models. Without a written RFC, the split between **RFC-062** (monolithic viewer RFC)
and **corpus_metrics** behavior was hard to navigate for contributors.

## Delivered architecture

### Frontend (`web/gi-kg-viewer/`)

| Piece | Role |
| ----- | ---- |
| **`DashboardView.vue`** | Fetches corpus stats, manifest, runs summary, digest glance; wires **Pinia**
  **artifacts** / **indexStats**; switches **`pipeline` / `contentIntelligence`** panels. |
| **`DashboardOverviewSection.vue`** | Summary strip + tab UI glue. |
| **Chart components** | **`CategoryLineChart`**, **`MultiSeriesLineChart`**, **`SimpleDoughnutChart`**,
  **`StackedStageBarChart`**, **`TypeCountBarChart`**, **`VerticalBarChart`**, **`MetricsPanel`**. |
| **`api/corpusMetricsApi.ts`** | **`fetchCorpusStats`**, **`fetchCorpusManifest`**, **`fetchCorpusRunsSummary`**. |
| **`api/digestApi.ts`** | Compact digest for dashboard one-liner. |
| **`utils/artifactMtimeBuckets.ts`** | Client-side GI/KG mtime bucketing (caps documented in code). |

**Behavioral** rules (refresh generation, loading flags, error handling) belong in this RFC; **visual**
density, tokens, and **aria** labels for the **Dashboard** row belong in **UXS-006** (tokens per **UXS-001**).

### Backend (`src/podcast_scraper/server/routes/corpus_metrics.py`)

Mounted under the app **`/api`** prefix:

| Method | Path | Purpose |
| ------ | ---- | ------- |
| **GET** | **`/corpus/stats`** | **`CorpusStatsResponse`** — feeds, episodes, digest topic config, publish-month
  rollups, optional list counts when catalog builder runs. |
| **GET** | **`/corpus/documents/manifest`** | Parsed **`corpus_manifest.json`** document for throughput bars. |
| **GET** | **`/corpus/documents/run-summary`** | Single-run style summary helper (when used). |
| **GET** | **`/corpus/runs/summary`** | **`CorpusRunsSummaryResponse`** — bounded scan of **`run.json`** under
  corpus root (cap **150** files in module). |

**Related routers** (not defined in `corpus_metrics.py` but consumed by the same view):

- **`GET /api/index/stats`**, **`POST /api/index/rebuild`** — **index** routes ([RFC-061](RFC-061-semantic-corpus-search.md)).
- **`GET /api/corpus/digest?compact=true`** — [RFC-068](RFC-068-corpus-digest-api-viewer.md).
- **`GET /api/corpus/feeds`** — [RFC-067](RFC-067-corpus-library-api-viewer.md) (feeds in index vs catalog bars).

### Data sources

- **Filesystem** under resolved corpus root: **`run.json`**, **`corpus_manifest.json`**, metadata trees
  ([RFC-063](RFC-063-multi-feed-corpus-append-resume.md)).
- **In-memory / client**: merged **GI/KG** artifact list from **`GET /api/artifacts`** + loaded JSON for
  graph metrics and mtime timelines (subject to client caps).

## Non-goals

- **Not** implementing new chart types or ML-based anomaly detection in this RFC’s scope.
- **Not** merging with **Streamlit** run-compare or **RFC-064** profile YAML.
- **Not** adding **Postgres** for dashboard queries (**RFC-051**).

## Testing

- **E2E**: **`web/gi-kg-viewer/e2e/dashboard.spec.ts`** (and related mocks) — see [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) **Dashboard** row.
- **Server**: extend **`tests/unit/podcast_scraper/server/`** and **`tests/integration/server/`** when
  changing **`corpus_metrics`** response shapes (existing tests may already cover stats).

## Relationship to RFC-062

**RFC-062** remains the **umbrella** viewer + server seed RFC. **RFC-071** is a **focused slice** for the
**Dashboard** product surface so PRD/RFC indexes and cross-links stay precise. Prefer editing **RFC-071**
for Dashboard-only API or chart-behavior notes; edit **RFC-062** when shell navigation, **API · Data**,
or shared stores change across tabs.

## References

- [PRD-025](../prd/PRD-025-corpus-intelligence-dashboard-viewer.md)
- [RFC-062](RFC-062-gi-kg-viewer-v2.md)
- [RFC-063](RFC-063-multi-feed-corpus-append-resume.md)
- [UXS-006](../uxs/UXS-006-dashboard.md); [UXS-001](../uxs/UXS-001-gi-kg-viewer.md)
