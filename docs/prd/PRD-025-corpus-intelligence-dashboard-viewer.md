# PRD-025: Corpus Intelligence Dashboard (GI/KG Viewer)

- **Status**: Implemented (v2.6.0) — documented retrospectively; behavior shipped in viewer **Dashboard**
  tab and supporting **`/api/corpus/*`** aggregates.
- **Authors**: Podcast Scraper Team
- **Related RFCs**:
  - [RFC-071: Corpus intelligence dashboard (viewer)](../rfc/RFC-071-corpus-intelligence-dashboard-viewer.md) —
    technical design and API consumption
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md) — SPA shell, **Dashboard** main tab
  - [RFC-063: Multi-feed corpus](../rfc/RFC-063-multi-feed-corpus-append-resume.md) — manifest /
    `run.json` layout feeding **Pipeline** charts
  - [RFC-061: Semantic corpus search (FAISS)](../rfc/RFC-061-semantic-corpus-search.md) — vector index
    stats in **Content intelligence**
  - [RFC-067](../rfc/RFC-067-corpus-library-api-viewer.md) / [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md) —
    catalog + digest data mixed into dashboard copy
- **Related PRDs** (adjacent product surfaces):
  - [PRD-016: Operational observability](../prd/PRD-016-operational-observability-pipeline-intelligence.md) —
    **distinct** from this PRD: PRD-016 covers CI metrics, GitHub Pages, frozen profiles (RFC-064/066),
    live monitor (RFC-065). **This** PRD is **corpus-local** analytics in the **viewer** only.
  - [PRD-017](../prd/PRD-017-grounded-insight-layer.md), [PRD-019](../prd/PRD-019-knowledge-graph-layer.md) —
    GI/KG artifacts summarized in charts
  - [PRD-021](../prd/PRD-021-semantic-corpus-search.md) — index footprint and doc-type bars
  - [PRD-022](../prd/PRD-022-corpus-library-episode-browser.md), [PRD-023](../prd/PRD-023-corpus-digest-recap.md) —
    feeds catalog + digest glance strings
- **Related UX specs**:
  - [UXS-001: GI/KG viewer](../uxs/UXS-001-gi-kg-viewer.md) — **Dashboard tab (charts)** layout, tokens,
    accessibility targets
- **Related Documents**:
  - [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
    — **Dashboard** row (Playwright contract)
- **Updated**: 2026-04-11 (retrospective PRD)

## Summary

Operators and developers need a **single in-viewer place** to see whether a corpus is healthy:
pipeline runs (manifest throughput, latest `run.json` stages, episode outcomes), and **content
intelligence** (vector index vs catalog, digest one-liner, GI/KG write-time timelines, publish-month
histograms, graph node types vs indexed doc types). The **Dashboard** main tab in **`web/gi-kg-viewer`**
delivers Chart.js-based panels grouped under **Pipeline** vs **Content intelligence**, fed by FastAPI
**`/api/corpus/*`** routes and client-side merges with the loaded graph and index stores.

## Background

Before the Dashboard tab, operators relied on **API · Data** cards, ad hoc **`cat`**, or external
tools to correlate **run.json**, **corpus_manifest.json**, catalog stats, and index health. The
Dashboard does not replace **RFC-064** frozen profiles or **RFC-066** Streamlit run compare; it answers
**“what does this corpus root look like right now?”** inside the same session as graph and search.

## Goals

1. **At-a-glance corpus summary** — feeds, episodes, digest topic bands, GI list counts when artifacts
   are listed (from **`GET /api/corpus/stats`** + client context).
2. **Pipeline visibility** — manifest document, run summaries, cumulative growth, latest run stage bars,
   episode outcome bars from **`run.json`** discovery ([RFC-063](../rfc/RFC-063-multi-feed-corpus-append-resume.md)).
3. **Content intelligence** — vector index glance (**`GET /api/index/stats`**), optional compact digest
   (**`GET /api/corpus/digest?compact=true`**), GI/KG mtime timelines (client-bucketed), publish-month
   catalog vs histogram insight, graph node-type vs index doc-type bars.
4. **Trust and navigation** — blurbs point operators to **API · Data** for corrective actions (reindex,
   refresh catalog).
5. **Testable contract** — Playwright **`dashboard.spec.ts`** and [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md).

## Non-goals

- **Not** a replacement for **Streamlit** run comparison ([RFC-047](../rfc/RFC-047-run-comparison-visual-tool.md),
  [RFC-066](../rfc/RFC-066-run-compare-performance-tab.md)) or **frozen release profiles** ([RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md)).
- **Not** real-time pipeline monitoring during a run — that is **[RFC-065](../rfc/RFC-065-live-pipeline-monitor.md)**
  (`--monitor`).
- **Not** arbitrary SQL or Postgres — **[RFC-051](../rfc/RFC-051-database-projection-gil-kg.md)** remains
  separate.
- **Not** natural-language dashboard queries; structured charts and labels only.

## User-facing requirements

| Area | Requirement |
| ---- | ----------- |
| **Entry** | **Dashboard** appears in **Main views** navigation with **Digest**, **Library**, **Graph**. |
| **Summary strip** | When API + corpus path healthy, show **Corpus summary counts** (`role="group"`,
  `aria-label="Corpus summary counts"`). |
| **Sections** | **Dashboard sections** tablist: **Pipeline** vs **Content intelligence**; one hint line
  under tabs. |
| **Pipeline charts** | Manifest-related bars, run duration, cumulative growth, latest-run stage stacked
  bars, episode-outcome horizontal bars when data exists. |
| **Content intelligence** | **Vector index and digest glance** region; GI+KG timelines (subject to client
  caps); publish-month bars + catalog vs bar-sum insight; node-type and doc-type bars with optional
  **% of vectors**. |
| **Loading / errors** | Optional loading copy; errors surfaced without breaking the shell. |
| **Visual contract** | [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) **Dashboard tab (charts)**. |

## Success criteria

1. With a healthy server and corpus path, an operator can open **Dashboard** and see **Pipeline** and
   **Content intelligence** without loading the graph canvas.
2. Charts use the same **corpus root** as **API · Data** and respect multi-feed layout where applicable.
3. **`make test-ui-e2e`** covers **Dashboard** surfaces per E2E map.
4. Documentation chain: **PRD-025** (this) → **RFC-071** → **UXS-001** § Dashboard.

## References

- [RFC-071](../rfc/RFC-071-corpus-intelligence-dashboard-viewer.md)
- [UXS-001](../uxs/UXS-001-gi-kg-viewer.md)
- [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md)
