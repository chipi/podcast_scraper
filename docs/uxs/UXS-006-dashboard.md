# UXS-006: Dashboard

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Product brief**: [DASHBOARD-SPEC.md](../wip/DASHBOARD-SPEC.md) (authoritative layout and tab content)
- **Related PRDs**:
  - [PRD-025: Corpus Intelligence Dashboard](../prd/PRD-025-corpus-intelligence-dashboard-viewer.md)
- **Related RFCs**:
  - [RFC-071: Corpus Intelligence Dashboard](../rfc/RFC-071-corpus-intelligence-dashboard-viewer.md)
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Implementation paths**:
  - `web/gi-kg-viewer/src/components/dashboard/DashboardView.vue`
  - `web/gi-kg-viewer/src/components/dashboard/BriefingCard.vue`
  - `web/gi-kg-viewer/src/components/dashboard/IndexStatusCard.vue`
  - `web/gi-kg-viewer/src/components/dashboard/TopicClustersStatusBlock.vue`
  - `web/gi-kg-viewer/src/utils/chartRegister.ts`
  - `web/gi-kg-viewer/src/stores/indexStats.ts`, `web/gi-kg-viewer/src/stores/dashboardNav.ts`

---

## Summary

The **Dashboard** main tab is **briefing + three tabs only**: **Coverage** (default),
**Intelligence**, **Pipeline**. Corpus artifact picking (**List**, **All** / **None**,
**Load into graph**) lives on the **status bar** (**List** opens
`data-testid="artifact-list-dialog"`). Deep links to **Library**, **Digest**, and
**Graph** use `dashboardNav` handoffs consumed when those tabs activate. Chart.js uses
shared Tufte-style defaults from `chartRegister.ts`.

---

## Layout

1. **Briefing card** (`data-testid="briefing-card"`) — last run / health / short actions; always above tabs.
2. **Tablist** `aria-label="Dashboard tabs"` — **Coverage** | **Intelligence** | **Pipeline**.
3. **Coverage** — coverage by month, feed coverage table, artifact activity (from listed artifacts), **Index status** (`data-testid="index-status-card"`) with **Update index** and **Full rebuild** (`index-status-update`, `index-status-full-rebuild`).
4. **Intelligence** — digest snapshot, **Topic clusters** status (`topic-clusters-status-block`), topic landscape, top voices (when API available), degraded stubs for momentum / emerging connections until RFC data exists.
5. **Pipeline** — run history strip, duration trend, stage timings, numeric outcomes, episodes per run; optional per-feed run heatmap only when server exposes stable per-feed fields.

The legacy **CorpusDataWorkspace** / **Pipeline | Content intelligence** split on Dashboard is removed; those components are not part of this surface.

---

## Corpus artifacts (status bar)

**List** (when health is ok and a corpus path is set) fetches `GET /api/artifacts` and opens the **Corpus artifacts** dialog. **Load into graph** switches the main tab to **Graph** when load succeeds (same behavior as the former workspace).

---

## E2E contract

[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) — **Dashboard** tab,
`briefing-card`, **Dashboard tabs** tablist, **Index status** card, **`artifact-list-dialog`**;
`openCorpusDataWorkspace` only switches to **Dashboard** and waits for the briefing card.

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-06 | Initial content (in UXS-001)                                   |
| 2026-04-13 | Extracted from UXS-001 into standalone UXS-006                 |
| 2026-04-19 | Corpus workspace on Dashboard; left rail query-only IA         |
| 2026-04-19 | Dashboard: briefing + tabs; artifacts via status bar dialog    |
