# GH-743 — Viewer API polling (adaptive + visibility)

This note documents client-side polling after [GitHub issue 743](https://github.com/chipi/podcast_scraper/issues/743) (adaptive intervals and pausing when the browser tab is in the background).

## Endpoints and cadence

| Source | Endpoint(s) | When it runs | Interval / behavior |
| ------ | ------------- | ------------ | ------------------- |
| `PipelineJobsCard` | `GET /api/jobs?path=…` | Corpus path set, health OK, jobs API on, and at least one job is **queued** or **running** | **Adaptive:** about **2.5s** while the job registry snapshot keeps changing; about **12s** after **3** consecutive unchanged snapshots (quiet refreshes). **Paused** when `document.hidden` (tab in background). Elapsed-time UI ticks every **1s** only while the tab is **visible**. |
| `PipelineJobExplorePanel` | `GET /api/jobs/subprocess-log-tail?…` | Same jobs API + corpus path; job status **running** or **queued** | **Adaptive:** about **3s** while the log tail fingerprint is changing; about **10s** after **3** unchanged tails. **Paused** when the tab is hidden. Initial load fetches the log tail only (Metrics / Command tabs). |
| `indexStats` store | `GET /api/index/stats` | After index rebuild POST while `rebuild_in_progress` is true (poll loop) | **Adaptive:** **2.5s** while index stats fingerprint is changing; **10s** after **4** unchanged polls. **Paused** when the tab is hidden; resumes when the tab becomes visible if rebuild is still in progress. Hard stop after **80** poll cycles. |
| `DashboardView` | `GET /api/corpus/runs/summary` (and coverage, feeds, digest, persons) | Corpus path or health changes | **No timer** — refresh on dependency change only. |
| `PipelineJobHistoryStrip` | `GET /api/jobs?path=…` | Corpus path / health / jobs API change | **No timer** — refresh on dependency change only. |

## Tests

- `web/gi-kg-viewer/src/composables/usePageVisible.test.ts` — visibility ref updates on `visibilitychange`.

## Related files

- `web/gi-kg-viewer/src/composables/usePageVisible.ts`
- `web/gi-kg-viewer/src/components/dashboard/PipelineJobsCard.vue`
- `web/gi-kg-viewer/src/components/dashboard/PipelineJobExplorePanel.vue`
- `web/gi-kg-viewer/src/stores/indexStats.ts`
