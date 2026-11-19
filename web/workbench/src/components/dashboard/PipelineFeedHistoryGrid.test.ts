// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import type { CorpusFeedItem } from '../../api/corpusLibraryApi'
import PipelineFeedHistoryGrid from './PipelineFeedHistoryGrid.vue'

/** Compact per-(feed, run) item factory. Mirrors the backend layout. */
function item(
  feedId: string | null,
  createdAt: string,
  outcomes: { ok?: number; failed?: number; skipped?: number } = {},
): CorpusRunSummaryItem {
  return {
    relative_path: `feeds/${feedId ?? 'none'}/${createdAt}/run.json`,
    run_id: `${feedId ?? 'none'}-${createdAt}`,
    created_at: createdAt,
    run_duration_seconds: 60,
    episodes_scraped_total: (outcomes.ok ?? 0) + (outcomes.failed ?? 0),
    errors_total: outcomes.failed ?? 0,
    gi_artifacts_generated: null,
    kg_artifacts_generated: null,
    time_scraping_seconds: null,
    time_parsing_seconds: null,
    time_normalizing_seconds: null,
    time_io_and_waiting_seconds: null,
    episode_outcomes: {
      ok: outcomes.ok ?? 0,
      failed: outcomes.failed ?? 0,
      skipped: outcomes.skipped ?? 0,
    },
    ads_filtered_count: null,
    dialogue_insights_dropped_count: null,
    topics_normalized_count: null,
    entity_kinds_repaired_count: null,
    ad_chars_excised_preroll: null,
    ad_chars_excised_postroll: null,
    ad_episodes_with_excision_count: null,
    feed_id: feedId,
  }
}

function feedCatalog(entries: [string, string][]): CorpusFeedItem[] {
  return entries.map(([feed_id, display_title]) => ({
    feed_id,
    display_title,
    episode_count: 0,
    rss_url: null,
    description: null,
  }))
}

const TESTID = '[data-testid="pipeline-feed-history-grid"]'

describe('PipelineFeedHistoryGrid', () => {
  it('silently hides when no run carries feed_id (legacy flat layout)', () => {
    const w = mount(PipelineFeedHistoryGrid, {
      props: {
        runs: [
          item(null, '2026-07-01', { ok: 3 }),
          item(null, '2026-07-02', { ok: 3 }),
        ],
      },
    })
    expect(w.find(TESTID).exists()).toBe(false)
  })

  it('silently hides for single-feed corpora (UXS-006 §6.5)', () => {
    const w = mount(PipelineFeedHistoryGrid, {
      props: {
        runs: [
          item('rss_alpha', '2026-07-01', { ok: 3 }),
          item('rss_alpha', '2026-07-02', { ok: 3 }),
        ],
      },
    })
    expect(w.find(TESTID).exists()).toBe(false)
  })

  it('renders the grid capped at the last five distinct run-days when multi-feed', () => {
    const days = ['2026-07-01', '2026-07-02', '2026-07-03', '2026-07-04', '2026-07-05', '2026-07-06']
    const runs = days.flatMap((d) => [
      item('rss_alpha', d, { ok: 3 }),
      item('rss_beta', d, { ok: 3 }),
    ])
    const feeds = feedCatalog([
      ['rss_alpha', 'Alpha Feed'],
      ['rss_beta', 'Beta Feed'],
    ])
    const w = mount(PipelineFeedHistoryGrid, { props: { runs, feeds } })
    const grid = w.find(TESTID)
    expect(grid.exists()).toBe(true)
    // 2 feeds × 5 days = 10 cells; oldest day dropped by the -5 slice.
    expect(grid.findAll('[data-testid="pipeline-feed-history-cell"]').length).toBe(10)
    const headers = grid.findAll('thead th').map((h) => h.text().trim())
    expect(headers).toEqual(['', '2026-07-02', '2026-07-03', '2026-07-04', '2026-07-05', '2026-07-06'])
    // Row headers use the catalog display_title, not the raw feed_id.
    const rowHeaders = grid.findAll('tbody th').map((h) => h.text().trim())
    expect(rowHeaders).toEqual(['Alpha Feed', 'Beta Feed'])
  })

  it('flags the worst feed in the insight line when a feed fails ≥2 of the window', () => {
    const runs = [
      item('rss_alpha', '2026-07-01', { ok: 3 }),
      item('rss_beta', '2026-07-01', { failed: 3 }),
      item('rss_alpha', '2026-07-02', { ok: 3 }),
      item('rss_beta', '2026-07-02', { failed: 3 }),
      item('rss_alpha', '2026-07-03', { ok: 3 }),
      item('rss_beta', '2026-07-03', { ok: 3 }),
    ]
    const feeds = feedCatalog([
      ['rss_alpha', 'Alpha'],
      ['rss_beta', 'Beta'],
    ])
    const w = mount(PipelineFeedHistoryGrid, { props: { runs, feeds } })
    const insight = w.find('[data-testid="pipeline-feed-history-grid-insight"]')
    expect(insight.text()).toBe('Feed Beta failed 2 of last 3 runs')
  })

  it('reports the clean-window insight when every cell in the window succeeded', () => {
    const runs = ['2026-07-01', '2026-07-02'].flatMap((d) => [
      item('rss_alpha', d, { ok: 3 }),
      item('rss_beta', d, { ok: 3 }),
    ])
    const feeds = feedCatalog([
      ['rss_alpha', 'Alpha'],
      ['rss_beta', 'Beta'],
    ])
    const w = mount(PipelineFeedHistoryGrid, { props: { runs, feeds } })
    const insight = w.find('[data-testid="pipeline-feed-history-grid-insight"]')
    expect(insight.text()).toBe('All feeds succeeded in last 2 runs')
  })

  it('falls back to feed_id in row header when no catalog entry matches', () => {
    const runs = ['2026-07-01', '2026-07-02'].flatMap((d) => [
      item('rss_alpha', d, { ok: 3 }),
      item('rss_beta', d, { ok: 3 }),
    ])
    const w = mount(PipelineFeedHistoryGrid, { props: { runs } })
    const rowHeaders = w.findAll('tbody th').map((h) => h.text().trim())
    expect(rowHeaders).toEqual(['rss_alpha', 'rss_beta'])
  })
})
