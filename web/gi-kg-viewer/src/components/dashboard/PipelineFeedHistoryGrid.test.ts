// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import type {
  CorpusRunFeedOutcomeItem,
  CorpusRunSummaryItem,
} from '../../api/corpusMetricsApi'
import PipelineFeedHistoryGrid from './PipelineFeedHistoryGrid.vue'

/** Compact run-item factory. */
function run(
  createdAt: string,
  byFeed: CorpusRunFeedOutcomeItem[] | null,
): CorpusRunSummaryItem {
  return {
    relative_path: `runs/${createdAt}/run.json`,
    run_id: createdAt,
    created_at: createdAt,
    run_duration_seconds: 60,
    episodes_scraped_total: 5,
    errors_total: 0,
    gi_artifacts_generated: null,
    kg_artifacts_generated: null,
    time_scraping_seconds: null,
    time_parsing_seconds: null,
    time_normalizing_seconds: null,
    time_io_and_waiting_seconds: null,
    episode_outcomes: {},
    ads_filtered_count: null,
    dialogue_insights_dropped_count: null,
    topics_normalized_count: null,
    entity_kinds_repaired_count: null,
    ad_chars_excised_preroll: null,
    ad_chars_excised_postroll: null,
    ad_episodes_with_excision_count: null,
    by_feed: byFeed,
  }
}

function feed(
  id: string,
  title: string,
  status: CorpusRunFeedOutcomeItem['status'],
): CorpusRunFeedOutcomeItem {
  return { feed_id: id, display_title: title, status, ok: 1, failed: 0, skipped: 0 }
}

const TESTID = '[data-testid="pipeline-feed-history-grid"]'

describe('PipelineFeedHistoryGrid', () => {
  it('silently hides when no run carries by_feed data (backend field absent)', () => {
    const w = mount(PipelineFeedHistoryGrid, {
      props: {
        runs: [run('2026-07-01', null), run('2026-07-02', null)],
      },
    })
    expect(w.find(TESTID).exists()).toBe(false)
  })

  it('silently hides for single-feed corpora even with by_feed data (UXS-006 §6.5)', () => {
    const w = mount(PipelineFeedHistoryGrid, {
      props: {
        runs: [
          run('2026-07-01', [feed('f1', 'Only feed', 'succeeded')]),
          run('2026-07-02', [feed('f1', 'Only feed', 'succeeded')]),
        ],
      },
    })
    expect(w.find(TESTID).exists()).toBe(false)
  })

  it('renders the grid with the last five runs when multi-feed and by_feed present', () => {
    const feeds = ['a', 'b'].map((k) => feed(k, `Feed ${k}`, 'succeeded'))
    const runs = ['2026-07-01', '2026-07-02', '2026-07-03', '2026-07-04', '2026-07-05', '2026-07-06'].map(
      (d) => run(d, feeds),
    )
    const w = mount(PipelineFeedHistoryGrid, { props: { runs } })
    const grid = w.find(TESTID)
    expect(grid.exists()).toBe(true)
    expect(grid.findAll('[data-testid="pipeline-feed-history-cell"]').length).toBe(10)
    // Oldest run in the visible window is the 2nd overall (the 1st is dropped by the -5 slice).
    const headers = grid.findAll('thead th').map((h) => h.text().trim())
    expect(headers).toEqual(['', '2026-07-02', '2026-07-03', '2026-07-04', '2026-07-05', '2026-07-06'])
  })

  it('flags the worst feed in the insight line when a feed fails ≥2 of the window', () => {
    const runs = [
      run('2026-07-01', [feed('a', 'Alpha', 'succeeded'), feed('b', 'Beta', 'failed')]),
      run('2026-07-02', [feed('a', 'Alpha', 'succeeded'), feed('b', 'Beta', 'failed')]),
      run('2026-07-03', [feed('a', 'Alpha', 'succeeded'), feed('b', 'Beta', 'succeeded')]),
    ]
    const w = mount(PipelineFeedHistoryGrid, { props: { runs } })
    const insight = w.find('[data-testid="pipeline-feed-history-grid-insight"]')
    expect(insight.text()).toBe('Feed Beta failed 2 of last 3 runs')
  })

  it('reports the clean-window insight when every cell in the window succeeded', () => {
    const feeds = [feed('a', 'Alpha', 'succeeded'), feed('b', 'Beta', 'succeeded')]
    const runs = ['2026-07-01', '2026-07-02'].map((d) => run(d, feeds))
    const w = mount(PipelineFeedHistoryGrid, { props: { runs } })
    const insight = w.find('[data-testid="pipeline-feed-history-grid-insight"]')
    expect(insight.text()).toBe('All feeds succeeded in last 2 runs')
  })
})
