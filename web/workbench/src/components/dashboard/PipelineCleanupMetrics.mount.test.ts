// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import PipelineCleanupMetrics from './PipelineCleanupMetrics.vue'

function run(over: Partial<CorpusRunSummaryItem> = {}): CorpusRunSummaryItem {
  return {
    relative_path: 'r',
    run_id: 'r1',
    created_at: null,
    run_duration_seconds: null,
    episodes_scraped_total: null,
    errors_total: null,
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
    ...over,
  }
}

describe('PipelineCleanupMetrics (mounted)', () => {
  it('shows the empty message when there is no run', () => {
    const w = mount(PipelineCleanupMetrics, { props: { run: null } })
    expect(w.text()).toContain('No run metrics available yet for this corpus.')
    expect(w.find('dl').exists()).toBe(false)
  })

  it('renders dashes and a legacy notice for a pre-#652 run', () => {
    const w = mount(PipelineCleanupMetrics, { props: { run: run() } })
    expect(w.text()).toContain('predates the #652')
    expect(w.text()).not.toContain('No cleanup filters fired')
    const ads = w.get('[data-testid="pipeline-cleanup-ads"]')
    expect(ads.text()).toContain('Ads filtered')
    expect(ads.text()).toContain('—')
  })

  it('says no filters fired when every counter is zero, with no badges', () => {
    const w = mount(PipelineCleanupMetrics, {
      props: {
        run: run({
          ads_filtered_count: 0,
          dialogue_insights_dropped_count: 0,
          topics_normalized_count: 0,
          entity_kinds_repaired_count: 0,
        }),
      },
    })
    expect(w.text()).toContain('No cleanup filters fired on this run.')
    expect(w.text()).not.toContain('predates')
    expect(w.get('[data-testid="pipeline-cleanup-ads"]').text()).not.toContain('ads ')
    expect(w.findAll('span.rounded-full')).toHaveLength(0)
  })

  it('formats counts with separators and badges only the non-zero rows', () => {
    const w = mount(PipelineCleanupMetrics, {
      props: {
        run: run({
          ads_filtered_count: 1234,
          dialogue_insights_dropped_count: 0,
          topics_normalized_count: 7,
          entity_kinds_repaired_count: null,
        }),
      },
    })
    expect(w.get('[data-testid="pipeline-cleanup-ads"]').text()).toContain('1,234')
    expect(w.get('[data-testid="pipeline-cleanup-topics"]').text()).toContain('7')
    const badges = w.findAll('span.rounded-full').map((b) => b.text())
    expect(badges.sort()).toEqual(['ads', 'topics'])
    expect(w.text()).not.toContain('No cleanup filters fired')
    expect(w.text()).not.toContain('predates')
  })
})
