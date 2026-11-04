// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import PipelineAdExcisionMetrics from './PipelineAdExcisionMetrics.vue'

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

describe('PipelineAdExcisionMetrics (mounted)', () => {
  it('shows the empty message when there is no run', () => {
    const w = mount(PipelineAdExcisionMetrics, { props: { run: null } })
    expect(w.text()).toContain('No run metrics available yet for this corpus.')
    expect(w.find('dl').exists()).toBe(false)
  })

  it('renders dashes and a legacy notice for a pre-#663 run', () => {
    const w = mount(PipelineAdExcisionMetrics, { props: { run: run() } })
    expect(w.text()).toContain('predates the #663')
    expect(w.text()).not.toContain('No ad regions were excised')
    expect(w.get('[data-testid="pipeline-ad-excision-preroll"]').text()).toContain('—')
  })

  it('says nothing was excised when all counters are zero', () => {
    const w = mount(PipelineAdExcisionMetrics, {
      props: {
        run: run({
          ad_chars_excised_preroll: 0,
          ad_chars_excised_postroll: 0,
          ad_episodes_with_excision_count: 0,
        }),
      },
    })
    expect(w.text()).toContain('No ad regions were excised on this run.')
    expect(w.findAll('span.rounded-full')).toHaveLength(0)
  })

  it('formats counts and badges only non-zero rows', () => {
    const w = mount(PipelineAdExcisionMetrics, {
      props: {
        run: run({
          ad_chars_excised_preroll: 12000,
          ad_chars_excised_postroll: 0,
          ad_episodes_with_excision_count: 3,
        }),
      },
    })
    expect(w.get('[data-testid="pipeline-ad-excision-preroll"]').text()).toContain('12,000')
    expect(w.get('[data-testid="pipeline-ad-excision-episodes"]').text()).toContain('3')
    expect(w.findAll('span.rounded-full').map((b) => b.text()).sort()).toEqual([
      'episodes',
      'preroll',
    ])
    expect(w.text()).not.toContain('No ad regions were excised')
  })
})
