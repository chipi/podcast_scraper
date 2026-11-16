// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CorpusCoverageResponse } from '../../api/corpusCoverageApi'
import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import type { IndexStatsEnvelope } from '../../api/indexStatsApi'
import { briefingCoverage } from '../../extensions'
import { useDashboardNavStore } from '../../stores/dashboardNav'
import BriefingCard from './BriefingCard.vue'

const NOW = Date.parse('2026-06-15T12:00:00Z')

function run(over: Partial<CorpusRunSummaryItem> = {}): CorpusRunSummaryItem {
  return {
    relative_path: 'feeds/a/run_1/run.json',
    run_id: 'r1',
    created_at: '2026-06-15T10:00:00Z',
    run_duration_seconds: 204,
    episodes_scraped_total: 1500,
    errors_total: 0,
    gi_artifacts_generated: null,
    kg_artifacts_generated: null,
    time_scraping_seconds: null,
    time_parsing_seconds: null,
    time_normalizing_seconds: null,
    time_io_and_waiting_seconds: null,
    episode_outcomes: { ok: 10 },
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

function coverage(over: Partial<CorpusCoverageResponse> = {}): CorpusCoverageResponse {
  return {
    path: '/c',
    total_episodes: 100,
    with_gi: 90,
    with_kg: 90,
    with_both: 90,
    with_neither: 0,
    by_month: [],
    by_feed: [],
    ...over,
  }
}

function indexEnv(over: Partial<IndexStatsEnvelope> = {}, vectors = 5000): IndexStatsEnvelope {
  return {
    available: true,
    stats: {
      total_vectors: vectors,
      doc_type_counts: {},
      feeds_indexed: ['a', 'b'],
      embedding_model: 'm',
      embedding_dim: 3,
      last_updated: '2026-06-14T12:00:00Z',
      index_size_bytes: 1,
    },
    ...over,
  }
}

type Props = InstanceType<typeof BriefingCard>['$props']

function mountCard(over: Partial<Props> = {}) {
  return mount(BriefingCard, {
    props: {
      runs: [run()],
      coverage: coverage(),
      indexEnv: indexEnv(),
      catalogFeedCount: 2,
      corpusPath: '/corpus',
      apiReady: true,
      ...over,
    } as Props,
  })
}

function actionTexts(w: ReturnType<typeof mountCard>): string[] {
  return w.findAll('[data-testid="briefing-action-item"]').map((li) => li.find('span').text())
}

describe('BriefingCard', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  describe('no corpus / not ready / no runs', () => {
    it('prompts for a corpus path and renders nothing else', () => {
      const w = mountCard({ corpusPath: '   ' })
      expect(w.get('[data-testid="briefing-no-corpus"]').text()).toContain('Set a corpus path')
      expect(w.find('[data-testid="briefing-last-run"]').exists()).toBe(false)
    })

    it('says it is waiting for the API', () => {
      const w = mountCard({ apiReady: false })
      expect(w.get('[data-testid="briefing-last-run"]').text()).toContain('Waiting for API…')
    })

    it('explains how to start when there are no runs', () => {
      const w = mountCard({ runs: [] })
      expect(w.get('[data-testid="briefing-last-run"]').text()).toContain('No pipeline runs found')
    })
  })

  describe('last run summary', () => {
    it('shows success, episodes, duration and age', () => {
      const w = mountCard()
      const t = w.get('[data-testid="briefing-last-run"]').text()
      expect(t).toContain('Success')
      expect(t).toContain(`${(1500).toLocaleString()} episodes`)
      expect(t).toContain('3m 24s')
      expect(t).toContain('2 hours ago')
    })

    it('shows dashes for missing duration and timestamp', () => {
      const w = mountCard({
        runs: [run({ run_duration_seconds: null, created_at: null, episodes_scraped_total: null })],
      })
      const t = w.get('[data-testid="briefing-last-run"]').text()
      expect(t).toContain('0 episodes')
      expect(t.match(/—/g)).toHaveLength(2)
    })

    it('classifies a mix of ok and failed episodes as Partial (warning dot)', () => {
      const w = mountCard({ runs: [run({ episode_outcomes: { ok: 5, failed: 2 } })] })
      const t = w.get('[data-testid="briefing-last-run"]')
      expect(t.text()).toContain('Partial')
      expect(t.find('.text-warning').exists()).toBe(true)
    })

    it('classifies all-failed episodes as Failed (danger dot)', () => {
      const w = mountCard({ runs: [run({ episode_outcomes: { failed: 3 } })] })
      const t = w.get('[data-testid="briefing-last-run"]')
      expect(t.text()).toContain('Failed')
      expect(t.find('.text-danger').exists()).toBe(true)
    })

    it('classifies zero outcomes with errors as Failed, and without errors as Success', () => {
      expect(
        mountCard({ runs: [run({ episode_outcomes: {}, errors_total: 4 })] }).text(),
      ).toContain('Failed')
      expect(
        mountCard({ runs: [run({ episode_outcomes: {}, errors_total: 0 })] }).get(
          '[data-testid="briefing-last-run"]',
        ).text(),
      ).toContain('Success')
    })

    it('emits open-pipeline-job-history from Details', async () => {
      const w = mountCard()
      await w.get('[data-testid="briefing-last-run-details"]').trigger('click')
      expect(w.emitted('open-pipeline-job-history')).toHaveLength(1)
    })
  })

  describe('corpus health line', () => {
    it('shows episode count, KG percentage, vectors label and feed count', () => {
      const w = mountCard()
      const t = w.get('[data-testid="briefing-corpus-health"]').text()
      expect(t).toContain('100 episodes')
      expect(t).toContain('90% with KG')
      expect(t).not.toContain('with GI')
      expect(t).toContain('5.0k vectors')
      expect(t).toContain('2 feeds')
    })

    it.each([
      [500, '500 vectors'],
      [1000, '1.0k vectors'],
      [10_000, '10k vectors'],
      [2_500_000, '2.5M vectors'],
    ])('formats %d index vectors as %s', (n, label) => {
      const w = mountCard({ indexEnv: indexEnv({}, n) })
      expect(w.get('[data-testid="briefing-corpus-health"]').text()).toContain(label)
    })

    it('shows dashes when coverage is missing or the index is unavailable', () => {
      const w = mountCard({ coverage: null, indexEnv: { available: false } })
      const t = w.get('[data-testid="briefing-corpus-health"]').text()
      expect(t).toContain('0 episodes')
      expect(t).toContain('— with KG')
      expect(t).not.toContain('vectors')
    })

    it('warns on low KG coverage and low index feed coverage', () => {
      const w = mountCard({
        coverage: coverage({ with_kg: 10 }),
        catalogFeedCount: 10,
      })
      const buttons = w.get('[data-testid="briefing-corpus-health"]').findAll('button')
      expect(buttons[1].classes()).toContain('text-warning')
      expect(buttons[2].classes()).toContain('text-warning')
    })

    it('does not warn when KG and index coverage are healthy', () => {
      const w = mountCard()
      const buttons = w.get('[data-testid="briefing-corpus-health"]').findAll('button')
      expect(buttons[1].classes()).not.toContain('text-warning')
      expect(buttons[2].classes()).not.toContain('text-warning')
    })

    it('titles the vectors button with the per-episode explanation, or unavailable', () => {
      const ok = mountCard()
      expect(ok.get('[data-testid="briefing-corpus-health"]').findAll('button')[2].attributes('title')).toContain(
        'Roughly 50.0 rows per catalog episode',
      )
      const none = mountCard({ indexEnv: null })
      expect(
        none.get('[data-testid="briefing-corpus-health"]').findAll('button')[2].attributes('title'),
      ).toBe('Vector index stats unavailable.')
    })

    it('selects the coverage tab from the KG and index buttons', async () => {
      const w = mountCard()
      const buttons = w.get('[data-testid="briefing-corpus-health"]').findAll('button')
      await buttons[1].trigger('click')
      await buttons[2].trigger('click')
      expect(w.emitted('select-tab')).toEqual([['coverage'], ['coverage']])
    })

    it('opens the library with an unfiltered handoff from the episode and feed buttons', async () => {
      const w = mountCard()
      const nav = useDashboardNavStore()
      const buttons = w.get('[data-testid="briefing-corpus-health"]').findAll('button')
      await buttons[0].trigger('click')
      expect(nav.pending).toEqual({ kind: 'library' })
      nav.consumeHandoff()
      await buttons[3].trigger('click')
      expect(nav.pending).toEqual({ kind: 'library' })
      expect(w.emitted('open-library')).toHaveLength(2)
    })
  })

  describe('action items', () => {
    it('says everything looks good when nothing needs attention', () => {
      const w = mountCard()
      expect(w.get('[data-testid="briefing-all-clear"]').text()).toContain('Everything looks good')
      expect(w.findAll('[data-testid="briefing-action-item"]')).toHaveLength(0)
    })

    it('lists failed episodes and hands off to the library from the run date', async () => {
      const w = mountCard({ runs: [run({ episode_outcomes: { ok: 3, failed: 1 } })] })
      expect(actionTexts(w)).toEqual(['→ 1 episode failed in last run'])
      const nav = useDashboardNavStore()
      await w.get('[data-testid="briefing-action-item"] button').trigger('click')
      expect(nav.pending).toEqual({ kind: 'library', since: '2026-06-15' })
      expect(w.emitted('open-library')).toHaveLength(1)
    })

    it('pluralises failed episodes', () => {
      const w = mountCard({ runs: [run({ episode_outcomes: { ok: 3, failed: 4 } })] })
      expect(actionTexts(w)[0]).toBe('→ 4 episodes failed in last run')
    })

    it('flags missing KG when coverage is below the threshold and hands off missingKgOnly', async () => {
      const w = mountCard({ coverage: coverage({ with_kg: 40, with_neither: 60 }) })
      expect(actionTexts(w)).toEqual(['→ 60 episodes have no KG artifacts'])
      const nav = useDashboardNavStore()
      await w.get('[data-testid="briefing-action-item"] button').trigger('click')
      expect(nav.pending).toEqual({ kind: 'library', missingKgOnly: true })
    })

    it('uses singular wording for exactly one episode without KG', () => {
      const w = mountCard({ coverage: coverage({ total_episodes: 3, with_kg: 1, with_neither: 1 }) })
      expect(actionTexts(w)).toEqual(['→ 2 episodes have no KG artifacts'])
      const one = mountCard({ coverage: coverage({ total_episodes: 2, with_kg: 1, with_neither: 0 }) })
      expect(actionTexts(one)).toEqual([])
      const neither = mountCard({ coverage: coverage({ total_episodes: 10, with_kg: 9, with_neither: 1 }) })
      expect(actionTexts(neither)).toEqual(['→ 1 episode have no KG artifacts'])
    })

    it('falls back to the with_neither count when KG coverage is healthy and nothing else is listed', async () => {
      const w = mountCard({ coverage: coverage({ with_kg: 90, with_neither: 3 }) })
      expect(actionTexts(w)).toEqual(['→ 3 episodes have no KG artifacts'])
      const nav = useDashboardNavStore()
      await w.get('[data-testid="briefing-action-item"] button').trigger('click')
      expect(nav.pending).toEqual({ kind: 'library' })
    })

    it('does not use the with_neither fallback when a failure item is already listed', () => {
      const w = mountCard({
        runs: [run({ episode_outcomes: { ok: 1, failed: 1 } })],
        coverage: coverage({ with_kg: 90, with_neither: 3 }),
      })
      expect(actionTexts(w)).toEqual(['→ 1 episode failed in last run'])
    })

    it('flags a missing vector index and emits open-index-controls', async () => {
      const w = mountCard({ indexEnv: { available: false } })
      expect(actionTexts(w)).toEqual(['→ Vector index has not been built'])
      await w.get('[data-testid="briefing-action-item"] button').trigger('click')
      expect(w.emitted('open-index-controls')).toHaveLength(1)
    })

    it('flags a stale index only when old and a rebuild is recommended', () => {
      const stale = indexEnv({ reindex_recommended: true })
      stale.stats!.last_updated = '2026-06-01T12:00:00Z'
      expect(actionTexts(mountCard({ indexEnv: stale }))).toEqual(['→ Index last rebuilt 14 days ago'])

      const notRecommended = indexEnv({ reindex_recommended: false })
      notRecommended.stats!.last_updated = '2026-06-01T12:00:00Z'
      expect(actionTexts(mountCard({ indexEnv: notRecommended }))).toEqual([])

      const recent = indexEnv({ reindex_recommended: true })
      expect(actionTexts(mountCard({ indexEnv: recent }))).toEqual([])
    })

    it('ignores a stale-index check when the timestamp is missing or unparseable', () => {
      const a = indexEnv({ reindex_recommended: true })
      a.stats!.last_updated = ''
      const b = indexEnv({ reindex_recommended: true })
      b.stats!.last_updated = 'garbage'
      expect(actionTexts(mountCard({ indexEnv: a }))).toEqual([])
      expect(actionTexts(mountCard({ indexEnv: b }))).toEqual([])
    })

    it('flags no runs in over 7 days when history spans more than 8 days', async () => {
      const runs = [
        run({ created_at: '2026-06-01T00:00:00Z' }),
        run({ created_at: '2026-05-01T00:00:00Z', relative_path: 'old' }),
      ]
      const w = mountCard({ runs })
      expect(actionTexts(w)).toEqual(['→ No pipeline runs in over 7 days'])
      await w.get('[data-testid="briefing-action-item"] button').trigger('click')
      expect(w.emitted('open-pipeline-run-history')).toHaveLength(1)
    })

    it('does not flag staleness for a short history or a recent newest run', () => {
      expect(
        actionTexts(
          mountCard({
            runs: [run({ created_at: '2026-06-01T00:00:00Z' }), run({ created_at: '2026-05-30T00:00:00Z' })],
          }),
        ),
      ).toEqual([])
      expect(
        actionTexts(
          mountCard({
            runs: [run({ created_at: '2026-06-14T00:00:00Z' }), run({ created_at: '2026-05-01T00:00:00Z' })],
          }),
        ),
      ).toEqual([])
    })

    it('ignores unparseable run timestamps in the staleness check', () => {
      expect(
        actionTexts(
          mountCard({ runs: [run({ created_at: 'bad' }), run({ created_at: 'worse' })] }),
        ),
      ).toEqual([])
    })

    it('caps the list at three items', () => {
      const runs = [
        run({ created_at: '2026-06-01T00:00:00Z', episode_outcomes: { ok: 1, failed: 1 } }),
        run({ created_at: '2026-05-01T00:00:00Z', relative_path: 'old' }),
      ]
      const w = mountCard({
        runs,
        coverage: coverage({ with_kg: 10 }),
        indexEnv: { available: false },
      })
      // failed + no KG + no index + stale = 4 candidates, 3 shown.
      expect(actionTexts(w)).toHaveLength(3)
      expect(actionTexts(w)[2]).toBe('→ Vector index has not been built')
    })
  })

  describe('a build\'s coverage (briefingCoverage)', () => {
    afterEach(() => briefingCoverage.reset())

    it('adds its stat after the KG share and its actions after the KG item, each opening the Library', async () => {
      const seen: number[] = []
      briefingCoverage.register({
        id: 'probe',
        stat: (c) => ({ text: `${c.with_gi} probed`, title: 'Probe', warn: true }),
        actions: (_c, before) => {
          seen.push(before)
          return [{ text: 'Probe needs a look', primary: 'Open', handoff: { kind: 'library', params: { probe: '1' } } }]
        },
      })
      const w = mountCard({ coverage: coverage({ with_kg: 40, with_neither: 60 }) })
      const stat = w.get('[data-testid="briefing-stat-probe"]')
      expect(stat.text()).toBe('90 probed')
      expect(stat.classes()).toContain('text-warning')
      expect(actionTexts(w)).toEqual(['→ 60 episodes have no KG artifacts', '→ Probe needs a look'])
      expect(seen.at(-1)).toBe(1)
      const nav = useDashboardNavStore()
      await w.findAll('[data-testid="briefing-action-item"]')[1].get('button').trigger('click')
      expect(nav.pending).toEqual({ kind: 'library', params: { probe: '1' } })
      expect(w.emitted('open-library')).toHaveLength(1)
      await stat.trigger('click')
      expect(w.emitted('select-tab')).toEqual([['coverage']])
    })

    it('skips a null stat and shows no build figures without coverage', () => {
      briefingCoverage.register({ id: 'quiet', stat: () => null })
      expect(mountCard().find('[data-testid="briefing-stat-quiet"]').exists()).toBe(false)
      briefingCoverage.register({ id: 'loud', stat: () => ({ text: 'x', title: 't', warn: false }) })
      expect(mountCard({ coverage: null }).find('[data-testid="briefing-stat-loud"]').exists()).toBe(false)
    })
  })
})
