// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchCorpusCoverage } from '../../api/corpusCoverageApi'
import type { CorpusCoverageResponse } from '../../api/corpusCoverageApi'
import { fetchCorpusFeeds } from '../../api/corpusLibraryApi'
import { fetchCorpusRunsSummary } from '../../api/corpusMetricsApi'
import type { CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import { fetchCorpusDigest } from '../../api/digestApi'
import { useDashboardNavStore } from '../../stores/dashboardNav'
import { useIndexStatsStore } from '../../stores/indexStats'
import { useShellStore } from '../../stores/shell'
import DashboardView from './DashboardView.vue'

vi.mock('../../api/corpusCoverageApi', () => ({ fetchCorpusCoverage: vi.fn() }))
vi.mock('../../api/digestApi', () => ({ fetchCorpusDigest: vi.fn() }))
vi.mock('../../api/corpusLibraryApi', () => ({ fetchCorpusFeeds: vi.fn() }))
vi.mock('../../api/corpusMetricsApi', () => ({ fetchCorpusRunsSummary: vi.fn() }))
vi.mock('../../lib/analytics', () => ({ track: vi.fn() }))

const mockCoverage = vi.mocked(fetchCorpusCoverage)
const mockDigest = vi.mocked(fetchCorpusDigest)
const mockFeeds = vi.mocked(fetchCorpusFeeds)
const mockRuns = vi.mocked(fetchCorpusRunsSummary)

const STUBS = {
  BriefingCard: true,
  CoverageByMonthChart: true,
  FeedCoverageTable: true,
  ArtifactActivityChart: true,
  IndexStatusCard: true,
  CostRollupCard: true,
  IntelligenceSnapshot: true,
  TopicBriefingCards: true,
  QueryActivityChart: true,
  PipelineCleanupMetrics: true,
  PipelineAdExcisionMetrics: true,
  PipelineJobsCard: true,
  PipelineJobHistoryStrip: true,
  PipelineRunHistoryStrip: true,
  PipelineJobLogDialog: true,
  PipelineFeedHistoryGrid: true,
  PipelineStageChart: true,
  VerticalBarChart: true,
}

function run(id: string, over: Partial<CorpusRunSummaryItem> = {}): CorpusRunSummaryItem {
  return {
    relative_path: `feeds/a/${id}/run.json`,
    run_id: id,
    created_at: '2026-06-10T10:00:00Z',
    run_duration_seconds: 100,
    episodes_scraped_total: 10,
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

const COVERAGE: CorpusCoverageResponse = {
  path: '/corpus',
  total_episodes: 4,
  with_gi: 2,
  with_kg: 2,
  with_both: 1,
  with_neither: 1,
  by_month: [{ month: '2026-02', total: 4, with_gi: 2, with_kg: 2, with_both: 1 }],
  by_feed: [{ feed_id: 'f1', display_title: 'F1', total: 4, with_gi: 2, with_kg: 2 }],
}

const DIGEST = {
  path: '/corpus',
  window: '7d',
  window_start_utc: '',
  window_end_utc: '',
  compact: false,
  rows: [],
  topics: [],
  topics_unavailable_reason: null,
} as never

function connect(path = '/corpus', health: string | null = 'ok') {
  const shell = useShellStore()
  shell.corpusPath = path
  shell.healthStatus = health
  vi.spyOn(shell, 'fetchArtifactList').mockResolvedValue(undefined)
  const indexStats = useIndexStatsStore()
  vi.spyOn(indexStats, 'refreshIndexStats').mockResolvedValue(undefined)
  return { shell, indexStats }
}

async function mountView() {
  const w = mount(DashboardView, { global: { stubs: STUBS }, attachTo: document.body })
  await flushPromises()
  return w
}

const comp = (w: Awaited<ReturnType<typeof mountView>>, name: string) =>
  w.findComponent({ name })

async function openTab(w: Awaited<ReturnType<typeof mountView>>, id: string) {
  await w.get(`[data-testid="dashboard-tab-${id}"]`).trigger('click')
  await flushPromises()
}

describe('DashboardView', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    for (const m of [mockCoverage, mockDigest, mockFeeds, mockRuns]) m.mockReset()
    mockRuns.mockResolvedValue({ path: '/corpus', runs: [] })
    mockCoverage.mockResolvedValue(COVERAGE)
    mockFeeds.mockResolvedValue({ path: '/corpus', feeds: [] } as never)
    mockDigest.mockResolvedValue(DIGEST)
    HTMLElement.prototype.scrollIntoView ??= () => {}
  })

  describe('loading the dashboard', () => {
    it('does not fetch without a corpus path and clears loading', async () => {
      connect('   ')
      const w = await mountView()
      expect(mockRuns).not.toHaveBeenCalled()
      expect(w.text()).not.toContain('Loading dashboard…')
      expect(comp(w, 'BriefingCard').props('runs')).toEqual([])
      expect(comp(w, 'BriefingCard').props('apiReady')).toBe(true)
    })

    it('does not fetch until the API is healthy', async () => {
      connect('/corpus', null)
      const w = await mountView()
      expect(mockRuns).not.toHaveBeenCalled()
      expect(comp(w, 'BriefingCard').props('apiReady')).toBe(false)
    })

    it('shows a loading line while the summary is in flight', async () => {
      connect()
      let resolve!: (v: { path: string; runs: CorpusRunSummaryItem[] }) => void
      mockRuns.mockReturnValue(new Promise((r) => { resolve = r }))
      const w = mount(DashboardView, { global: { stubs: STUBS } })
      await flushPromises()
      expect(w.text()).toContain('Loading dashboard…')
      resolve({ path: '/corpus', runs: [] })
      await flushPromises()
      expect(w.text()).not.toContain('Loading dashboard…')
    })

    it('feeds the briefing card with runs, coverage, feed count and the index envelope', async () => {
      const { indexStats, shell } = connect()
      mockRuns.mockResolvedValue({ path: '/corpus', runs: [run('a')] })
      mockFeeds.mockResolvedValue({ path: '/corpus', feeds: [{ feed_id: 'f1' }, { feed_id: 'f2' }] } as never)
      const w = await mountView()
      const card = comp(w, 'BriefingCard')
      expect(mockRuns).toHaveBeenCalledWith('/corpus')
      expect(card.props('runs')).toHaveLength(1)
      expect(card.props('coverage')).toEqual(COVERAGE)
      expect(card.props('catalogFeedCount')).toBe(2)
      expect(card.props('corpusPath')).toBe('/corpus')
      expect(indexStats.refreshIndexStats).toHaveBeenCalled()
      expect(shell.fetchArtifactList).toHaveBeenCalled()
    })

    it('shows the error and clears runs/coverage when the run summary fails', async () => {
      connect()
      mockRuns.mockRejectedValue(new Error('summary down'))
      const w = await mountView()
      expect(w.text()).toContain('summary down')
      expect(comp(w, 'BriefingCard').props('runs')).toEqual([])
      expect(comp(w, 'BriefingCard').props('coverage')).toBeNull()
      expect(w.text()).not.toContain('Loading dashboard…')
    })

    it('stringifies a non-Error failure', async () => {
      connect()
      mockRuns.mockRejectedValue('plain string')
      const w = await mountView()
      expect(w.text()).toContain('plain string')
    })

    it('treats a non-array runs payload as no runs', async () => {
      connect()
      mockRuns.mockResolvedValue({ path: '/corpus', runs: null as never })
      const w = await mountView()
      expect(comp(w, 'BriefingCard').props('runs')).toEqual([])
    })

    it('degrades gracefully when coverage, feeds and digest fail', async () => {
      connect()
      mockRuns.mockResolvedValue({ path: '/corpus', runs: [run('a')] })
      mockCoverage.mockRejectedValue(new Error('x'))
      mockFeeds.mockRejectedValue(new Error('x'))
      mockDigest.mockRejectedValue(new Error('x'))
      const w = await mountView()
      expect(comp(w, 'BriefingCard').props('runs')).toHaveLength(1)
      expect(comp(w, 'BriefingCard').props('coverage')).toBeNull()
      expect(comp(w, 'BriefingCard').props('catalogFeedCount')).toBe(0)
      expect(w.find('[data-testid="dashboard-tab-coverage"]').exists()).toBe(true)
      expect(w.text()).not.toContain('Loading dashboard…')
      expect(comp(w, 'CoverageByMonthChart').exists()).toBe(false)
    })

    it('defaults feeds to empty when the payload omits them', async () => {
      connect()
      mockFeeds.mockResolvedValue({ path: '/corpus' } as never)
      const w = await mountView()
      expect(comp(w, 'BriefingCard').props('catalogFeedCount')).toBe(0)
    })

    it('reloads when the corpus path changes and discards a superseded response', async () => {
      const { shell } = connect('/first')
      let resolveFirst!: (v: { path: string; runs: CorpusRunSummaryItem[] }) => void
      mockRuns.mockReturnValueOnce(new Promise((r) => { resolveFirst = r }))
      const w = mount(DashboardView, { global: { stubs: STUBS } })
      await flushPromises()
      mockRuns.mockResolvedValueOnce({ path: '/second', runs: [run('new')] })
      shell.corpusPath = '/second'
      await flushPromises()
      resolveFirst({ path: '/first', runs: [run('stale1'), run('stale2')] })
      await flushPromises()
      expect(mockRuns).toHaveBeenLastCalledWith('/second')
      expect(comp(w, 'BriefingCard').props('runs')).toHaveLength(1)
      expect(w.text()).not.toContain('Loading dashboard…')
    })

    it('drops stale feeds and digest results from a superseded load', async () => {
      const { shell } = connect('/first')
      let resolveFeeds!: (v: never) => void
      mockFeeds.mockReturnValueOnce(new Promise((r) => { resolveFeeds = r as never }))
      const w = mount(DashboardView, { global: { stubs: STUBS } })
      await flushPromises()
      mockFeeds.mockResolvedValueOnce({ path: '/b', feeds: [{ feed_id: 'only' }] } as never)
      shell.corpusPath = '/second'
      await flushPromises()
      resolveFeeds({ path: '/a', feeds: [{ feed_id: '1' }, { feed_id: '2' }, { feed_id: '3' }] } as never)
      await flushPromises()
      expect(comp(w, 'BriefingCard').props('catalogFeedCount')).toBe(1)
    })

    it('ignores a superseded summary failure', async () => {
      const { shell } = connect('/first')
      let rejectFirst!: (e: Error) => void
      mockRuns.mockReturnValueOnce(new Promise((_r, rej) => { rejectFirst = rej }))
      const w = mount(DashboardView, { global: { stubs: STUBS } })
      await flushPromises()
      mockRuns.mockResolvedValueOnce({ path: '/b', runs: [run('ok')] })
      shell.corpusPath = '/second'
      await flushPromises()
      rejectFirst(new Error('late failure'))
      await flushPromises()
      expect(w.text()).not.toContain('late failure')
      expect(comp(w, 'BriefingCard').props('runs')).toHaveLength(1)
    })
  })

  describe('briefing card wiring', () => {
    it('opens the Configuration index dialog request', async () => {
      const { indexStats } = connect()
      const w = await mountView()
      const before = indexStats.dialogOpenNonce
      comp(w, 'BriefingCard').vm.$emit('open-index-controls')
      expect(indexStats.dialogOpenNonce).toBe(before + 1)
    })

    it('forwards open-library', async () => {
      connect()
      const w = await mountView()
      comp(w, 'BriefingCard').vm.$emit('open-library')
      expect(w.emitted('open-library')).toHaveLength(1)
    })

    it('switches tab from select-tab', async () => {
      connect()
      const w = await mountView()
      comp(w, 'BriefingCard').vm.$emit('select-tab', 'pipeline')
      await flushPromises()
      expect(w.find('[data-testid="pipeline-jobs-history-panel"]').exists()).toBe(true)
    })

    it('opens Pipeline > Job history and scrolls to the panel', async () => {
      connect()
      const scroll = vi.spyOn(HTMLElement.prototype, 'scrollIntoView').mockImplementation(() => {})
      const w = await mountView()
      comp(w, 'BriefingCard').vm.$emit('open-pipeline-job-history')
      await flushPromises()
      expect(comp(w, 'PipelineJobHistoryStrip').exists()).toBe(true)
      expect(scroll).toHaveBeenCalledWith({ behavior: 'smooth', block: 'nearest' })
      scroll.mockRestore()
    })

    it('opens Pipeline > Run history with the loaded runs', async () => {
      connect()
      mockRuns.mockResolvedValue({ path: '/corpus', runs: [run('a')] })
      const scroll = vi.spyOn(HTMLElement.prototype, 'scrollIntoView').mockImplementation(() => {})
      const w = await mountView()
      comp(w, 'BriefingCard').vm.$emit('open-pipeline-run-history')
      await flushPromises()
      expect(comp(w, 'PipelineRunHistoryStrip').props('runs')).toHaveLength(1)
      expect(scroll).toHaveBeenCalled()
      scroll.mockRestore()
    })
  })

  describe('Coverage tab', () => {
    it('renders the coverage widgets from the coverage response', async () => {
      const { indexStats } = connect()
      const w = await mountView()
      indexStats.indexEnvelope = {
        available: true,
        stats: { feeds_indexed: ['f1'] } as never,
      }
      await flushPromises()
      expect(comp(w, 'CoverageByMonthChart').props('rows')).toEqual(COVERAGE.by_month)
      expect(comp(w, 'FeedCoverageTable').props('rows')).toEqual(COVERAGE.by_feed)
      expect(comp(w, 'FeedCoverageTable').props('feedsIndexed')).toEqual(['f1'])
      expect(comp(w, 'ArtifactActivityChart').exists()).toBe(true)
      expect(comp(w, 'IndexStatusCard').exists()).toBe(true)
    })

    it('passes an empty indexed-feed list when the index has no stats', async () => {
      connect()
      const w = await mountView()
      expect(comp(w, 'FeedCoverageTable').props('feedsIndexed')).toEqual([])
    })

    it('marks artifact activity loaded only once the artifact count is known', async () => {
      const { shell } = connect()
      const w = await mountView()
      expect(comp(w, 'ArtifactActivityChart').props('loaded')).toBe(false)
      shell.artifactCount = 3
      await flushPromises()
      expect(comp(w, 'ArtifactActivityChart').props('loaded')).toBe(true)
    })

    it('hands a month off to the library as a missing-KG date range', async () => {
      connect()
      const w = await mountView()
      const nav = useDashboardNavStore()
      comp(w, 'CoverageByMonthChart').vm.$emit('select-month', '2026-02')
      expect(nav.pending).toEqual({
        kind: 'library',
        since: '2026-02-01',
        until: '2026-02-28',
        missingKgOnly: true,
      })
      expect(w.emitted('open-library')).toHaveLength(1)
    })

    it('hands a feed off to the library', async () => {
      connect()
      const w = await mountView()
      const nav = useDashboardNavStore()
      comp(w, 'FeedCoverageTable').vm.$emit('select-feed', 'f1')
      expect(nav.pending).toEqual({ kind: 'library', feedId: 'f1' })
      expect(w.emitted('open-library')).toHaveLength(1)
    })
  })

  describe('Intelligence tab', () => {
    it('shows the digest snapshot and its sections; People only with a registered card', async () => {
      connect()
      const w = await mountView()
      await openTab(w, 'intelligence')
      expect(comp(w, 'IntelligenceSnapshot').props('digest')).toEqual(DIGEST)
      expect(w.text()).toContain('Topics')
      expect(w.text()).toContain('Pipeline & activity')
      // Top voices is a downstream build's card (E04 step 8b); with none registered, no People.
      expect(w.text()).not.toContain('People')
    })

    it('re-fetches the digest (7d, 3 rows) when the tab opens', async () => {
      connect()
      const w = await mountView()
      mockDigest.mockClear()
      await openTab(w, 'intelligence')
      expect(mockDigest).toHaveBeenCalledWith('/corpus', { window: '7d', compact: false, maxRows: 3 })
    })

    it('keeps a null digest when the intelligence digest refresh fails', async () => {
      connect()
      const w = await mountView()
      mockDigest.mockRejectedValue(new Error('x'))
      await openTab(w, 'intelligence')
      expect(comp(w, 'IntelligenceSnapshot').props('digest')).toBeNull()
    })

    it('skips the intelligence refresh when the API is not ready', async () => {
      const { shell } = connect()
      const w = await mountView()
      shell.healthStatus = null
      await flushPromises()
      mockDigest.mockClear()
      await openTab(w, 'intelligence')
      expect(mockDigest).not.toHaveBeenCalled()
    })

    it('forwards open-digest from the snapshot', async () => {
      connect()
      const w = await mountView()
      await openTab(w, 'intelligence')
      comp(w, 'IntelligenceSnapshot').vm.$emit('open-digest')
      expect(w.emitted('open-digest')).toHaveLength(1)
    })

    it('forwards go-graph from the built-in topic card', async () => {
      connect()
      const w = await mountView()
      await openTab(w, 'intelligence')
      comp(w, 'TopicBriefingCards').vm.$emit('go-graph', 'topic:a', 'fallback')
      expect(w.emitted('go-graph')).toEqual([['topic:a', 'fallback']])
    })

    it('passes the digest to the topic briefing card', async () => {
      connect()
      const w = await mountView()
      await openTab(w, 'intelligence')
      expect(comp(w, 'TopicBriefingCards').props('digest')).toEqual(DIGEST)
    })

    it('passes the latest run to the cleanup and ad-excision panels', async () => {
      connect()
      mockRuns.mockResolvedValue({ path: '/corpus', runs: [run('newest'), run('older')] })
      const w = await mountView()
      await openTab(w, 'intelligence')
      expect(comp(w, 'PipelineCleanupMetrics').props('run')?.run_id).toBe('newest')
      expect(comp(w, 'PipelineAdExcisionMetrics').props('run')?.run_id).toBe('newest')
    })

    it('passes null to the run panels when there are no runs', async () => {
      connect()
      const w = await mountView()
      await openTab(w, 'intelligence')
      expect(comp(w, 'PipelineCleanupMetrics').props('run')).toBeNull()
    })
  })

  describe('Pipeline tab', () => {
    it('defaults to the Jobs sub-tab and switches between sub-tabs', async () => {
      connect()
      const w = await mountView()
      await openTab(w, 'pipeline')
      expect(comp(w, 'PipelineJobsCard').exists()).toBe(true)
      await w.get('[data-testid="dashboard-pipeline-subtab-job-history"]').trigger('click')
      expect(comp(w, 'PipelineJobHistoryStrip').exists()).toBe(true)
      expect(comp(w, 'PipelineJobsCard').exists()).toBe(false)
      await w.get('[data-testid="dashboard-pipeline-subtab-history"]').trigger('click')
      expect(comp(w, 'PipelineRunHistoryStrip').exists()).toBe(true)
      await w.get('[data-testid="dashboard-pipeline-subtab-jobs"]').trigger('click')
      expect(comp(w, 'PipelineJobsCard').exists()).toBe(true)
    })

    it('switches to Job history when the jobs card asks for it', async () => {
      connect()
      const w = await mountView()
      await openTab(w, 'pipeline')
      comp(w, 'PipelineJobsCard').vm.$emit('go-to-job-history')
      await flushPromises()
      expect(comp(w, 'PipelineJobHistoryStrip').exists()).toBe(true)
    })

    it('renders no trend charts and zero outcomes without runs', async () => {
      connect()
      const w = await mountView()
      await openTab(w, 'pipeline')
      expect(w.find('[data-testid="pipeline-duration-trend"]').exists()).toBe(false)
      expect(w.find('[data-testid="pipeline-episodes-per-run"]').exists()).toBe(false)
      expect(comp(w, 'PipelineStageChart').exists()).toBe(false)
      const t = w.get('[data-testid="pipeline-episode-outcomes"]').text()
      expect(t).toContain('0 succeeded')
      expect(t).toContain('0 skipped')
      expect(t).toContain('0 failed')
      expect(t).not.toContain('View failures')
    })

    it('shows the latest run outcomes and a View failures handoff when some failed', async () => {
      connect()
      mockRuns.mockResolvedValue({
        path: '/corpus',
        runs: [run('a', { episode_outcomes: { ok: 1200, skipped: 3, failed: 2 } })],
      })
      const w = await mountView()
      await openTab(w, 'pipeline')
      const t = w.get('[data-testid="pipeline-episode-outcomes"]')
      expect(t.text()).toContain(`${(1200).toLocaleString()} succeeded`)
      expect(t.text()).toContain('3 skipped')
      expect(t.text()).toContain('2 failed')
      const nav = useDashboardNavStore()
      await t.get('button').trigger('click')
      expect(nav.pending).toEqual({ kind: 'library' })
      expect(w.emitted('open-library')).toHaveLength(1)
    })

    it('charts stage timings from the latest run, clamping negatives to zero', async () => {
      connect()
      mockRuns.mockResolvedValue({
        path: '/corpus',
        runs: [
          run('a', {
            time_scraping_seconds: 30,
            time_parsing_seconds: -5,
            time_normalizing_seconds: null,
            time_io_and_waiting_seconds: 10,
          }),
        ],
      })
      const w = await mountView()
      await openTab(w, 'pipeline')
      expect(comp(w, 'PipelineStageChart').props('stages')).toEqual([
        { label: 'Scraping', seconds: 30 },
        { label: 'Parsing', seconds: 0 },
        { label: 'Normalizing', seconds: 0 },
        { label: 'I/O & waiting', seconds: 10 },
      ])
    })

    it('hides the stage chart when every stage is zero', async () => {
      connect()
      mockRuns.mockResolvedValue({ path: '/corpus', runs: [run('a')] })
      const w = await mountView()
      await openTab(w, 'pipeline')
      expect(comp(w, 'PipelineStageChart').exists()).toBe(false)
    })

    describe('duration trend', () => {
      const dated = (durations: number[]) =>
        durations.map((d, i) =>
          run(`r${i}`, {
            created_at: `2026-06-${String(i + 1).padStart(2, '0')}T00:00:00Z`,
            run_duration_seconds: d,
          }),
        )

      async function trendTitle(durations: number[]) {
        connect()
        mockRuns.mockResolvedValue({ path: '/corpus', runs: dated(durations) })
        const w = await mountView()
        await openTab(w, 'pipeline')
        const chart = w.get('[data-testid="pipeline-duration-trend"]')
        void chart
        return w.findAllComponents({ name: 'VerticalBarChart' })[0]
      }

      it('reports a slower latest run', async () => {
        const c = await trendTitle([100, 100, 100, 100, 150])
        expect(c.props('title')).toBe('Latest run 36% slower than 5-run average')
        expect(c.props('values')).toEqual([100, 100, 100, 100, 150])
        expect(c.props('labels')[0]).toBe('2026-06-01')
      })

      it('reports a faster latest run', async () => {
        const c = await trendTitle([100, 100, 100, 100, 50])
        expect(c.props('title')).toBe('Latest run 44% faster than 5-run average')
      })

      it('reports stable durations, and a single run as stable', async () => {
        expect((await trendTitle([100, 102, 98, 100, 101])).props('title')).toBe(
          'Run duration stable — within 10% of 5-run average',
        )
        expect((await trendTitle([100])).props('title')).toBe(
          'Run duration stable — within 10% of 5-run average',
        )
      })

      it('keeps only the five most recent runs in chronological order', async () => {
        const c = await trendTitle([1, 2, 3, 4, 5, 6, 7])
        expect(c.props('values')).toEqual([3, 4, 5, 6, 7])
      })

      it('ignores runs without a full date or a duration', async () => {
        connect()
        mockRuns.mockResolvedValue({
          path: '/corpus',
          runs: [
            run('short', { created_at: '2026', run_duration_seconds: 5 }),
            run('nodur', { run_duration_seconds: null }),
            run('nodate', { created_at: null }),
          ],
        })
        const w = await mountView()
        await openTab(w, 'pipeline')
        expect(w.find('[data-testid="pipeline-duration-trend"]').exists()).toBe(false)
      })
    })

    describe('episodes per run', () => {
      const dated = (eps: number[]) =>
        eps.map((e, i) =>
          run(`r${i}`, {
            created_at: `2026-06-${String(i + 1).padStart(2, '0')}T00:00:00Z`,
            episodes_scraped_total: e,
          }),
        )

      async function perRunChart(eps: number[]) {
        connect()
        mockRuns.mockResolvedValue({ path: '/corpus', runs: dated(eps) })
        const w = await mountView()
        await openTab(w, 'pipeline')
        w.get('[data-testid="pipeline-episodes-per-run"]')
        return w.findAllComponents({ name: 'VerticalBarChart' }).at(-1)!
      }

      it('reports the average episodes per run', async () => {
        const c = await perRunChart([10, 20, 30])
        expect(c.props('title')).toBe('Average 20 episodes per run')
        expect(c.props('values')).toEqual([10, 20, 30])
      })

      it('flags a declining volume when the last three runs are below average', async () => {
        const c = await perRunChart([100, 100, 100, 10, 10, 10])
        expect(c.props('title')).toBe('Processing volume declining — last 3 runs below average')
      })

      it('does not flag a decline with fewer than three runs', async () => {
        const c = await perRunChart([100, 1])
        expect(c.props('title')).toBe('Average 51 episodes per run')
      })

      it('caps the series at the 48 most recent runs and treats null as zero', async () => {
        connect()
        const runs = Array.from({ length: 50 }, (_, i) =>
          run(`r${i}`, {
            created_at: new Date(Date.UTC(2026, 0, 1 + i)).toISOString(),
            episodes_scraped_total: i === 49 ? null : i + 1,
          }),
        )
        mockRuns.mockResolvedValue({ path: '/corpus', runs })
        const w = await mountView()
        await openTab(w, 'pipeline')
        const c = w.findAllComponents({ name: 'VerticalBarChart' }).at(-1)!
        expect(c.props('values')).toHaveLength(48)
        expect(c.props('values')[0]).toBe(3)
        expect(c.props('values').at(-1)).toBe(0)
      })
    })

    it('passes runs and feeds to the feed history grid', async () => {
      connect()
      mockRuns.mockResolvedValue({ path: '/corpus', runs: [run('a')] })
      mockFeeds.mockResolvedValue({ path: '/corpus', feeds: [{ feed_id: 'f1' }] } as never)
      const w = await mountView()
      await openTab(w, 'pipeline')
      const grid = comp(w, 'PipelineFeedHistoryGrid')
      expect(grid.props('runs')).toHaveLength(1)
      expect(grid.props('feeds')).toHaveLength(1)
    })
  })

  it('always mounts the single shared job log dialog', async () => {
    connect()
    const w = await mountView()
    expect(comp(w, 'PipelineJobLogDialog').exists()).toBe(true)
    await openTab(w, 'pipeline')
    expect(comp(w, 'PipelineJobLogDialog').exists()).toBe(true)
  })
})
