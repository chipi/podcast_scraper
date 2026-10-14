// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ClusteringHealthCard from './ClusteringHealthCard.vue'
import { useShellStore } from '../../stores/shell'
import { getClusteringHealth, postStorylineSeed, type ClusteringHealth } from '../../api/clusteringHealthApi'
import { submitEnrichmentJob } from '../../api/enrichmentApi'

vi.mock('../../api/clusteringHealthApi', () => ({
  getClusteringHealth: vi.fn(),
  postStorylineSeed: vi.fn(),
}))
vi.mock('../../api/enrichmentApi', () => ({ submitEnrichmentJob: vi.fn() }))
// The chart draws on a canvas happy-dom does not have; what it is given is what is tested.
vi.mock('./MultiSeriesLineChart.vue', () => ({
  default: {
    props: ['title', 'labels', 'series', 'helpText'],
    template: '<div data-testid="chart" :data-title="title" :data-points="labels.length" />',
  },
}))

const health = (over: Partial<ClusteringHealth['storylines']> = {}): ClusteringHealth => ({
  storylines: {
    latest: {
      ts: 1_791_936_000,
      source: 'artifact',
      ok: true,
      storylines: 99,
      active: 82,
      quiet: 17,
      candidates: 47,
      episodes_covered: 795,
      episodes_total: 2635,
      changes: {
        created: [{ id: 'thc:chips', name: 'chips' }],
        renamed: [{ id: 'thc:a', from: 'risk', to: 'risk management' }],
        absorbed: [{ id: 'thc:s', name: 'small', into: 'thc:a', into_name: 'risk management' }],
      },
    },
    health: { ok: true, source: 'artifact' },
    seed: { state: 'none', id: null },
    series: [
      { ts: 1_730_000_000, source: 'replay', storylines: 2 },
      { ts: 1_791_936_000, source: 'artifact', storylines: 99 },
    ],
    ...over,
  },
  themes: {
    latest: { ts: 1_791_936_000, themes: 2500, topics: 23166, kept: 2294, changes: {} },
    series: [{ ts: 1_791_936_000, themes: 2500 }],
  },
})

async function mountCard() {
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  const w = mount(ClusteringHealthCard)
  await flushPromises()
  return w
}

describe('ClusteringHealthCard (RFC-128, #2349)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(getClusteringHealth).mockReset()
    vi.mocked(postStorylineSeed).mockReset()
    vi.mocked(submitEnrichmentJob).mockReset()
  })

  it('shows the counts, coverage, registry health and the last run\'s changes by name', async () => {
    vi.mocked(getClusteringHealth).mockResolvedValue(health())
    const w = await mountCard()
    expect(w.get('[data-testid="clustering-storylines-count"]').text()).toContain('99')
    expect(w.get('[data-testid="clustering-storylines"]').text()).toContain('30%')
    expect(w.get('[data-testid="clustering-registry-health"]').text()).toContain('healthy')
    expect(w.get('[data-testid="clustering-themes-count"]').text()).toContain('2500')
    const changes = w.get('[data-testid="clustering-storyline-changes"]').text()
    expect(changes).toContain('Created 1')
    expect(changes).toContain('risk → risk management')
    expect(changes).toContain('small → risk management')
    expect(w.get('[data-testid="clustering-theme-changes"]').text()).toContain('No changes.')
  })

  it('charts storylines only once there are two points, and themes not with one', async () => {
    vi.mocked(getClusteringHealth).mockResolvedValue(health())
    const w = await mountCard()
    const charts = w.findAll('[data-testid="chart"]')
    expect(charts.map((c) => c.attributes('data-title'))).toEqual(['Storylines over time'])
    expect(charts[0].attributes('data-points')).toBe('2')
  })

  it('says when the registry needs attention', async () => {
    vi.mocked(getClusteringHealth).mockResolvedValue(
      health({ health: { ok: false, source: 'copy' } }),
    )
    const w = await mountCard()
    const cell = w.get('[data-testid="clustering-registry-health"]')
    expect(cell.text()).toContain('needs attention')
    expect(cell.text()).toContain('copy')
  })

  it('re-runs the storyline enricher alone', async () => {
    vi.mocked(getClusteringHealth).mockResolvedValue(health())
    vi.mocked(submitEnrichmentJob).mockResolvedValue({
      job_id: 'j1',
      status: 'queued',
      corpus_path: '/corpus',
    } as never)
    const w = await mountCard()
    await w.get('[data-testid="clustering-rerun-storylines"]').trigger('click')
    await flushPromises()
    expect(submitEnrichmentJob).toHaveBeenCalledWith('/corpus', {
      only: ['topic_theme_clusters'],
      corpus_only: true,
    })
    expect(w.get('[data-testid="clustering-notice"]').text()).toContain('job j1')
  })

  it('previews the history seed, then applies it, and offers undo once there is one', async () => {
    vi.mocked(getClusteringHealth).mockResolvedValue(health())
    vi.mocked(postStorylineSeed).mockResolvedValueOnce({
      mode: 'preview',
      current: 99,
      with_history: 51,
      folded: 59,
      added_from_history: [],
      candidates: 47,
      history_months: 24,
      already_taken: false,
    })
    const w = await mountCard()
    expect(w.find('[data-testid="clustering-seed-apply"]').exists()).toBe(false)
    expect(w.find('[data-testid="clustering-seed-undo"]').exists()).toBe(false)
    await w.get('[data-testid="clustering-seed-preview"]').trigger('click')
    await flushPromises()
    expect(w.get('[data-testid="clustering-seed-report"]').text()).toContain('99 storylines today')

    vi.mocked(postStorylineSeed).mockResolvedValueOnce({ mode: 'apply', applied: true })
    vi.mocked(getClusteringHealth).mockResolvedValue(
      health({ seed: { state: 'pending', id: 'abc' } }),
    )
    await w.get('[data-testid="clustering-seed-apply"]').trigger('click')
    await flushPromises()
    expect(postStorylineSeed).toHaveBeenLastCalledWith('/corpus', 'apply')
    expect(w.get('[data-testid="clustering-seed-state"]').text()).toBe('pending')
    expect(w.find('[data-testid="clustering-seed-undo"]').exists()).toBe(true)
  })
})
