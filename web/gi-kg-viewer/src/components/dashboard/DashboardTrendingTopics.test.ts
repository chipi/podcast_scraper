// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import DashboardTrendingTopics from './DashboardTrendingTopics.vue'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { fetchCachedCorpusEnvelope } from '../../composables/useEnrichmentEnvelopeCache'

vi.mock('../../composables/useEnrichmentEnvelopeCache', () => ({
  fetchCachedCorpusEnvelope: vi.fn(),
}))
const fetchEnvelope = vi.mocked(fetchCachedCorpusEnvelope)

const ENV = {
  data: {
    window_months: ['2026-01', '2026-02', '2026-03'],
    topics: [
      { topic_id: 'topic:ai', topic_label: 'ai', velocity_last_over_6mo: 2, total: 10, monthly_counts: { '2026-01': 1, '2026-02': 3, '2026-03': 6 } },
      { topic_id: 'topic:policy', topic_label: 'foreign policy', velocity_last_over_6mo: 4, total: 3, monthly_counts: { '2026-03': 3 } },
      { topic_id: 'topic:steady', topic_label: 'steady', velocity_last_over_6mo: 1, total: 20, monthly_counts: {} }, // not rising
    ],
  },
}

async function mountIt() {
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  const w = mount(DashboardTrendingTopics)
  for (let i = 0; i < 8; i++) await w.vm.$nextTick()
  return { w, subject: useSubjectStore() }
}

describe('DashboardTrendingTopics', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    fetchEnvelope.mockResolvedValue(ENV as never)
  })

  it('shows rising topics as pills by default, sorted by velocity', async () => {
    const { w } = await mountIt()
    const chips = w.findAll('[data-testid="trend-chip"]')
    // policy (4x) before ai (2x); steady excluded.
    expect(chips).toHaveLength(2)
    expect(chips[0].text()).toContain('foreign policy')
    expect(chips[0].text()).toContain('4×')
  })

  it('opens the topic node view when a pill is clicked', async () => {
    const { w, subject } = await mountIt()
    await w.findAll('[data-testid="trend-chip"]')[0].trigger('click')
    expect(subject.graphNodeCyId).toBe('topic:policy')
  })

  it('toggles to the stream and momentum views (one band/point per rising topic)', async () => {
    const { w } = await mountIt()
    await w.get('[data-testid="trend-view-stream"]').trigger('click')
    expect(w.findAll('[data-testid="trend-stream-band"]')).toHaveLength(2)
    await w.get('[data-testid="trend-view-momentum"]').trigger('click')
    expect(w.findAll('[data-testid="trend-momentum-point"]')).toHaveLength(2)
  })

  it('shows an empty state when nothing is rising', async () => {
    fetchEnvelope.mockResolvedValue({
      data: { topics: [{ topic_id: 'topic:flat', velocity_last_over_6mo: 0.9, total: 99, monthly_counts: {} }] },
    } as never)
    const { w } = await mountIt()
    expect(w.find('[data-testid="intelligence-trending-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="trend-chip"]').exists()).toBe(false)
  })
})
