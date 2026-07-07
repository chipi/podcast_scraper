// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import TrendingGlobal from './TrendingGlobal.vue'
import { useShellStore } from '../../stores/shell'
import { fetchCorpusTrending } from '../../api/corpusTrendingApi'

vi.mock('../../api/corpusTrendingApi', () => ({ fetchCorpusTrending: vi.fn() }))
const fetchTrending = vi.mocked(fetchCorpusTrending)

const ent = (over: Record<string, unknown>) => ({
  entity_id: 'topic:ai',
  kind: 'topic',
  label: 'AI',
  velocity: 2.1,
  volume: 3,
  heating_up: true,
  total: 9,
  series: [0, 1, 2, 4],
  ...over,
})

async function mountIt() {
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  const w = mount(TrendingGlobal)
  for (let i = 0; i < 8; i++) await w.vm.$nextTick()
  return w
}

describe('TrendingGlobal (operator global momentum view)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('renders a section per non-empty kind with rows + sparklines + velocity', async () => {
    fetchTrending.mockResolvedValue({
      status: 'ok',
      document: {
        as_of_week: '2026-W29',
        kinds: {
          topic: [ent({}), ent({ entity_id: 'topic:health', label: 'Health', velocity: 1.6 })],
          storyline: [ent({ entity_id: 'thc:risk', kind: 'storyline', label: 'Managing risk' })],
          episode: [],
        },
      },
    } as never)
    const w = await mountIt()
    expect(w.find('[data-testid="trending-global"]').exists()).toBe(true)
    expect(w.find('[data-testid="trending-global-asof"]').text()).toBe('2026-W29')
    // topic + storyline sections render; empty episode kind does not
    expect(w.find('[data-testid="trending-global-kind-topic"]').exists()).toBe(true)
    expect(w.find('[data-testid="trending-global-kind-storyline"]').exists()).toBe(true)
    expect(w.find('[data-testid="trending-global-kind-episode"]').exists()).toBe(false)
    expect(w.findAll('[data-testid="trending-global-row"]')).toHaveLength(3)
    expect(w.findAll('[data-testid="trending-global-sparkline"]').length).toBeGreaterThan(0)
    expect(w.find('[data-testid="trending-global-kind-topic"]').text()).toContain('AI')
    expect(w.find('[data-testid="trending-global-kind-topic"]').text()).toContain('2.1×')
  })

  it('hides entirely when there is no trending signal', async () => {
    fetchTrending.mockResolvedValue({
      status: 'ok',
      document: { as_of_week: '2026-W29', kinds: {} },
    } as never)
    const w = await mountIt()
    expect(w.find('[data-testid="trending-global"]').exists()).toBe(false)
  })

  it('hides on a missing corpus (404)', async () => {
    fetchTrending.mockResolvedValue({ status: 'missing' } as never)
    const w = await mountIt()
    expect(w.find('[data-testid="trending-global"]').exists()).toBe(false)
  })
})
