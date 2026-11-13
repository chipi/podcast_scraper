// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, h } from 'vue'

import { dashboardCards } from '../../extensions'
import { useShellStore } from '../../stores/shell'
import DashboardView from './DashboardView.vue'

vi.mock('../../api/corpusCoverageApi', () => ({ fetchCorpusCoverage: vi.fn(() => new Promise(() => {})) }))
vi.mock('../../api/digestApi', () => ({ fetchCorpusDigest: vi.fn(() => new Promise(() => {})) }))
vi.mock('../../api/corpusLibraryApi', () => ({ fetchCorpusFeeds: vi.fn(() => new Promise(() => {})) }))
vi.mock('../../api/corpusMetricsApi', () => ({ fetchCorpusRunsSummary: vi.fn(() => new Promise(() => {})) }))

const card = (label: string) =>
  defineComponent({
    emits: ['go-graph'],
    setup(_, { emit }) {
      return () => h('button', { 'data-testid': `card-${label}`, onClick: () => emit('go-graph', `t:${label}`) }, label)
    },
  })

async function mountIntelligence() {
  const w = mount(DashboardView, {
    global: {
      stubs: {
        TopicBriefingCards: { template: '<div data-testid="card-briefing" />' },
        IntelligenceSnapshot: true,
      },
    },
  })
  await flushPromises()
  const tab = w.findAll('button').find((b) => b.text().trim() === 'Intelligence')
  await tab!.trigger('click')
  await flushPromises()
  return w
}

describe('DashboardView extension cards', () => {
  beforeEach(() => setActivePinia(createPinia()))
  afterEach(() => dashboardCards.reset())

  it('renders only its own topic card, and no People section, when nothing is registered', async () => {
    const w = await mountIntelligence()
    const ids = w.findAll('[data-testid^="card-"]').map((n) => n.attributes('data-testid'))
    expect(ids).toEqual(['card-briefing'])
    expect(w.find('#dash-intel-people').exists()).toBe(false)
  })

  it('places registered cards by order and region, with their span, and forwards go-graph', async () => {
    dashboardCards.register({ id: 'first', region: 'topics', order: 10, component: card('first') })
    dashboardCards.register({ id: 'mid', region: 'topics', order: 30, span: 4, component: card('mid') })
    dashboardCards.register({ id: 'top', region: 'overview', component: card('top') })
    dashboardCards.register({ id: 'voices', region: 'people', span: 2, component: card('voices') })
    const w = await mountIntelligence()
    const ids = w.findAll('[data-testid^="card-"]').map((n) => n.attributes('data-testid'))
    expect(ids).toEqual(['card-top', 'card-first', 'card-mid', 'card-briefing', 'card-voices'])
    expect(w.find('[data-testid="card-mid"]').element.parentElement!.className).toContain('xl:col-span-4')
    expect(w.find('#dash-intel-people').text()).toBe('People')
    expect(w.find('[data-testid="card-voices"]').element.parentElement!.className).toContain('xl:col-span-2')
    await w.find('[data-testid="card-first"]').trigger('click')
    expect(w.emitted('go-graph')).toEqual([['t:first', undefined]])
  })

  it('renders Coverage tab cards after the by-month chart with the coverage, and forwards open-library', async () => {
    const { fetchCorpusCoverage } = await import('../../api/corpusCoverageApi')
    const { fetchCorpusRunsSummary } = await import('../../api/corpusMetricsApi')
    // Coverage loads after the runs summary.
    vi.mocked(fetchCorpusRunsSummary).mockResolvedValueOnce({ path: '/c', runs: [] } as never)
    const cov = { path: '/c', total_episodes: 1, with_gi: 0, with_kg: 1, with_both: 0, with_neither: 0, by_month: [], by_feed: [] }
    vi.mocked(fetchCorpusCoverage).mockResolvedValueOnce(cov as never)
    const shell = useShellStore()
    shell.corpusPath = '/c'
    shell.healthStatus = 'ok'
    let received: unknown
    dashboardCards.register({
      id: 'cov',
      region: 'coverage',
      component: defineComponent({
        props: { coverage: { type: Object, default: null } },
        emits: ['open-library'],
        setup(props, { emit }) {
          return () => {
            received = props.coverage
            return h('button', { 'data-testid': 'card-cov', onClick: () => emit('open-library') }, 'cov')
          }
        },
      }),
    })
    const w = mount(DashboardView, { global: { stubs: { CoverageByMonthChart: { template: '<div data-testid="month-chart" />' } } } })
    await flushPromises()
    const html = w.html()
    expect(html.indexOf('month-chart')).toBeLessThan(html.indexOf('card-cov'))
    expect(received).toEqual(cov)
    await w.get('[data-testid="card-cov"]').trigger('click')
    expect(w.emitted('open-library')).toHaveLength(1)
  })
})
