// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, h } from 'vue'

import { dashboardCards } from '../../extensions'
import DashboardView from './DashboardView.vue'

vi.mock('../../api/corpusCoverageApi', () => ({ fetchCorpusCoverage: vi.fn(() => new Promise(() => {})) }))
vi.mock('../../api/digestApi', () => ({ fetchCorpusDigest: vi.fn(() => new Promise(() => {})) }))
vi.mock('../../api/corpusLibraryApi', () => ({ fetchCorpusFeeds: vi.fn(() => new Promise(() => {})) }))
vi.mock('../../api/corpusMetricsApi', () => ({ fetchCorpusRunsSummary: vi.fn(() => new Promise(() => {})) }))
vi.mock('../../api/corpusPersonsApi', () => ({ fetchCorpusTopPersons: vi.fn(() => new Promise(() => {})) }))

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
        DashboardTopicPerspectives: { template: '<div data-testid="card-perspectives" />' },
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

  it('renders only its own topic cards when nothing is registered', async () => {
    const w = await mountIntelligence()
    const ids = w.findAll('[data-testid^="card-"]').map((n) => n.attributes('data-testid'))
    expect(ids).toEqual(['card-perspectives', 'card-briefing'])
  })

  it('places registered cards by order and region, with their span, and forwards go-graph', async () => {
    dashboardCards.register({ id: 'first', region: 'topics', order: 10, component: card('first') })
    dashboardCards.register({ id: 'mid', region: 'topics', order: 30, span: 4, component: card('mid') })
    dashboardCards.register({ id: 'top', region: 'overview', component: card('top') })
    const w = await mountIntelligence()
    const ids = w.findAll('[data-testid^="card-"]').map((n) => n.attributes('data-testid'))
    expect(ids).toEqual(['card-top', 'card-first', 'card-perspectives', 'card-mid', 'card-briefing'])
    expect(w.find('[data-testid="card-mid"]').element.parentElement!.className).toContain('xl:col-span-4')
    await w.find('[data-testid="card-first"]').trigger('click')
    expect(w.emitted('go-graph')).toEqual([['t:first', undefined]])
  })
})
