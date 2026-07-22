import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import TopicBrowseView from './TopicBrowseView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'home', component: { template: '<div/>' } },
      { path: '/browse/topics', name: 'browse-topics', component: TopicBrowseView },
      { path: '/topic/:id', name: 'topic', component: { template: '<div/>' }, props: true },
    ],
  })
}

async function mountView() {
  setActivePinia(createPinia())
  const router = makeRouter()
  await router.push({ name: 'browse-topics' })
  await router.isReady()
  const w = mount(TopicBrowseView, {
    global: { plugins: [i18n, router, createPinia()] },
  })
  await flushPromises()
  return { w, router }
}

afterEach(() => vi.restoreAllMocks())

describe('TopicBrowseView (#1261-6)', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getTrending').mockResolvedValue([
      {
        entity_id: 'topic:ai',
        kind: 'topic',
        label: 'Artificial Intelligence',
        velocity: 0.5,
        volume: 20,
        heating_up: true,
        total: 40,
        series: [],
      },
      {
        entity_id: 'topic:climate',
        kind: 'topic',
        label: 'Climate',
        velocity: 0.4,
        volume: 15,
        heating_up: false,
        total: 30,
        series: [],
      },
    ])
    vi.spyOn(api, 'getStorylines').mockResolvedValue([
      { id: 'thc:energy', label: 'Energy transition', size: 5, anchor_topic_id: 'topic:energy' },
    ])
  })

  it('lists trending topics with links to /topic/:id', async () => {
    const { w } = await mountView()
    expect(w.find('[data-testid="topic-browse-view"]').exists()).toBe(true)
    expect(w.text()).toContain('Artificial Intelligence')
    expect(w.text()).toContain('Climate')
    const links = w.findAll('a[href^="/topic/"]')
    expect(links.length).toBeGreaterThanOrEqual(2)
    expect(links[0].attributes('href')).toBe('/topic/topic:ai')
  })

  it('lists storylines linking to the anchor topic id', async () => {
    const { w } = await mountView()
    expect(w.text()).toContain('Energy transition')
    const storylineLink = w.findAll('a[href^="/topic/"]').find((a) => a.text().includes('Energy'))
    expect(storylineLink?.attributes('href')).toBe('/topic/topic:energy')
  })

  it('shows the empty message when both endpoints returned nothing', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue([])
    vi.spyOn(api, 'getStorylines').mockResolvedValue([])
    const { w } = await mountView()
    expect(w.text()).toContain('Nothing to browse yet')
  })

  it('shows the empty message when both endpoints rejected', async () => {
    vi.spyOn(api, 'getTrending').mockRejectedValue(new Error('offline'))
    vi.spyOn(api, 'getStorylines').mockRejectedValue(new Error('offline'))
    const { w } = await mountView()
    expect(w.text()).toContain('Nothing to browse yet')
  })
})
