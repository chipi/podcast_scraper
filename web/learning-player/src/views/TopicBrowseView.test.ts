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
    global: { plugins: [i18n, router, createPinia()], stubs: { teleport: true } },
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

  it('renders trending topics as sparkline rows that open the topic page (#11)', async () => {
    const { w, router } = await mountView()
    expect(w.find('[data-testid="topic-browse-view"]').exists()).toBe(true)
    expect(w.text()).toContain('Artificial Intelligence')
    expect(w.text()).toContain('Climate')
    // #11 — trending now uses Home's sparkline treatment, sorted hottest-first, not a flat chip grid.
    expect(w.find('[data-testid="trend-sparks"]').exists()).toBe(true)
    const rows = w.findAll('[data-testid="trend-spark-row"]')
    expect(rows.length).toBeGreaterThanOrEqual(2)
    const push = vi.spyOn(router, 'push')
    await rows[0].trigger('click') // hottest first → topic:ai (v 0.5) leads topic:climate (v 0.4)
    expect(push).toHaveBeenCalledWith({ name: 'topic', params: { id: 'topic:ai' } })
  })

  it('offers a back-to-Home button when standalone (#13)', async () => {
    const { w } = await mountView()
    const back = w.find('[data-testid="browse-back-home"]')
    expect(back.exists()).toBe(true)
    expect(back.attributes('href')).toBe('/')
  })

  it('hides the heading + back-to-Home when embedded in the Browse hub', async () => {
    setActivePinia(createPinia())
    const router = makeRouter()
    await router.push({ name: 'browse-topics' })
    await router.isReady()
    const w = mount(TopicBrowseView, {
      props: { embedded: true },
      global: { plugins: [i18n, router, createPinia()] },
    })
    await flushPromises()
    expect(w.find('[data-testid="browse-back-home"]').exists()).toBe(false)
    expect(w.find('h1').exists()).toBe(false)
  })

  it('opens the storyline sheet (not the anchor topic) when a storyline is tapped (#9)', async () => {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue({
      id: 'topic:energy',
      label: 'Energy',
      cluster_id: null,
      cluster_label: null,
      cluster_size: 0,
      theme_cluster_id: 'thc:energy',
      theme_cluster_label: 'Energy transition',
      theme_cluster_size: 2,
      theme_sibling_topics: [
        { id: 'topic:grid', label: 'Grid', cluster_id: null, cluster_label: null, cluster_size: 0 },
      ],
      episode_count: 3,
      episodes: [],
      related_people: [],
    })
    const { w } = await mountView()
    expect(w.text()).toContain('Energy transition')
    await w.find('[data-testid="browse-storyline"]').trigger('click')
    await flushPromises()
    // The sheet opens, titled with the storyline — not a jump to the anchor topic page.
    const sheet = w.find('[data-testid="storyline-card"]')
    expect(sheet.exists()).toBe(true)
    expect(sheet.get('h2').text()).toBe('Energy transition')
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

  // Regression guard: the trending/theme-cluster endpoints cap at `limit ≤ 50` (server le=50,
  // app_discover.py). Requesting 60 returned 422, the `.catch(() => [])` swallowed it, and the tab
  // rendered empty on prod — indistinguishable in tests from a legitimately empty corpus, which is
  // why the committed e2e corpus (no temporal_velocity → empty trending) never caught it. Assert the
  // request stays within the bound so a future bump past 50 fails here instead of silently on-device.
  it('requests trending + storylines within the server limit bound (≤50)', async () => {
    await mountView()
    for (const call of vi.mocked(api.getTrending).mock.calls) {
      expect(
        call[2],
        `getTrending limit ${call[2]} exceeds the server le=50 cap → 422`
      ).toBeLessThanOrEqual(50)
    }
    for (const call of vi.mocked(api.getStorylines).mock.calls) {
      expect(
        call[0],
        `getStorylines limit ${call[0]} exceeds the server le=50 cap → 422`
      ).toBeLessThanOrEqual(50)
    }
  })
})
