import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
const readCached = vi.fn(async (_k: string): Promise<unknown> => null)
const writeCached = vi.fn(async (_k: string, _v: unknown): Promise<void> => {})
vi.mock('../services/contentCache', () => ({
  isArrayCache: (v: unknown) => Array.isArray(v),
  hasArrayFields:
    (...f: string[]) =>
    (v: unknown) =>
      typeof v === 'object' &&
      v !== null &&
      !Array.isArray(v) &&
      f.every((k) => Array.isArray((v as Record<string, unknown>)[k])),
  readCached: (k: string) => readCached(k),
  writeCached: (k: string, v: unknown) => writeCached(k, v),
}))

import TopicBrowseView from './TopicBrowseView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'home', component: { template: '<div/>' } },
      { path: '/browse/topics', name: 'browse-topics', component: TopicBrowseView },
      { path: '/topic/:id', name: 'topic', component: { template: '<div/>' }, props: true },
      { path: '/storyline/:id', name: 'storyline', component: { template: '<div/>' }, props: true },
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

  it('navigates to the storyline page (keyed by anchor topic) when a storyline is tapped (F4.5)', async () => {
    const { w, router } = await mountView()
    const push = vi.spyOn(router, 'push')
    expect(w.text()).toContain('Energy transition')
    await w.find('[data-testid="browse-storyline"]').trigger('click')
    expect(push).toHaveBeenCalledWith({ name: 'storyline', params: { id: 'topic:energy' } })
  })

  it('shows the empty message when both endpoints returned nothing', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue([])
    vi.spyOn(api, 'getStorylines').mockResolvedValue([])
    const { w } = await mountView()
    expect(w.text()).toContain('Nothing trending over this window yet')
  })

  it('shows the empty message when both endpoints rejected', async () => {
    vi.spyOn(api, 'getTrending').mockRejectedValue(new Error('offline'))
    vi.spyOn(api, 'getStorylines').mockRejectedValue(new Error('offline'))
    const { w } = await mountView()
    expect(w.text()).toContain('Nothing trending over this window yet')
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

  // RFC-103 R2 — default window is 3m, and switching the segmented control refetches for the pick.
  it('defaults to the 3m window and refetches when the window changes', async () => {
    const { w } = await mountView()
    expect(vi.mocked(api.getTrending).mock.calls[0]?.[3]).toBe('3m') // 4th arg is the window
    await w.get('[data-testid="trend-window-6m"]').trigger('click')
    await flushPromises()
    expect(api.getTrending).toHaveBeenCalledWith('topic', 'corpus', 50, '6m')
  })

  /**
   * `.catch(() => [])` collapsed a FAILURE into emptiness, so offline this tab rendered as a corpus
   * with no topics rather than as a page we could not load — #1591's defect, one tab over (#1909).
   */
  describe('offline (#1909)', () => {
    beforeEach(() => {
      readCached.mockReset().mockResolvedValue(null)
      writeCached.mockReset().mockResolvedValue(undefined)
    })

    it('shows what it last loaded instead of an empty corpus', async () => {
      readCached.mockResolvedValue([{ id: 'x:1', label: 'From Last Time', count: 3 }])
      vi.spyOn(api, 'getTrending').mockRejectedValue(new Error('offline'))
      const { w } = await mountView()
      await flushPromises()
      await flushPromises()
      expect(w.text(), 'the cached rows are gone').toContain('From Last Time')
      expect(w.find('[data-testid="browse-stale-topics"]').exists()).toBe(true)
    })

    it('snapshots a successful load, keyed by window', async () => {
      const { w } = await mountView()
      await flushPromises()
      // Keyed by SCOPE + WINDOW (#2030): the "mine" and corpus lenses never share a cache slot.
      expect(writeCached.mock.calls.map((c) => c[0])).toContain('browse.topics.corpus.3m')
      expect(w.find('[data-testid="browse-stale-topics"]').exists()).toBe(false)
    })

    it('is empty, not stale, when there is nothing cached either', async () => {
      vi.spyOn(api, 'getTrending').mockRejectedValue(new Error('offline'))
      const { w } = await mountView()
      await flushPromises()
      await flushPromises()
      expect(w.find('[data-testid="browse-stale-topics"]').exists()).toBe(false)
    })
  })
})
