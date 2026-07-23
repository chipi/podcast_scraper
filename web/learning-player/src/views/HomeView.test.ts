import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeSummary, Me, Podcast } from '../services/types'
import HomeView from './HomeView.vue'
import { useAuthStore } from '../stores/auth'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    { path: '/catalog', name: 'catalog', component: { template: '<div/>' } },
    { path: '/search', name: 'search', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/browse/topics', name: 'browse-topics', component: { template: '<div/>' } },
    { path: '/browse/people', name: 'browse-people', component: { template: '<div/>' } },
  ],
})

function ep(slug: string, title: string): EpisodeSummary {
  return {
    slug, title, feed_id: 'f', podcast_title: 'Show', publish_date: '2024-01-01',
    duration_seconds: 1800, episode_image_url: null, feed_image_url: null, artwork_url: null,
    status: 'ready', summary_preview: 'r', topics: [], has_transcript: true, has_summary: true,
    has_gi: false, has_kg: false, has_bridge: false,
  }
}

beforeEach(() => {
  setActivePinia(createPinia())
  // The embedded TrendingTopics + Storylines fetch corpus enrichment / theme clusters; keep these
  // tests off the network (their own coverage lives in TrendingTopics.test.ts / Storylines.test.ts).
  vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue({})
  vi.spyOn(api, 'getStorylines').mockResolvedValue([])
  vi.spyOn(api, 'getTrending').mockResolvedValue([])
  try {
    localStorage.removeItem('lp.interests.dismissed')
  } catch {
    /* happy-dom storage edge — ignore */
  }
})
afterEach(() => vi.restoreAllMocks())

function signIn(): void {
  useAuthStore().user = { user_id: 'u', email: 'e@x.com', name: 'N' } as unknown as Me
}

describe('HomeView (discover state, signed out)', () => {
  it('renders the ask hero, What\'s new and Your shows', async () => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({
      items: [ep('a-1', 'First Ep'), ep('a-2', 'Second Ep')], page: 1, page_size: 8, total: 2, has_more: false,
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([
      { feed_id: 'showa', title: 'Show A', artwork_url: null, image_url: null, episode_count: 2 } as Podcast,
    ])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([]) // no history → discover state

    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain("Find any moment you've heard.") // discover hero
    expect(w.text()).toContain("What's new")
    expect(w.text()).toContain('First Ep')
    expect(w.text()).toContain('All shows')
    expect(w.text()).toContain('Show A')
  })

  it('submitting the search navigates to /search', async () => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({ items: [], page: 1, page_size: 8, total: 0, has_more: false })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    const push = vi.spyOn(router, 'push')
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    await w.find('input#home-search').setValue('memory')
    await w.find('form').trigger('submit')
    expect(push).toHaveBeenCalledWith({ name: 'search', query: { q: 'memory' } })
  })
})

describe('HomeView interests card (3.5)', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({ items: [], page: 1, page_size: 8, total: 0, has_more: false })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
  })

  it('is hidden when signed out', async () => {
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).not.toContain('Personalize your Home')
  })

  it('shows to signed-in users and opens the cluster picker', async () => {
    vi.spyOn(api, 'getTopClusters').mockResolvedValue([{ id: 'tc:ai', label: 'AI', size: 3 }])
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router], stubs: { teleport: true } } })
    await flushPromises()
    expect(w.text()).toContain('Personalize your Home')
    await w.findAll('button').find((b) => b.text() === 'Choose interests')!.trigger('click')
    await flushPromises()
    expect(w.find('[role="dialog"]').exists()).toBe(true)
    expect(w.text()).toContain('AI') // a cluster chip in the picker
  })

  it('dismissing hides the card', async () => {
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    await w.findAll('button').find((b) => b.text() === 'Not now')!.trigger('click')
    expect(w.text()).not.toContain('Personalize your Home')
  })

  // #1261-9: browse-nav strip surfaces the standalone browse pages
  it('renders "Browse topics" and "Browse people" links to /browse/topics and /browse/people', async () => {
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const nav = w.get('[data-testid="home-browse-nav"]')
    const links = nav.findAll('a')
    const hrefs = links.map((a) => a.attributes('href'))
    expect(hrefs).toContain('/browse/topics')
    expect(hrefs).toContain('/browse/people')
    expect(nav.text()).toContain('Browse topics')
    expect(nav.text()).toContain('Browse people')
  })
})
