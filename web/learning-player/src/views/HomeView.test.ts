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
  it('renders the ask hero and What\'s new, but no shows section', async () => {
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
    // "Your shows" is per-user (#1585): signed out there are no follows, so no section — and
    // crucially it must NOT fall back to showing the whole catalogue, which is what it used to do.
    expect(w.text()).not.toContain('Your shows')
    expect(w.text()).not.toContain('Show A')
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

describe('HomeView distinguishes empty from broken (#1591)', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
  })

  it('renders an error with a retry when the fetch fails', async () => {
    // The defect: every section did `.catch(() => [])` and then hid itself when empty, so a total
    // API outage rendered the same page as a brand-new account — a hero, a search box, two chips.
    vi.spyOn(api, 'getDiscover').mockRejectedValue(new Error('boom'))
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()

    expect(w.find('[data-testid="section-error"]').exists()).toBe(true)
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
    // The section header still renders — it is what tells you this content exists at all.
    expect(w.text()).toContain("What's new")
  })

  it('retry re-fetches and recovers', async () => {
    // No error state anywhere in the app previously offered a retry: the only move was a reload.
    const spy = vi.spyOn(api, 'getDiscover').mockRejectedValueOnce(new Error('boom'))
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.find('[data-testid="section-error"]').exists()).toBe(true)

    spy.mockResolvedValueOnce({
      items: [ep('a-1', 'Recovered Ep')], page: 1, page_size: 8, total: 1, has_more: false,
    })
    await w.get('[data-testid="section-retry"]').trigger('click')
    await flushPromises()

    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
    expect(w.text()).toContain('Recovered Ep')
  })

  it('a successful-but-empty load still hides the section', async () => {
    // Hide when the SYSTEM is empty — there is no action the user can take, so an empty shell is
    // noise. Contrast "Your shows", which is empty because of a user action not yet taken.
    vi.spyOn(api, 'getDiscover').mockResolvedValue({
      items: [], page: 1, page_size: 8, total: 0, has_more: false,
    })
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()

    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
    expect(w.text()).not.toContain("What's new")
  })
})

describe('HomeView "Your shows" is your follows, not the catalogue (#1585)', () => {
  const catalogue = [
    { feed_id: 'showa', title: 'Show A', artwork_url: null, image_url: null, episode_count: 2 } as Podcast,
    { feed_id: 'showb', title: 'Show B', artwork_url: null, image_url: null, episode_count: 5 } as Podcast,
  ]

  beforeEach(() => {
    vi.spyOn(api, 'getDiscover').mockResolvedValue({ items: [], page: 1, page_size: 8, total: 0, has_more: false })
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    vi.spyOn(api, 'getPodcasts').mockResolvedValue(catalogue)
  })

  it('renders only the followed shows, joined to catalogue artwork', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([
      { feed_id: 'showb', feed_url: null, title: 'Show B', added_at: 1 },
    ])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Your shows')
    expect(w.text()).toContain('Show B')
    // Show A is in the corpus but NOT followed. Before #1585 this section rendered the whole
    // catalogue while calling itself "Your shows".
    expect(w.text()).not.toContain('Show A')
  })

  it('offers the action, not just a description of it, when you follow nothing', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    // A section that silently self-hides can't tell a new user the feature exists — but an empty
    // state that only *describes* following is barely better, since it sends you off to a show page
    // to find the control. Suggested shows carry the follow control itself.
    expect(w.text()).toContain('Your shows')
    expect(w.text()).toContain('Follow a show')
    expect(w.findAll('[aria-pressed]').length).toBeGreaterThan(0)
  })

  it('following from the empty state moves the show into the grid, in place', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
    vi.spyOn(api, 'followShow').mockResolvedValue([
      { feed_id: 'showa', feed_url: null, title: 'Show A', added_at: 1 },
    ])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Follow a show') // empty state

    await w.get('[aria-pressed]').trigger('click')
    await flushPromises()

    // The whole point of putting the control here: no navigation, no reload.
    expect(w.text()).not.toContain('Follow a show')
    expect(w.text()).toContain('Show A')
  })

  it('still renders a followed feed that is absent from the catalogue', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([
      { feed_id: 'gone', feed_url: null, title: 'Departed Show', added_at: 1 },
    ])
    signIn()
    const w = mount(HomeView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Departed Show')
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
