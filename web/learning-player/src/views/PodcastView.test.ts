import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import type { EpisodeSummary, LibraryItem, Podcast } from '../services/types'
import PodcastView from './PodcastView.vue'

const FEED = 'feed-1'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'catalog', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: PodcastView },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function show(): Podcast {
  return {
    feed_id: FEED,
    title: 'The Show',
    artwork_url: null,
    image_url: null,
    description: null,
    episode_count: 1,
  }
}

function ep(slug: string): EpisodeSummary {
  return {
    slug, title: 'An episode', feed_id: FEED, podcast_title: 'The Show',
    publish_date: '2024-01-01', duration_seconds: 1800, episode_image_url: null,
    feed_image_url: null, artwork_url: null, status: 'ready', summary_preview: 'recap',
    topics: [], has_transcript: true, has_summary: true, has_gi: false, has_kg: false,
    has_bridge: false,
  }
}

function libItem(feedId = FEED): LibraryItem {
  return { feed_id: feedId, feed_url: null, title: 'The Show', added_at: 1 }
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(api, 'getPodcasts').mockResolvedValue([show()])
  vi.spyOn(api, 'listPodcastEpisodes').mockResolvedValue({
    items: [ep('e-1')], page: 1, page_size: 20, total: 1, has_more: false,
  })
})
afterEach(() => vi.restoreAllMocks())

async function mountView(signedIn = true) {
  if (signedIn) useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
  const w = mount(PodcastView, {
    props: { feedId: FEED },
    global: {
      plugins: [i18n, router],
      stubs: { PodcastSignalsBand: true, ShowActivityChart: true, EpisodeCard: true },
    },
  })
  await flushPromises()
  return w
}

describe('PodcastView — follow show', () => {
  it('is hidden when signed out (feed subscriptions are per-user)', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
    const w = await mountView(false)
    expect(w.find('[data-testid="follow-show"]').exists()).toBe(false)
  })

  it('follows the show: POSTs the feed id and flips to Following', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
    const post = vi.spyOn(api, 'followShow').mockResolvedValue([libItem()])
    const w = await mountView()

    const btn = w.find('[data-testid="follow-show"]')
    expect(btn.text()).toContain('Follow show')
    expect(btn.attributes('aria-pressed')).toBe('false')

    await btn.trigger('click')
    await flushPromises()

    expect(post).toHaveBeenCalledWith(FEED, { title: 'The Show' })
    expect(w.find('[data-testid="follow-show"]').text()).toContain('Following')
    expect(w.find('[data-testid="follow-show"]').attributes('aria-pressed')).toBe('true')
  })

  it('renders Following when the show is already in the library', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([libItem()])
    const w = await mountView()
    expect(w.find('[data-testid="follow-show"]').text()).toContain('Following')
  })

  it('unfollows an already-followed show', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([libItem()])
    const del = vi.spyOn(api, 'unfollowShow').mockResolvedValue([])
    const w = await mountView()

    await w.find('[data-testid="follow-show"]').trigger('click')
    await flushPromises()

    expect(del).toHaveBeenCalledWith(FEED)
    expect(w.find('[data-testid="follow-show"]').text()).toContain('Follow show')
  })

  it('reverts the optimistic flip when the POST fails', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
    vi.spyOn(api, 'followShow').mockRejectedValue(new api.ApiError(401, 'nope'))
    const w = await mountView()

    await w.find('[data-testid="follow-show"]').trigger('click')
    await flushPromises()

    // The button must not claim a subscription the server refused.
    expect(w.find('[data-testid="follow-show"]').text()).toContain('Follow show')
    expect(w.find('[data-testid="follow-show"]').attributes('aria-pressed')).toBe('false')
  })
})
