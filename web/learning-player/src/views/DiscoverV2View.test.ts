/**
 * Discover 2's "Latest from shows you follow": the ▶ Moments door on each row follows the
 * episode's own has_moments — an insights file is not a moment to play (operator 2026-10-10).
 */
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { EpisodeSummary, Me } from '../services/types'
import { useAuthStore } from '../stores/auth'
import DiscoverV2View from './DiscoverV2View.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'discover2', component: stub },
    { path: '/browse', name: 'browse', component: stub },
    { path: '/person/:id', name: 'person', component: stub },
    { path: '/episode/:slug', name: 'player', component: stub },
    { path: '/:rest(.*)*', name: 'other', component: stub },
  ],
})

function ep(slug: string, over: Partial<EpisodeSummary>): EpisodeSummary {
  return {
    slug, title: `Title ${slug}`, feed_id: 'f', podcast_title: 'Show', publish_date: '2026-10-01',
    duration_seconds: 1800, episode_image_url: null, feed_image_url: null, artwork_url: null,
    status: 'ready', summary_preview: null, summary_text: null, summary_bullets: [], topics: [],
    has_transcript: true, has_summary: false, has_gi: true, has_kg: false, has_bridge: false,
    ...over,
  } as EpisodeSummary
}

beforeEach(() => {
  setActivePinia(createPinia())
  useAuthStore().user = { user_id: 'u', email: 'e@x.com', name: 'N' } as unknown as Me
  vi.spyOn(api, 'getLibrary').mockResolvedValue([{ feed_id: 'f' }] as never)
  vi.spyOn(api, 'getCompleted').mockResolvedValue([] as never)
  vi.spyOn(api, 'getDerivedInterests').mockResolvedValue([] as never)
  vi.spyOn(api, 'getHighlightsPage').mockResolvedValue({ items: [], total: 0 } as never)
  vi.spyOn(api, 'getTrendingEpisodes').mockResolvedValue([] as never)
  vi.spyOn(api, 'getUserInterests').mockResolvedValue([] as never)
  vi.spyOn(api, 'getTrending').mockResolvedValue([] as never)
  vi.spyOn(api, 'getPodcastsByIds').mockResolvedValue([] as never)
  vi.spyOn(api, 'searchCorpus').mockResolvedValue({ query: '', results: [] } as never)
})
afterEach(() => vi.restoreAllMocks())

describe('DiscoverV2View: Latest rows offer ▶ Moments only where there is a moment', () => {
  it('has_moments decides, not has_gi', async () => {
    vi.spyOn(api, 'listPodcastEpisodes').mockResolvedValue({
      items: [ep('with', { has_moments: true }), ep('without', { has_moments: false })],
      total: 2,
    } as never)
    const w = mount(DiscoverV2View, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const rows = w.get('[data-testid="d2-latest"]').findAll('li')
    expect(rows).toHaveLength(2)
    const door = (slug: string) =>
      rows.find((r) => r.text().includes(`Title ${slug}`))!.find('[data-testid="moments-link"]').exists()
    expect(door('with')).toBe(true)
    expect(door('without')).toBe(false)
  })
})
