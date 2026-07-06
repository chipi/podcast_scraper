import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, EpisodeSummary, FavoriteInsight, PlaybackPosition } from '../services/types'
import LibraryView from './LibraryView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'catalog', component: { template: '<div/>' } },
    { path: '/library', name: 'library', component: LibraryView },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
  ],
})

function summary(over: Partial<EpisodeSummary> = {}): EpisodeSummary {
  return {
    slug: 'fav-1', title: 'Saved Episode', feed_id: 'f', podcast_title: 'Show',
    publish_date: '2024-03-10', duration_seconds: 1800, episode_image_url: null,
    feed_image_url: null, artwork_url: null, status: 'ready', summary_preview: 'A recap.',
    summary_text: null, summary_bullets: [], topics: [], has_transcript: true,
    has_summary: false, has_gi: false, has_kg: false, has_bridge: false, ...over,
  }
}

function detail(over: Partial<EpisodeDetail> = {}): EpisodeDetail {
  return {
    slug: 'recent-1', title: 'Recently Played', feed_id: 'f', podcast_title: 'Show',
    publish_date: '2024-03-09', duration_seconds: 2400, episode_image_url: null,
    feed_image_url: null, artwork_url: null, summary_title: null, summary_bullets: [],
    summary_text: 'Heard recently.', has_transcript: true, has_summary: false,
    has_gi: false, has_kg: false, has_bridge: false, ...over,
  }
}

function insight(over: Partial<FavoriteInsight> = {}): FavoriteInsight {
  return {
    ref: 'ins-1', text: 'A grounded saved insight.', episode_slug: 'fav-1',
    podcast_title: 'Show', start_ms: 65_000, ...over,
  }
}

function tabButton(w: ReturnType<typeof mount>, label: string) {
  return w.findAll('button').find((b) => b.text() === label)!
}

beforeEach(() => {
  setActivePinia(createPinia())
  // QueueView (embedded) hydrates the queue; EpisodeCards embed FavoriteButton.
  vi.spyOn(api, 'getQueue').mockResolvedValue([])
  vi.spyOn(api, 'putQueue').mockResolvedValue()
  vi.spyOn(api, 'getFavorites').mockResolvedValue({ episodes: [], insights: [] })
  vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
  vi.spyOn(api, 'getEpisode').mockResolvedValue(detail())
  // HighlightsView (embedded) hydrates the capture store on mount.
  vi.spyOn(api, 'getHighlights').mockResolvedValue([])
  vi.spyOn(api, 'getNotes').mockResolvedValue([])
  // ResurfacingInbox (embedded) loads on mount.
  vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [], paused: false })
})
afterEach(() => vi.restoreAllMocks())

describe('LibraryView', () => {
  it('renders all tabs with their labels', async () => {
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const labels = w.findAll('button').map((b) => b.text())
    expect(labels).toContain('Saved')
    expect(labels).toContain('Highlights')
    expect(labels).toContain('Revisit')
    expect(labels).toContain('Queue')
    expect(labels).toContain('Recent')
    expect(labels).not.toContain('Knowledge') // merged into Saved as a section
  })

  it('Highlights tab shows the captured highlights (grouped) with an export link', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([
      {
        id: 'h1', episode_slug: 'fav-1', kind: 'span', start_ms: 65_000, end_ms: 68_000,
        char_start: 0, char_end: 5, segment_ids: ['s1'], quote_text: 'a captured line',
        speaker: null, source_insight_id: null, color: null, created_at: 1, anchor_status: null,
      },
    ])
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail({ slug: 'fav-1', title: 'Saved Episode' }))
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    await tabButton(w, 'Highlights').trigger('click')
    expect(w.text()).toContain('a captured line')
    expect(w.text()).toContain('1:05') // 65_000ms jump link
    const exportLink = w.findAll('a').find((a) => (a.attributes('href') ?? '').includes('export.md'))
    expect(exportLink).toBeTruthy()
  })

  it('Saved lists favorited episodes via EpisodeCard', async () => {
    vi.spyOn(api, 'getFavorites').mockResolvedValue({
      episodes: [summary({ slug: 'a', title: 'Alpha Saved' })],
      insights: [],
    })
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Alpha Saved')
    expect(w.findAll('a').map((a) => a.attributes('href'))).toContain('/episode/a')
  })

  it('Saved shows the empty state when there are no favorites at all', async () => {
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Nothing saved yet. Tap the heart on an episode or insight')
  })

  it('Saved shows saved insights in the Insights section (no separate tab) with a ?t= jump', async () => {
    vi.spyOn(api, 'getFavorites').mockResolvedValue({ episodes: [], insights: [insight()] })
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    // Saved is the default tab — insights render here under the "Insights" section, no tab switch.
    expect(w.text()).toContain('Insights') // section heading
    expect(w.text()).toContain('A grounded saved insight.')
    expect(w.text()).toContain('1:05') // 65_000ms → 1:05
    const link = w.findAll('a').find((a) => (a.attributes('href') ?? '').includes('/episode/fav-1'))!
    expect(link.attributes('href')).toContain('t=65')
  })

  it('Saved shows Episodes + Insights as separate sections when both are present', async () => {
    vi.spyOn(api, 'getFavorites').mockResolvedValue({
      episodes: [summary({ slug: 'a', title: 'Alpha Saved' })],
      insights: [insight()],
    })
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const headings = w.findAll('h2').map((h) => h.text())
    expect(headings).toContain('Episodes')
    expect(headings).toContain('Insights')
    expect(w.text()).toContain('Alpha Saved')
    expect(w.text()).toContain('A grounded saved insight.')
  })

  it('Recent renders an EpisodeCard per playback-history entry (hydrated from detail)', async () => {
    const positions: PlaybackPosition[] = [{ slug: 'recent-1', position_seconds: 30, updated_at: null }]
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue(positions)
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail({ slug: 'recent-1', title: 'Recently Played' }))
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    await tabButton(w, 'Recent').trigger('click')
    expect(w.text()).toContain('Recently Played')
    expect(w.findAll('a').map((a) => a.attributes('href'))).toContain('/episode/recent-1')
  })
})
