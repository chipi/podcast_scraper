import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, EpisodeStats, Highlight } from '../services/types'
import { useAuthStore } from '../stores/auth'
import PlayerView from './PlayerView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'catalog', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: PlayerView, props: true },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/search', name: 'search', component: { template: '<div/>' } },
  ],
})

function detail(over: Partial<EpisodeDetail> = {}): EpisodeDetail {
  return {
    slug: 'ep-1', title: 'The Episode', feed_id: 'f', podcast_title: 'The Show',
    publish_date: '2024-03-10', duration_seconds: 1800, episode_image_url: null,
    feed_image_url: null, artwork_url: null, summary_title: 'A title',
    summary_bullets: [], summary_text: 'The pull-quote summary prose.',
    has_transcript: true, has_summary: true, has_gi: true, has_kg: true, has_bridge: false, ...over,
  }
}

function epStats(over: Partial<EpisodeStats> = {}): EpisodeStats {
  return {
    slug: 'ep-1', listeners: 1200, opens: 3400, insights: 5,
    daily: [{ date: '2024-03-01', count: 3 }, { date: '2024-03-02', count: 5 }], ...over,
  }
}

async function mountPlayer(slug = 'ep-1') {
  setActivePinia(createPinia())
  await router.push({ name: 'player', params: { slug } })
  await router.isReady()
  const w = mount(PlayerView, {
    props: { slug },
    global: { plugins: [i18n, router], stubs: { teleport: true } },
  })
  await flushPromises()
  return w
}

beforeEach(() => {
  vi.spyOn(api, 'getEpisode').mockResolvedValue(detail())
  vi.spyOn(api, 'getSegments').mockResolvedValue({ version: '1', episode_slug: 'ep-1', segments: [] })
  vi.spyOn(api, 'getAudioSource').mockResolvedValue({
    episode_slug: 'ep-1', url: 'https://cdn/audio.mp3', mime: 'audio/mpeg',
    duration_seconds: 1800, media_id: null, strategy: 'direct', resolved_url: null,
    verified: null, content_length: null,
  })
  vi.spyOn(api, 'getPlayback').mockResolvedValue(null)
  vi.spyOn(api, 'getInsights').mockResolvedValue({ episode_slug: 'ep-1', insights: [] })
  vi.spyOn(api, 'getEntities').mockResolvedValue({
    episode_slug: 'ep-1', persons: [], orgs: [], topics: [],
  })
  vi.spyOn(api, 'getEpisodeStats').mockResolvedValue(epStats())
  vi.spyOn(api, 'logListen').mockResolvedValue()
  vi.spyOn(api, 'putPlayback').mockResolvedValue()
})
afterEach(() => vi.restoreAllMocks())

describe('PlayerView', () => {
  it('logs the listen and fetches per-episode reach on mount', async () => {
    await mountPlayer('ep-1')
    expect(api.logListen).toHaveBeenCalledWith('ep-1')
    expect(api.getEpisodeStats).toHaveBeenCalledWith('ep-1')
  })

  it('renders the per-episode reach cluster: listeners, opens (compacted) and the insights count', async () => {
    // insights: 6 grounded insights → the 💡 badge shows that count (from getInsights, not stats).
    vi.spyOn(api, 'getInsights').mockResolvedValue({
      episode_slug: 'ep-1',
      insights: Array.from({ length: 6 }, (_, i) => ({
        id: `i${i}`, text: `insight ${i}`, grounded: true, insight_type: null,
        confidence: null, position_hint: null, quotes: [],
      })),
    })
    const w = await mountPlayer('ep-1')
    // compact(): 1200 → "1.2k", 3400 → "3.4k".
    expect(w.text()).toContain('1.2k') // listeners
    expect(w.text()).toContain('3.4k') // opens
    expect(w.text()).toContain('💡 6') // insights count from getInsights
  })

  it('compacts large counts without a decimal at/above 10k', async () => {
    vi.spyOn(api, 'getEpisodeStats').mockResolvedValue(epStats({ opens: 12000, listeners: 50 }))
    const w = await mountPlayer('ep-1')
    expect(w.text()).toContain('12k') // 12000 → "12k" (no decimal ≥ 10000)
    expect(w.text()).toContain('50') // small listener count rendered as-is
  })

  it('renders the episode summary as the artwork pull-quote', async () => {
    const w = await mountPlayer('ep-1')
    expect(w.text()).toContain('The pull-quote summary prose.')
    expect(w.text()).toContain('The Episode') // title masthead
  })

  it('shows the mark-moment control only when signed in and captures on tap (P2)', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([])
    const created: Highlight = {
      id: 'm1', episode_slug: 'ep-1', kind: 'moment', start_ms: 0, end_ms: null,
      char_start: null, char_end: null, segment_ids: [], quote_text: null, speaker: null,
      source_insight_id: null, color: null, created_at: 1, anchor_status: null,
    }
    const create = vi.spyOn(api, 'createHighlight').mockResolvedValue(created)
    const w = await mountPlayer('ep-1')
    // signed out → no capture affordance
    expect(w.find('[aria-label="Mark this moment"]').exists()).toBe(false)
    // sign in → the control appears (auth-gated)
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    await flushPromises()
    const mark = w.find('[aria-label="Mark this moment"]')
    expect(mark.exists()).toBe(true)
    await mark.trigger('click')
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'moment', episode_slug: 'ep-1' }),
    )
  })
})
