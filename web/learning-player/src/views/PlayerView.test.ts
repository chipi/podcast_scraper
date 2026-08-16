import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, EpisodeStats, EpisodeSummary, Highlight } from '../services/types'
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
  vi.spyOn(api, 'getRelated').mockResolvedValue({
    items: [],
    page: 1,
    page_size: 6,
    total: 0,
    has_more: false,
  })
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
    // #1595 — insights moved OUT of the stats cluster into a labelled first-class control.
    expect(w.get('[data-testid="player-open-insights"]').text()).toContain('6 insights') // insights count from getInsights
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

  it('offers mark-moment to everyone as a teaser, and captures on tap once signed in (#1590)', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([])
    vi.spyOn(api, 'getNotes').mockResolvedValue([])
    const created: Highlight = {
      id: 'm1', episode_slug: 'ep-1', kind: 'moment', start_ms: 0, end_ms: null,
      char_start: null, char_end: null, segment_ids: [], quote_text: null, speaker: null,
      source_insight_id: null, color: null, created_at: 1, anchor_status: null,
    }
    const create = vi.spyOn(api, 'createHighlight').mockResolvedValue(created)
    const w = await mountPlayer('ep-1')
    // Signed out the control RENDERS — it used to be hidden, which hid the cheapest entry to the
    // learning loop from exactly the visitors deciding whether to sign up (#1590). It reads as a
    // teaser, and it does not claim a saved state.
    expect(w.find('[aria-label="Sign in to mark this moment"]').exists()).toBe(true)
    expect(w.find('[aria-label="Mark this moment"]').exists()).toBe(false)
    expect(create).not.toHaveBeenCalled()

    // Signed in → same control, real action.
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    auth.loaded = true
    await flushPromises()
    const mark = w.find('[aria-label="Mark this moment"]')
    expect(mark.exists()).toBe(true)
    await mark.trigger('click')
    await flushPromises()
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'moment', episode_slug: 'ep-1' }),
    )
  })

  // #1261-4: related-episodes rail
  it('renders the "More like this" rail when getRelated returns peers', async () => {
    const peer: EpisodeSummary = {
      slug: 'peer-1',
      title: 'Peer Episode One',
      feed_id: 'f',
      podcast_title: 'Peer Show',
      publish_date: '2024-02-01',
      duration_seconds: 1200,
      episode_image_url: null,
      feed_image_url: null,
      artwork_url: null,
      status: 'ready',
      summary_preview: null,
      summary_text: null,
      summary_bullets: [],
      topics: [],
      has_transcript: true,
      has_summary: false,
      has_gi: false,
      has_kg: false,
      has_bridge: false,
    }
    vi.spyOn(api, 'getRelated').mockResolvedValue({
      items: [peer],
      page: 1,
      page_size: 6,
      total: 1,
      has_more: false,
    })
    const w = await mountPlayer('ep-1')
    expect(w.find('[data-testid="related-episodes-rail"]').exists()).toBe(true)
    expect(w.text()).toContain('More like this')
    expect(w.text()).toContain('Peer Episode One')
    expect(api.getRelated).toHaveBeenCalledWith('ep-1', 6)
  })

  it('hides the rail entirely when getRelated returns no items', async () => {
    // Default beforeEach mock returns items: [] — the section should not render.
    const w = await mountPlayer('ep-1')
    expect(w.find('[data-testid="related-episodes-rail"]').exists()).toBe(false)
    expect(w.text()).not.toContain('More like this')
  })

  it('hides the rail when getRelated rejects — silent degrade, no listener-visible error', async () => {
    vi.spyOn(api, 'getRelated').mockRejectedValue(new Error('offline'))
    const w = await mountPlayer('ep-1')
    expect(w.find('[data-testid="related-episodes-rail"]').exists()).toBe(false)
  })

  it('announces a capture FAILURE rather than confirming a save that did not happen (S8)', async () => {
    // The store swallows write failures, and this announced "Marked" unconditionally — so on a
    // flaky connection a screen-reader user was told their highlight saved when nothing was
    // stored. A false confirmation is worse than silence: it stops them retrying.
    vi.spyOn(api, 'getHighlights').mockResolvedValue([])
    vi.spyOn(api, 'getNotes').mockResolvedValue([])
    vi.spyOn(api, 'createHighlight').mockRejectedValue(new Error('502'))

    const w = await mountPlayer('ep-1')
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    auth.loaded = true
    await flushPromises()

    await w.find('[aria-label="Mark this moment"]').trigger('click')
    await flushPromises()

    const live = w.find('[aria-live]')
    expect(live.exists()).toBe(true)
    expect(live.text()).toContain("Couldn't save that")
    expect(live.text()).not.toContain('Marked')
  })
})

describe('PlayerView — the summary is opened, not laid over the artwork', () => {
  it('shows a labelled control instead of the summary itself', async () => {
    // The old overlay put the full prose over the hero: hover-revealed on desktop, permanently on
    // for touch. So on a phone the artwork was covered by default, and the text was clipped to the
    // hero's fixed square — a real summary ended in an ellipsis you could not read past.
    const w = await mountPlayer()
    const opener = w.find('[data-testid="player-open-summary"]')
    expect(opener.exists()).toBe(true)
    // The prose is not rendered until asked for. Scoped to the dialog deliberately: the Knowledge
    // Panel has its own Summary section, so asserting the string is absent from the whole page
    // would be asserting something this change never claimed.
    expect(w.find('[data-testid="episode-summary-text"]').exists()).toBe(false)
  })

  it('opens the full summary on demand, with the headline above it', async () => {
    const w = await mountPlayer()
    await w.find('[data-testid="player-open-summary"]').trigger('click')
    await flushPromises()

    const body = w.find('[data-testid="episode-summary-text"]')
    expect(body.exists()).toBe(true)
    expect(body.text()).toContain('The pull-quote summary prose.')
    expect(w.find('[data-testid="episode-summary-dialog"]').text()).toContain('A title')
  })

  it('renders a long summary in full — no truncation, no ellipsis', async () => {
    // The point of the change: length must stop being a reason the reader cannot finish it.
    const long = 'Sentence. '.repeat(400).trim()
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail({ summary_text: long }))
    const w = await mountPlayer()
    await w.find('[data-testid="player-open-summary"]').trigger('click')
    await flushPromises()

    const text = w.find('[data-testid="episode-summary-text"]').text()
    expect(text.length).toBeGreaterThan(3000)
    expect(text.endsWith('Sentence.')).toBe(true)
    expect(text).not.toContain('…')
  })

  it('falls back to the headline when there is no prose body', async () => {
    vi.spyOn(api, 'getEpisode').mockResolvedValue(
      detail({ summary_text: '', summary_title: 'Only a headline' }),
    )
    const w = await mountPlayer()
    await w.find('[data-testid="player-open-summary"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="episode-summary-text"]').text()).toBe('Only a headline')
  })

  it('offers no control at all when the episode has no summary', async () => {
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail({ summary_text: '', summary_title: '' }))
    const w = await mountPlayer()
    expect(w.find('[data-testid="player-open-summary"]').exists()).toBe(false)
  })

  it('closes again', async () => {
    const w = await mountPlayer()
    await w.find('[data-testid="player-open-summary"]').trigger('click')
    await flushPromises()
    await w.find('[data-testid="episode-summary-close"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="episode-summary-text"]').exists()).toBe(false)
  })
})
