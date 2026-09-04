import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import { ApiError } from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, EpisodeStats, EpisodeSummary, Highlight } from '../services/types'
import { useAuthStore } from '../stores/auth'
import { clearPlayerViewCache } from './player-view-cache'
import PlayerView from './PlayerView.vue'
import { useDownloadsStore } from '../stores/downloads'

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

/**
 * Every PlayerView mounted by this file, torn down after each test.
 *
 * Not tidiness — correctness. A wrapper that is never unmounted keeps its watchers alive AND keeps
 * the pinia it was mounted with, so a later test's `router.push` fires the dead component's
 * route watchers against a stale (often signed-in) auth store. That surfaced as a signed-out test
 * seeing `markSurfaced` called twice by components belonging to two entirely different describe
 * blocks. Any test asserting "this was NOT called" is unreliable while zombies are listening.
 */
const mountedPlayers: Array<{ unmount: () => void }> = []
afterEach(() => {
  while (mountedPlayers.length) mountedPlayers.pop()!.unmount()
})

async function mountPlayer(slug = 'ep-1') {
  setActivePinia(createPinia())
  await router.push({ name: 'player', params: { slug } })
  await router.isReady()
  const w = mount(PlayerView, {
    props: { slug },
    global: { plugins: [i18n, router], stubs: { teleport: true } },
  })
  mountedPlayers.push(w)
  await flushPromises()
  return w
}

beforeEach(() => {
  clearPlayerViewCache() // #16 snapshot cache is module-scope; each test is a fresh app launch
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
  it('fetches per-episode reach on mount', async () => {
    await mountPlayer('ep-1')
    expect(api.getEpisodeStats).toHaveBeenCalledWith('ep-1')
  })

  it('no longer logs the listen itself — the playback path does (#1924)', async () => {
    // It moved to stores/player.ts via an injected logger, because the view never observed
    // auto-advance or the mini-player, so most real listening went unrecorded.
    await mountPlayer('ep-1')
    expect(api.logListen).not.toHaveBeenCalled()
  })

  it('reopening an episode paints instantly from the snapshot cache, no loading flash (#16)', async () => {
    // First open populates the module-scope snapshot cache.
    const first = await mountPlayer('ep-1')
    expect(first.text()).toContain('The Episode')
    first.unmount()
    mountedPlayers.length = 0

    // Reopen while the episode fetch HANGS — a cold load would show the loading text and no body.
    vi.spyOn(api, 'getEpisode').mockReturnValue(new Promise<EpisodeDetail>(() => {}))
    setActivePinia(createPinia())
    await router.push({ name: 'player', params: { slug: 'ep-1' } })
    await router.isReady()
    const w = mount(PlayerView, {
      props: { slug: 'ep-1' },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
    })
    mountedPlayers.push(w)
    await flushPromises()

    // Painted from cache despite the hung request — content shown, no loading state.
    expect(w.text()).toContain('The Episode')
    expect(w.text()).not.toContain(en.player.loading)
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
    // 404, a genuine refusal. 401 no longer belongs here: a dead session queues the capture
    // and announcing "Marked" is then TRUE (advisor 1.1).
    vi.spyOn(api, 'createHighlight').mockRejectedValue(new ApiError(404, 'gone'))

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

describe('a failure must not be reported as an absence (Player #6)', () => {
  it('says "not found" only for an actual 404', async () => {
    vi.spyOn(api, 'getEpisode').mockRejectedValue(new api.ApiError(404, 'nope'))
    const w = await mountPlayer()
    expect(w.text()).toContain(en.player.notFound)
    expect(w.find('[data-testid="player-retry"]').exists()).toBe(false)
  })

  it('offers a retry when the load failed for any other reason', async () => {
    // A dropped connection used to tell the user an episode that exists does not — a dead end, with
    // no reload prompt, for something that would work on the next tap.
    vi.spyOn(api, 'getEpisode').mockRejectedValue(new api.ApiError(500, 'boom'))
    const w = await mountPlayer()
    expect(w.text()).not.toContain(en.player.notFound)
    expect(w.text()).toContain(en.player.loadFailed)
    expect(w.find('[data-testid="player-retry"]').exists()).toBe(true)
  })

  it('a retry actually re-requests the episode', async () => {
    const get = vi.spyOn(api, 'getEpisode').mockRejectedValue(new api.ApiError(500, 'boom'))
    const w = await mountPlayer()
    get.mockResolvedValue(detail())
    await w.find('[data-testid="player-retry"]').trigger('click')
    await flushPromises()
    expect(w.text()).toContain('The Episode')
    expect(w.text()).not.toContain(en.player.loadFailed)
  })

  it('an absent transcript is "pending"; an unreadable one says so', async () => {
    // The route 500s on a segments file it cannot read. Collapsing that into the same "Transcript
    // pending — audio still plays" as a not-yet-written transcript meant a permanently broken
    // artifact read as "coming soon" forever, and nothing ever prompted anyone to look at it.
    vi.spyOn(api, 'getSegments').mockRejectedValue(new api.ApiError(404, 'no transcript'))
    let w = await mountPlayer()
    expect(w.find('[data-testid="player-transcript-empty"]').text()).toBe(
      en.player.transcriptPending,
    )

    vi.spyOn(api, 'getSegments').mockRejectedValue(new api.ApiError(500, 'unreadable'))
    w = await mountPlayer()
    expect(w.find('[data-testid="player-transcript-empty"]').text()).toBe(
      en.player.transcriptBroken,
    )
  })

  it('a request that never landed must not invent a "broken" transcript (#1906)', async () => {
    // A 404/500 is the server telling us something. A transport failure tells us nothing, and
    // reporting a broken artifact because the network dropped is the app lying about its own
    // data — offline, every episode would claim its transcript was corrupt.
    vi.spyOn(api, 'getSegments').mockRejectedValue(new TypeError('Failed to fetch'))
    const w = await mountPlayer()
    expect(w.find('[data-testid="player-transcript-empty"]').text()).not.toBe(
      en.player.transcriptBroken,
    )
  })
})

describe('arriving with ?revisit advances the spaced ladder (#35)', () => {
  // Marking on ARRIVAL rather than on click is what lets one mechanism serve all three surfaces:
  // the inbox jump link, the Your Week card and the digest email all just carry the marker. Before
  // this the only advance path in the product was the inbox's dismiss button, so anyone who
  // consumed revisit through Your Week or the email was re-sent the same five items every week.

  // Every mount is tracked and torn down. Not tidiness — the first version of these tests leaked
  // mounted PlayerViews, and a leaked instance still holds a `route.query.revisit` watcher plus
  // its OWN (signed-in) pinia. A later test's router.push then fired the dead component's watcher,
  // so "marks nothing when signed out" saw markSurfaced called once and the async-auth test saw it
  // four times — failures that had nothing to do with the code under test.
  async function mountAt(query: Record<string, string>, signedIn: boolean) {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    if (signedIn) {
      auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
      auth.loaded = true
    }
    await router.push({ name: 'player', params: { slug: 'ep-1' }, query })
    await router.isReady()
    const w = mount(PlayerView, {
      props: { slug: 'ep-1' },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
    })
    mountedPlayers.push(w)
    await flushPromises()
    return w
  }

  it('marks the highlight surfaced when the player is reached with ?revisit', async () => {
    const mark = vi.spyOn(api, 'markSurfaced').mockResolvedValue()
    await mountAt({ revisit: 'h1' }, true)
    expect(mark).toHaveBeenCalledWith('h1')
  })

  it('marks nothing on an ordinary visit', async () => {
    // Otherwise every episode open would consume a repetition of something.
    const mark = vi.spyOn(api, 'markSurfaced').mockResolvedValue()
    await mountAt({}, true)
    expect(mark).not.toHaveBeenCalled()
  })

  it('marks nothing when signed out', async () => {
    const mark = vi.spyOn(api, 'markSurfaced').mockResolvedValue()
    await mountAt({ revisit: 'h1' }, false)
    expect(mark).not.toHaveBeenCalled() // a 401 is not a review
  })

  it('still marks when auth resolves AFTER mount', async () => {
    // Auth hydration is async, so checking only at mount would silently drop the revisit of a user
    // who IS signed in but whose session had not loaded yet — the common case on a cold open from
    // an email link, which is exactly the path this feature exists to serve.
    const mark = vi.spyOn(api, 'markSurfaced').mockResolvedValue()
    await mountAt({ revisit: 'h9' }, false)
    expect(mark).not.toHaveBeenCalled()

    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    auth.loaded = true
    await flushPromises()
    expect(mark).toHaveBeenCalledWith('h9')
  })

  it('consumes one repetition per arrival, not one per auth change', async () => {
    const mark = vi.spyOn(api, 'markSurfaced').mockResolvedValue()
    await mountAt({ revisit: 'h1' }, true)
    const auth = useAuthStore()
    auth.user = null
    await flushPromises()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    await flushPromises()
    expect(mark).toHaveBeenCalledTimes(1)
  })

  it('a failed mark never surfaces as a player error', async () => {
    // Bookkeeping must not break playback. Failing to record just leaves the item due, which is
    // the safe direction: the user sees it again rather than losing it.
    vi.spyOn(api, 'markSurfaced').mockRejectedValue(new api.ApiError(500, 'nope'))
    const w = await mountAt({ revisit: 'h1' }, true)
    expect(w.text()).not.toContain(en.player.loadFailed)
  })

  // #1906 — the flagship scenario. getEpisode used to be the ONE call in the critical path with
  // no .catch(), so any transport failure aborted the whole load and a downloaded episode showed
  // the error screen instead of playing off the user's own disk.
  it('renders a downloaded episode from the registry when the network is gone', async () => {
    setActivePinia(createPinia())
    const downloads = useDownloadsStore()
    downloads.loaded = true
    downloads.entries = {
      'ep-1': {
        slug: 'ep-1',
        state: 'downloaded',
        updatedAt: 1,
        uri: 'file:///ep-1.mp3',
        path: 'offline-audio/anon/ep-1.mp3',
        title: 'Index Investing Without the Myths',
        showTitle: 'Long Horizon Notes',
        durationSeconds: 416,
      },
    }
    vi.spyOn(api, 'getEpisode').mockRejectedValue(new TypeError('Failed to fetch'))
    vi.spyOn(api, 'getAudioSource').mockRejectedValue(new TypeError('Failed to fetch'))
    vi.spyOn(api, 'getPlayback').mockRejectedValue(new TypeError('Failed to fetch'))

    await router.push({ name: 'player', params: { slug: 'ep-1' } })
    await router.isReady()
    const w = mount(PlayerView, {
      props: { slug: 'ep-1' },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
    })
    mountedPlayers.push(w)
    await flushPromises()

    // The registry carries this metadata precisely so this path can render with no API.
    expect(w.text()).toContain('Index Investing Without the Myths')
    expect(w.text()).toContain('Long Horizon Notes')
    // NOTE: this asserts the RENDER half of the fix. The src substitution itself runs through
    // localSourceFor(), which is isNative()-guarded and therefore always null under happy-dom —
    // that half is only observable on a device (#1908).
  })

  it('a real 404 still means not-found, even with the offline fallback in place', async () => {
    // The fallback must not swallow a genuine "this episode does not exist".
    vi.spyOn(api, 'getEpisode').mockRejectedValue(new api.ApiError(404, 'gone'))
    const w = await mountPlayer('ep-1')
    expect(w.text()).not.toContain('Index Investing Without the Myths')
  })

  // #1906 — a failed refresh must not delete what is already on screen. The keep-on-transport-error
  // fixes covered five surfaces; only the transcript one had a test.
  it('keeps a painted rail when a REVALIDATION drops the network', async () => {
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
    const first = await mountPlayer('ep-1')
    expect(first.text()).toContain('Peer Episode One')

    // Reopen the SAME episode (a #16 cache hit) with the network gone.
    vi.spyOn(api, 'getRelated').mockRejectedValue(new TypeError('Failed to fetch'))
    const second = await mountPlayer('ep-1')
    // The request told us nothing; emptying the rail would delete content the user is looking at.
    expect(second.text()).toContain('Peer Episode One')
  })

  it('still clears a rail when the SERVER answers that there is nothing', async () => {
    vi.spyOn(api, 'getRelated').mockResolvedValue({
      items: [
        {
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
        } as EpisodeSummary,
      ],
      page: 1,
      page_size: 6,
      total: 1,
      has_more: false,
    })
    await mountPlayer('ep-1')

    // A 500 IS information — the rail should go, unlike a dropped connection.
    vi.spyOn(api, 'getRelated').mockRejectedValue(new api.ApiError(500, 'boom'))
    const second = await mountPlayer('ep-1')
    expect(second.find('[data-testid="related-episodes-rail"]').exists()).toBe(false)
  })
})
