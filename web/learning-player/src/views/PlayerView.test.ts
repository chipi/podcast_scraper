import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import playerViewSource from './PlayerView.vue?raw'
import { ApiError } from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, EpisodeStats, EpisodeSummary, Highlight } from '../services/types'
import { useAuthStore } from '../stores/auth'
import { clearPlayerViewCache, getPlayerViewSnapshot } from './player-view-cache'
// Only `localSourceFor` is faked — the rest of the module stays real, so `localArtworkFor` and
// `localTranscriptFor` behave exactly as they do in every other test here (null, off-native).
const localSourceFor = vi.fn((_slug: string): string | null => null)
const localPosition = vi.fn((_slug: string): { seconds: number; finished: boolean; updatedAt: number } | null => null)
vi.mock('../services/downloads', async (orig) => ({
  ...(await orig<typeof import('../services/downloads')>()),
  localSourceFor: (slug: string) => localSourceFor(slug),
}))
vi.mock('../services/playbackPositions', async (orig) => ({
  ...(await orig<typeof import('../services/playbackPositions')>()),
  localPosition: (slug: string) => localPosition(slug),
}))

import { usePlayerStore } from '../stores/player'
import PlayerView from './PlayerView.vue'
import KnowledgePanel from '../components/KnowledgePanel.vue'
import EpisodeDescriptionSheet from '../components/EpisodeDescriptionSheet.vue'
import { useDownloadsStore } from '../stores/downloads'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'catalog', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: PlayerView, props: true },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/search', name: 'search', component: { template: '<div/>' } },
    // The capture receipt links here (#1592). Without the route registered, RouterLink's
    // setup throws and the failure surfaces as an unmount TypeError, not a missing route.
    { path: '/library', name: 'library', component: { template: '<div/>' } },
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
  vi.spyOn(api, 'logListen').mockResolvedValue(true)
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
    // "Brief", not "Insights" and not "Notes" (operator 2026-10-10): the panel is the episode's
    // brief (summary, key points, people, downloads); "Notes" alone already means the user's own.
    // The items inside stay insights. PL.5: the opener carries the label only, no count (the count
    // lives on the panel's Insights section header, UXS-014).
    expect(w.get('[data-testid="player-open-insights"]').text()).toBe('Brief')
    expect(w.get('[data-testid="player-open-insights"]').text()).not.toMatch(/\d/)
  })

  it('marks the open count with a chart glyph, not ▶ — the chip is not a play control (operator 2026-10-05)', async () => {
    const w = await mountPlayer('ep-1')
    const reach = w.get('[data-testid="player-reach"]')
    expect(reach.find('[data-testid="player-reach-opens-icon"]').exists()).toBe(true)
    expect(reach.text()).not.toContain('▶')
  })

  it('compacts large counts without a decimal at/above 10k', async () => {
    vi.spyOn(api, 'getEpisodeStats').mockResolvedValue(epStats({ opens: 12000, listeners: 50 }))
    const w = await mountPlayer('ep-1')
    expect(w.text()).toContain('12k') // 12000 → "12k" (no decimal ≥ 10000)
    expect(w.text()).toContain('50') // small listener count rendered as-is
  })

  it('renders the episode title masthead', async () => {
    const w = await mountPlayer('ep-1')
    expect(w.text()).toContain('The Episode')
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

  it('shows a FAILED capture visibly, not only to screen readers (#1592)', async () => {
    // The bug. `markMoment` announced `capture.saveFailed` into the sr-only live region and
    // returned before setting any visual state, so a sighted user could not distinguish a failed
    // save from a tap that missed. The reasoning was right — a false confirmation is worse than
    // silence — and the fix reached screen readers only.
    vi.spyOn(api, 'getHighlights').mockResolvedValue([])
    vi.spyOn(api, 'getNotes').mockResolvedValue([])
    // A PERMANENT refusal (422), not a generic Error. A dead socket or a 502 is not an answer:
    // the store queues those for replay and correctly reports success (#1925), so rejecting with a
    // plain Error tests the offline path, not the failure path. Learned by writing it wrong.
    vi.spyOn(api, 'createHighlight').mockRejectedValue(new api.ApiError(422, 'refused'))

    const w = await mountPlayer('ep-1')
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    auth.loaded = true
    await flushPromises()

    await w.find('[data-testid="capture-moment"]').trigger('click')
    await flushPromises()

    const control = w.find('[data-testid="capture-moment"]')
    expect(control.text()).toContain("Couldn't save that")
    // And it must NOT claim success: no followable receipt for something that was not stored.
    expect(w.find('[data-testid="capture-receipt"]').exists()).toBe(false)
  })

  it('a successful capture offers a route to where it went (#1592)', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([])
    vi.spyOn(api, 'getNotes').mockResolvedValue([])
    vi.spyOn(api, 'createHighlight').mockResolvedValue({
      id: 'm1', episode_slug: 'ep-1', kind: 'moment', start_ms: 0, end_ms: null,
      char_start: null, char_end: null, segment_ids: [], quote_text: null, speaker: null,
      source_insight_id: null, color: null, created_at: 1, anchor_status: null,
    } as Highlight)

    const w = await mountPlayer('ep-1')
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    auth.loaded = true
    await flushPromises()

    await w.find('[data-testid="capture-moment"]').trigger('click')
    await flushPromises()

    const receipt = w.find('[data-testid="capture-receipt"]')
    expect(receipt.exists()).toBe(true)
    expect(receipt.attributes('href')).toBe('/library?tab=saved')
  })

  describe('the transport pays the notch inset only while pinned (#2004 item 6)', () => {
    it('renders unpinned padding at rest, with a sentinel to detect pinning', async () => {
      // At the top of the page the transport sits under the artwork and is NOT stuck, so the
      // safe-area inset earns nothing there — it was ~75px of dead band between artwork and player.
      vi.spyOn(api, 'getHighlights').mockResolvedValue([])
      vi.spyOn(api, 'getNotes').mockResolvedValue([])
      const w = await mountPlayer('ep-1')
      const el = w.find('[data-testid="player-controls-sticky"]')
      expect(el.exists()).toBe(true)
      expect(el.attributes('data-stuck')).toBe('false')
      // At rest: no top padding (2026-09-30) — the artwork→panel gap is `mt-2` alone, so the
      // scrubber and timestamps sit higher on a phone. The inset below still applies once pinned.
      expect(el.classes()).toContain('pt-0')
      expect(el.classes().join(' ')).not.toContain('safe-area-inset-top')
    })

    it('APPLIES the notch inset once it is actually pinned', async () => {
      // Was a source-regex for the ternary, which would pass with the behaviour deleted. This drives
      // the observer: capture the callback, report the sentinel off-screen, assert the class flips.
      const callbacks: IntersectionObserverCallback[] = []
      vi.stubGlobal(
        'IntersectionObserver',
        class {
          constructor(cb: IntersectionObserverCallback) {
            callbacks.push(cb)
          }
          observe() {}
          disconnect() {}
        },
      )
      vi.spyOn(api, 'getHighlights').mockResolvedValue([])
      vi.spyOn(api, 'getNotes').mockResolvedValue([])
      const w = await mountPlayer('ep-1')

      const el = () => w.find('[data-testid="player-controls-sticky"]')
      expect(el().attributes('data-stuck')).toBe('false')
      expect(el().classes()).toContain('pt-0')

      // sentinel leaves the viewport → the transport is pinned
      callbacks.at(-1)?.([{ isIntersecting: false } as IntersectionObserverEntry], {} as IntersectionObserver)
      await flushPromises()
      expect(el().attributes('data-stuck')).toBe('true')
      expect(el().classes().join(' ')).toContain('safe-area-inset-top')
      expect(el().classes()).not.toContain('pt-2')

      vi.unstubAllGlobals()
    })
  })

  it('offers add-to-collection for THIS episode (#2013 follow-up)', async () => {
    // It existed on browse rows and on the show page, but not here — so you could pin an episode
    // from a list, or pin a whole show, but not the episode you were listening to.
    vi.spyOn(api, 'getHighlights').mockResolvedValue([])
    vi.spyOn(api, 'getNotes').mockResolvedValue([])
    const w = await mountPlayer('ep-1')
    expect(w.find('[data-testid="add-to-collection"]').exists()).toBe(true)
    // and it pins the EPISODE, not the show — the show page already has its own control
    expect(playerViewSource).toContain("{ kind: 'episode', ref: props.slug }")
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
    const rail = w.get('[data-testid="related-episodes-rail"]')
    // Scoped to the rail: the Knowledge Panel carries its own "More like this", so a page-wide text
    // match passed while this heading rendered as an unresolved <sectionheading> with no text.
    expect(rail.get('[data-testid="section-title"]').text()).toBe('More like this')
    // The standard rail slot (operator 2026-10-05), not a width of its own.
    expect(rail.get('li').classes()).toContain('lp-rail-item')
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
  // Marking on ARRIVAL rather than on click is what lets one mechanism serve every link that carries
  // the marker — the inbox jump link and the digest email (Home's Your Week card did too until
  // 2026-09-30, when Home stopped showing the digest's revisit section). Before this the only advance
  // path in the product was the inbox's dismiss button, so anyone who consumed revisit through a
  // link was re-sent the same five items every week.

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

  it('a revisit consumed here re-reads the shared Revisit count (2026-10-09)', async () => {
    vi.spyOn(api, 'markSurfaced').mockResolvedValue()
    const due = vi.spyOn(api, 'getResurfacingPage')
    await mountAt({ revisit: 'h1' }, true)
    await flushPromises()
    expect(due, 'the Revisit count and list were left counting the reviewed highlight').toHaveBeenCalled()
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

/**
 * A DOWNLOADED episode carries everything this view needs to render and play. Until now
 * `offlineEpisodeDetail` was consulted only when a fetch FAILED — never when it was merely slow —
 * so opening one on a working-but-slow network sat behind a spinner waiting for three round-trips
 * for data already on the disk. That is the "loading, loading" the operator reported.
 */
describe('a downloaded episode paints from disk, not from the network', () => {
  const SLUG = 'disk-ep'

  function markDownloaded(): void {
    localSourceFor.mockReturnValue('capacitor-file:///audio/disk-ep.mp3')
    const d = useDownloadsStore()
    d.entries[SLUG] = {
      slug: SLUG,
      state: 'downloaded',
      updatedAt: 1,
      uri: 'file:///audio/disk-ep.mp3',
      path: 'offline-audio/anon/disk-ep.mp3',
      title: 'Title From The Registry',
      showTitle: 'Disk Show',
      feedId: 'f1',
      durationSeconds: 100,
    } as never
  }

  /** A request that never answers — a slow network, not a broken one. */
  const hangs = <T,>() => new Promise<T>(() => {})

  beforeEach(() => {
    localSourceFor.mockReset().mockReturnValue(null)
    localPosition.mockReset().mockReturnValue(null)
  })

  it('renders and arms playback while the network is still hanging', async () => {
    vi.spyOn(api, 'getEpisode').mockImplementation(hangs)
    vi.spyOn(api, 'getAudioSource').mockImplementation(hangs)
    vi.spyOn(api, 'getPlayback').mockImplementation(hangs)
    const { w, player } = await mountDownloaded()

    expect(w.text(), 'the page waited on the network for data already on disk').toContain(
      'Title From The Registry',
    )
    expect(player.currentSlug, 'playback was not armed from the local file').toBe(SLUG)
    expect(w.find('[data-testid="player-loading"]').exists()).toBe(false)
  })

  it('starts from the position THIS device recorded', async () => {
    vi.spyOn(api, 'getEpisode').mockImplementation(hangs)
    vi.spyOn(api, 'getAudioSource').mockImplementation(hangs)
    vi.spyOn(api, 'getPlayback').mockImplementation(hangs)
    localPosition.mockReturnValue({ seconds: 42, finished: false, updatedAt: Date.now() })
    const { player } = await mountDownloaded()

    // The element only applies a start position once it knows a duration.
    player.duration = 100
    await flushPromises()
    expect(Math.round(player.el?.currentTime ?? 0)).toBe(42)
  })

  /**
   * `?play=1` — "Resume" on Home means resume, not "open the page about this episode".
   *
   * The intent rides the URL rather than Home reaching into the player store: the hero does not
   * hold the audio url, and a play() fired at click time would land before this view has applied
   * the saved position.
   */
  describe('the play intent', () => {
    async function mountWithQuery(query: Record<string, string>) {
      setActivePinia(createPinia())
      markDownloaded()
      localPosition.mockReturnValue({ seconds: 42, finished: false, updatedAt: Date.now() })
      await router.push({ name: 'player', params: { slug: SLUG }, query })
      await router.isReady()
      /**
       * Spy BEFORE mounting (operator 2026-09-19).
       *
       * This used to mount, then attach the spy, then set `duration = 100` to stand in for the
       * element working the length out late — the play intent waits on a known duration, so that
       * ordering was what let the spy see it. A downloaded episode now carries its length from the
       * registry into `player.load()`, so the duration is known DURING mount and the intent fires
       * there: exactly the point of the change (offline, the element often never reports one at
       * all). The spy has to exist before mount or it watches for a call that has already happened.
       */
      const player = usePlayerStore()
      const play = vi.spyOn(player, 'play').mockImplementation(() => {})
      const w = mount(PlayerView, {
        props: { slug: SLUG },
        global: { plugins: [i18n, router], stubs: { teleport: true } },
      })
      mountedPlayers.push(w)
      await flushPromises()
      // Still asserted, for the case the registry had no duration: the element reporting one must
      // drive the intent exactly as before.
      player.duration = 100
      await flushPromises()
      return { player, play }
    }

    beforeEach(() => {
      vi.spyOn(api, 'getEpisode').mockImplementation(hangs)
      vi.spyOn(api, 'getAudioSource').mockImplementation(hangs)
      vi.spyOn(api, 'getPlayback').mockImplementation(hangs)
    })

    it('starts playing when the caller asked to resume', async () => {
      const { play } = await mountWithQuery({ play: '1' })
      expect(play, 'arriving with ?play=1 left the episode paused').toHaveBeenCalled()
    })

    it('does NOT play without it — a notification opens an episode, it does not start audio', async () => {
      // The operator's rule for notifications (2026-09-18): tapping "new episode in a show you
      // follow" lands on the episode so you can look at it. Audio nobody asked for is a bug.
      const { play } = await mountWithQuery({})
      expect(play, 'opening an episode started audio unprompted').not.toHaveBeenCalled()
    })

    it("the notes panel's ▶ Play from seeks AND plays — an explicit ▶ starts audio (operator 2026-10-05)", async () => {
      // `?notes=1` opens the panel: it mounts on first open (2026-10-08), not behind the closed sheet.
      const { player, play } = await mountWithQuery({ notes: '1' })
      const panel = (mountedPlayers.at(-1) as ReturnType<typeof mount>).findComponent({
        name: 'KnowledgePanel',
      })
      expect(panel.exists(), 'the notes panel is not mounted on the player page').toBe(true)
      play.mockClear()
      panel.vm.$emit('play-from', 30)
      await flushPromises()
      expect(play, '▶ Play from left the episode paused').toHaveBeenCalled()
      expect(Math.round(player.el?.currentTime ?? 0)).toBe(30)
    })

    it('the notes panel is not built until it is first opened (2026-10-08)', async () => {
      // A closed <dialog> still renders its children: every episode built key points, insights, the
      // related rail and its artwork behind a sheet that might never open.
      await mountWithQuery({})
      const w = mountedPlayers.at(-1) as ReturnType<typeof mount>
      expect(w.findComponent({ name: 'KnowledgePanel' }).exists()).toBe(false)
      await router.push({ name: 'player', params: { slug: SLUG }, query: { notes: '1' } })
      await flushPromises()
      expect(w.findComponent({ name: 'KnowledgePanel' }).exists()).toBe(true)
    })

    it('▶ Play from a link to the episode ALREADY OPEN still seeks and plays (operator 2026-10-05)', async () => {
      /*
       * The start position applies once per episode, on load. A link to the SAME episode reuses this
       * view, so nothing re-applied it: on the player page, a topic card's "▶ Play from 0:30" for
       * this very episode left it at 0:42, paused. Found by probing, not by a report.
       */
      const { player, play } = await mountWithQuery({})
      play.mockClear()
      await router.push({ name: 'player', params: { slug: SLUG }, query: { t: '30', play: '1' } })
      await flushPromises()
      expect(Math.round(player.el?.currentTime ?? -1), 'the moment was not applied').toBe(30)
      expect(play, '▶ Play from left the episode paused').toHaveBeenCalledTimes(1)
    })

    it('a same-episode link WITHOUT ?play=1 seeks but does not start audio', async () => {
      const { player, play } = await mountWithQuery({})
      play.mockClear()
      await router.push({ name: 'player', params: { slug: SLUG }, query: { t: '30' } })
      await flushPromises()
      expect(Math.round(player.el?.currentTime ?? -1)).toBe(30)
      expect(play).not.toHaveBeenCalled()
    })

    it('plays from the resumed position, not from zero', async () => {
      // Playing before the seek is audible — a second or two of 0:00 before it jumps.
      const { player, play } = await mountWithQuery({ play: '1' })
      expect(play).toHaveBeenCalled()
      expect(Math.round(player.el?.currentTime ?? 0), 'play started before the seek').toBe(42)
    })
  })

  /**
   * `mountPlayer` activates its own pinia, so the registry has to be seeded between that and the
   * mount — a store written before it is replaced is a store the component never sees.
   */
  async function mountDownloaded() {
    setActivePinia(createPinia())
    markDownloaded()
    await router.push({ name: 'player', params: { slug: SLUG } })
    await router.isReady()
    const w = mount(PlayerView, {
      props: { slug: SLUG },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
    })
    mountedPlayers.push(w)
    await flushPromises()
    return { w, player: usePlayerStore() }
  }

  /**
   * The fast path starts a downloaded episode from the position THIS device recorded, because the
   * server has not answered yet. #1925's reconciliation stays the authority: if the server's turns
   * out to be newer, the start point is corrected — but only while the correction is invisible.
   */
  describe('the corrective seek', () => {
    /** getPlayback held open so the duration can land BEFORE the server position does. */
    function deferredPlayback() {
      let release!: (v: unknown) => void
      vi.spyOn(api, 'getEpisode').mockResolvedValue(detail())
      vi.spyOn(api, 'getAudioSource').mockResolvedValue({
        url: 'https://origin.example/a.mp3',
      } as never)
      vi.spyOn(api, 'getPlayback').mockReturnValue(
        new Promise((r) => (release = r as (v: unknown) => void)) as never,
      )
      localPosition.mockReturnValue({ seconds: 10, finished: false, updatedAt: 1000 })
      return { release: (v: unknown) => release(v) }
    }

    const newerServer = { position_seconds: 200, finished: false, updated_at: 1788900000 }

    it('corrects to the server position when it is newer and nothing has started', async () => {
      const { release } = deferredPlayback()
      const { player } = await mountDownloaded()
      player.duration = 600
      await flushPromises()
      expect(Math.round(player.el?.currentTime ?? 0), 'the device position was not applied').toBe(10)

      release(newerServer)
      await flushPromises()
      expect(Math.round(player.el?.currentTime ?? 0), 'the newer server position was ignored').toBe(
        200,
      )
    })

    it('leaves it alone once playback has started', async () => {
      // Correcting under someone who pressed play is an audible jump in what they are listening to.
      const { release } = deferredPlayback()
      const { player } = await mountDownloaded()
      player.duration = 600
      await flushPromises()
      player.playing = true

      release(newerServer)
      await flushPromises()
      expect(Math.round(player.el?.currentTime ?? 0)).toBe(10)
    })

    it('leaves it alone once the user has scrubbed', async () => {
      const { release } = deferredPlayback()
      const { player } = await mountDownloaded()
      player.duration = 600
      await flushPromises()
      if (player.el) player.el.currentTime = 300

      release(newerServer)
      await flushPromises()
      expect(Math.round(player.el?.currentTime ?? 0), 'the scrub was overwritten').toBe(300)
    })

    it('does not correct when the DEVICE position is the newer one', async () => {
      // #1925's reconciliation decides; this only carries out what it decided.
      //
      // The device stamp has to be clear of CLOCK_SKEW_MARGIN_MS to count as newer. Inside that
      // margin `shouldPush` falls through to forward-only, where the LARGER position wins whatever
      // its age — which is why an "older but bigger" server value still takes precedence, and why
      // this test says nothing unless the margin is actually cleared.
      const { release } = deferredPlayback()
      localPosition.mockReturnValue({
        seconds: 10,
        finished: false,
        updatedAt: 1788900000 * 1000 + 10 * 60 * 1000,
      })
      const { player } = await mountDownloaded()
      player.duration = 600
      await flushPromises()

      release({ position_seconds: 200, finished: false, updated_at: 1788900000 })
      await flushPromises()
      expect(Math.round(player.el?.currentTime ?? 0), 'the device position was overruled').toBe(10)
    })
  })

  it('does not snapshot the registry stand-in as if it were the real episode', async () => {
    // The disk detail is thin — no publish date, no summary, every `has_*` false. Recording it
    // would make a later reopen paint a degraded page from cache until revalidation healed it.
    // `loading` stopped being the right signal the moment the fast path began clearing it early.
    vi.spyOn(api, 'getEpisode').mockImplementation(hangs)
    vi.spyOn(api, 'getAudioSource').mockImplementation(hangs)
    vi.spyOn(api, 'getPlayback').mockImplementation(hangs)
    const { w } = await mountDownloaded()

    expect(w.text()).toContain('Title From The Registry')
    expect(
      getPlayerViewSnapshot(SLUG),
      'the thin disk detail was snapshotted before the server answered',
    ).toBeUndefined()
  })

  /**
   * "Couldn't load this episode" describes OUR request. "You didn't download this one" describes
   * the user's situation, and is the only one of the two they can act on — an episode they DID
   * download plays here regardless.
   */
  it('says an episode is not downloaded, rather than that loading failed', async () => {
    const offline = () => Promise.reject(new Error('offline'))
    vi.spyOn(api, 'getEpisode').mockImplementation(offline)
    vi.spyOn(api, 'getAudioSource').mockImplementation(offline)
    vi.spyOn(api, 'getPlayback').mockImplementation(offline)
    const w = await mountPlayer('never-downloaded')
    await flushPromises()

    expect(w.find('[data-testid="player-not-downloaded"]').exists()).toBe(true)
    expect(w.text()).not.toContain("Couldn't load this episode.")
  })

  it('a SERVER error is still a load failure, not a download nag', async () => {
    // The distinction is about whether anyone answered. A 500 answered.
    vi.spyOn(api, 'getEpisode').mockImplementation(() =>
      Promise.reject(new ApiError(500, 'boom')),
    )
    const w = await mountPlayer('server-broken')
    await flushPromises()

    expect(w.find('[data-testid="player-not-downloaded"]').exists()).toBe(false)
    expect(w.text()).toContain("Couldn't load this episode.")
  })
})

describe('S3.1 transcript language control', () => {
  const SEG = [{ id: 'seg_0000', start: 0, end: 5, text: 'Hello', speaker: 'Host' }]

  /** A translated episode as the API reports it: English served, source recorded (D-38). */
  function translated(lang: string | null, source: string | null = 'es') {
    return {
      version: '1',
      episode_slug: 'ep-1',
      segments: SEG,
      language: lang,
      source_language: source,
      machine_translated: lang === 'en',
      translation_model: lang === 'en' ? 'google/translategemma-12b-it' : null,
    }
  }

  it('does not render for an English-native episode', async () => {
    // `source_language` absent — there is no second rendering, so a toggle would have one position.
    vi.spyOn(api, 'getSegments').mockResolvedValue(translated('en', null))
    const w = await mountPlayer()
    expect(w.find('[data-testid="transcript-language-control"]').exists()).toBe(false)
  })

  it('renders for a translated episode, with English active by default (D-38)', async () => {
    vi.spyOn(api, 'getSegments').mockResolvedValue(translated('en'))
    const w = await mountPlayer()
    expect(w.find('[data-testid="transcript-language-control"]').exists()).toBe(true)
    expect(w.find('[data-testid="transcript-lang-en"]').attributes('aria-pressed')).toBe('true')
    expect(w.find('[data-testid="transcript-lang-source"]').attributes('aria-pressed')).toBe('false')
  })

  it('asks the API for the SOURCE language and swaps the transcript', async () => {
    const spy = vi.spyOn(api, 'getSegments').mockResolvedValue(translated('en'))
    const w = await mountPlayer()
    spy.mockResolvedValue({
      ...translated('es'),
      segments: [{ id: 'seg_0000', start: 0, end: 5, text: 'Hola', speaker: 'Host' }],
    })
    await w.find('[data-testid="transcript-lang-source"]').trigger('click')
    await flushPromises()
    // The REQUEST carried the source tag — `lang` selects the alternative, it is not the default.
    expect(spy).toHaveBeenLastCalledWith('ep-1', 'es')
    expect(w.text()).toContain('Hola')
    expect(w.find('[data-testid="transcript-lang-source"]').attributes('aria-pressed')).toBe('true')
  })

  it('goes back to the default with no lang, rather than asking for `en`', async () => {
    const spy = vi.spyOn(api, 'getSegments').mockResolvedValue(translated('en'))
    const w = await mountPlayer()
    spy.mockResolvedValue(translated('es'))
    await w.find('[data-testid="transcript-lang-source"]').trigger('click')
    await flushPromises()
    spy.mockResolvedValue(translated('en'))
    await w.find('[data-testid="transcript-lang-en"]').trigger('click')
    await flushPromises()
    // `null`, not `'en'`: D-38 says the RESOLVER owns the default, and pinning `en` would ask for
    // a rendering that may not exist instead of taking whatever the episode actually has.
    expect(spy).toHaveBeenLastCalledWith('ep-1', null)
  })

  it('keeps the transcript on screen when the switch fails', async () => {
    const spy = vi.spyOn(api, 'getSegments').mockResolvedValue(translated('en'))
    const w = await mountPlayer()
    spy.mockRejectedValue(new api.ApiError(500, 'boom'))
    await w.find('[data-testid="transcript-lang-source"]').trigger('click')
    await flushPromises()
    // A failed switch must not empty the panel, and the toggle snaps back to what IS shown.
    expect(w.text()).toContain('Hello')
    expect(w.find('[data-testid="transcript-lang-en"]').attributes('aria-pressed')).toBe('true')
  })

  it('reports the language ACTUALLY served, not the one requested', async () => {
    // Asked for English, none exists: the response says `es`, so the original reads as active.
    vi.spyOn(api, 'getSegments').mockResolvedValue(translated('es'))
    const w = await mountPlayer()
    expect(w.find('[data-testid="transcript-lang-source"]').attributes('aria-pressed')).toBe('true')
    expect(w.find('[data-testid="transcript-lang-en"]').attributes('aria-pressed')).toBe('false')
  })
})

describe('opening the episode notes from a link', () => {
  // The opener renders only once the episode HAS insights — without them both cases would show no
  // opener, and the first test would pass for the wrong reason.
  beforeEach(() => {
    vi.spyOn(api, 'getInsights').mockResolvedValue({
      episode_slug: 'ep-1',
      insights: [
        { id: 'i0', text: 'insight', grounded: true, insight_type: null, confidence: null, position_hint: null, quotes: [] },
      ],
    })
  })

  it('?panel=notes opens the episode-notes panel (the daily recap email, operator 2026-10-05)', async () => {
    setActivePinia(createPinia())
    await router.push({ name: 'player', params: { slug: 'ep-1' }, query: { panel: 'notes' } })
    const w = mount(PlayerView, {
      props: { slug: 'ep-1' },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
    })
    mountedPlayers.push(w)
    await flushPromises()
    // The panel mounts the first time it opens (`panelEverOpen`), so its presence is the signal.
    // The Brief door stays on the obi while the panel is open, so it cannot be.
    expect(w.findComponent(KnowledgePanel).exists()).toBe(true)
  })

  it('without it the panel stays shut', async () => {
    const w = await mountPlayer('ep-1')
    expect(w.findComponent(KnowledgePanel).exists()).toBe(false)
    expect(w.find('[data-testid="player-open-insights"]').exists()).toBe(true)
  })
})

describe('the obi: Moments, Brief and About down the artwork edge (operator 2026-10-10)', () => {
  const oneInsight = {
    episode_slug: 'ep-1',
    insights: [{ id: 'i0', text: 'insight', grounded: true, insight_type: null, confidence: null, position_hint: null, quotes: [] }],
  }

  it('shows three equal doors when the episode has insights and a description', async () => {
    vi.spyOn(api, 'getInsights').mockResolvedValue(oneInsight)
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail({ description: 'The publisher text.' }))
    const w = await mountPlayer('ep-1')
    const obi = w.get('[data-testid="player-obi"]')
    const doors = obi.findAll('button')
    // "About", not "Description": three equal doors at the app's label size leave 63–73 px per
    // label on a phone, and DESCRIPTION is 92 px (measured 2026-10-10).
    expect(doors.map((d) => d.text())).toEqual(['Moments', 'Brief', 'About'])
    // Labels alike on purpose (operator 2026-10-10: no door's label a different size), no glyph.
    expect(new Set(doors.map((d) => d.get('span').classes().join(' '))).size).toBe(1)
    expect(obi.text()).not.toContain('✦')
    // The doors themselves alike too, so every rule between them is the same low-key one
    // (operator 2026-10-10: the brighter rule under Moments was dropped).
    expect(new Set(doors.map((d) => d.classes().join(' '))).size).toBe(1)
  })

  it('the Moments door opens the Moments view (?moments=1) and starts the reel', async () => {
    vi.spyOn(api, 'getInsights').mockResolvedValue(oneInsight)
    vi.spyOn(api, 'getMoments').mockResolvedValue({
      episode_slug: 'ep-1',
      total_seconds: 20,
      moments: [{ insight_id: 'i1', text: 'A point', speaker: null, start_ms: 60_000, end_ms: 80_000, clip_text: '' }],
    })
    const w = await mountPlayer('ep-1')
    const start = vi.spyOn(usePlayerStore(), 'startReel').mockReturnValue(true)
    await w.get('[data-testid="player-open-moments"]').trigger('click')
    await flushPromises()
    expect(router.currentRoute.value.query.moments).toBe('1')
    expect(start).toHaveBeenCalledWith('ep-1', expect.any(Array))
  })

  it('Brief opens the panel, Description opens the description sheet', async () => {
    vi.spyOn(api, 'getInsights').mockResolvedValue(oneInsight)
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail({ description: 'The publisher text.' }))
    const w = await mountPlayer('ep-1')
    await w.get('[data-testid="player-open-insights"]').trigger('click')
    await flushPromises()
    expect(w.findComponent(KnowledgePanel).exists()).toBe(true)
    await w.get('[data-testid="player-open-description"]').trigger('click')
    await flushPromises()
    expect(w.findComponent(EpisodeDescriptionSheet).props('open')).toBe(true)
  })

  it('a missing door leaves only the other', async () => {
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail({ description: 'The publisher text.' }))
    const w = await mountPlayer('ep-1')
    expect(w.findAll('[data-testid="player-obi"] button').map((d) => d.text())).toEqual(['About'])
  })

  it('with neither insights nor a description there is no band, and Zone D takes the full width', async () => {
    const w = await mountPlayer('ep-1')
    expect(w.find('[data-testid="player-obi"]').exists()).toBe(false)
    expect(w.get('[data-testid="player-zone-d-rest"]').classes()).toContain('right-0')
  })

  it('with a band, Zone D stops at its edge', async () => {
    vi.spyOn(api, 'getInsights').mockResolvedValue(oneInsight)
    const w = await mountPlayer('ep-1')
    expect(w.get('[data-testid="player-zone-d-rest"]').classes()).toContain('right-[44px]')
  })
})


describe('Moments view on the player page (operator 2026-10-10)', () => {
  const MOMENTS = {
    episode_slug: 'ep-1',
    total_seconds: 40,
    moments: [
      { insight_id: 'i1', text: 'First point', speaker: 'Ann', start_ms: 60_000, end_ms: 80_000, clip_text: 'a' },
      { insight_id: 'i2', text: 'Second point', speaker: null, start_ms: 600_000, end_ms: 620_000, clip_text: 'b' },
    ],
  }
  const oneInsight = {
    episode_slug: 'ep-1',
    insights: [{ id: 'i0', text: 'insight', grounded: true, insight_type: null, confidence: null, position_hint: null, quotes: [] }],
  }
  beforeEach(() => {
    vi.spyOn(api, 'getInsights').mockResolvedValue(oneInsight)
    vi.spyOn(api, 'getMoments').mockResolvedValue(MOMENTS)
  })

  it('?moments=1 starts the reel on this episode with its moments, once the episode is ready', async () => {
    const w = await mountPlayer('ep-1')
    const player = usePlayerStore()
    const start = vi.spyOn(player, 'startReel').mockReturnValue(true)
    await router.push({ name: 'player', params: { slug: 'ep-1' }, query: { moments: '1' } })
    await flushPromises()
    expect(start).toHaveBeenCalledWith('ep-1', [
      { insightId: 'i1', text: 'First point', speaker: 'Ann', startMs: 60_000, endMs: 80_000 },
      { insightId: 'i2', text: 'Second point', speaker: null, startMs: 600_000, endMs: 620_000 },
    ])
    w.unmount()
  })

  it('while a reel runs the page IS the Moments view; Keep listening leaves it and the query', async () => {
    await router.push({ name: 'player', params: { slug: 'ep-1' }, query: { moments: '1' } })
    const w = await mountPlayer('ep-1')
    const player = usePlayerStore()
    vi.spyOn(player, 'startReel').mockReturnValue(true)
    const exit = vi.spyOn(player, 'exitReel').mockImplementation(() => {
      player.reel = null
    })
    player.reel = {
      slug: 'ep-1',
      moments: [{ insightId: 'i1', text: 'First point', speaker: 'Ann', startMs: 60_000, endMs: 80_000 }],
      index: 0,
      returnTo: 0,
      done: false,
    }
    await flushPromises()
    expect(w.find('[data-testid="moments-reel"]').exists()).toBe(true)
    expect(w.find('[data-testid="player-hero"]').exists()).toBe(false)
    await w.get('[data-testid="moments-keep"]').trigger('click')
    await flushPromises()
    expect(exit).toHaveBeenCalledWith(true)
    expect(router.currentRoute.value.query.moments).toBeUndefined()
    expect(w.find('[data-testid="player-hero"]').exists()).toBe(true)
  })

  it('an episode with no moments drops ?moments=1 and stays the episode', async () => {
    vi.spyOn(api, 'getMoments').mockResolvedValue({ episode_slug: 'ep-1', moments: [], total_seconds: 0 })
    const w = await mountPlayer('ep-1')
    const start = vi.spyOn(usePlayerStore(), 'startReel')
    await router.push({ name: 'player', params: { slug: 'ep-1' }, query: { moments: '1' } })
    await flushPromises()
    expect(start).not.toHaveBeenCalled()
    expect(router.currentRoute.value.query.moments).toBeUndefined()
    w.unmount()
  })

  it('marks the moments on the density strip in the episode view', async () => {
    const w = await mountPlayer('ep-1')
    await flushPromises()
    expect(w.findAll('[data-testid="player-moment-mark"]')).toHaveLength(2)
  })
})

describe('Step through insights (operator 2026-10-10)', () => {
  const q = (start: number) => ({
    text: 'q', speaker: null, char_start: null, char_end: null,
    start_ms: start * 1000, end_ms: start * 1000 + 4000,
  })
  const insightsAt = (...starts: number[]) => ({
    episode_slug: 'ep-1',
    insights: starts.map((st, i) => ({
      id: `i${i}`, text: `insight ${i}`, grounded: true, insight_type: null, confidence: null,
      position_hint: null, quotes: [q(st)],
    })),
  })

  it("› jumps to the next insight's first quote; ‹ goes back", async () => {
    vi.spyOn(api, 'getInsights').mockResolvedValue(insightsAt(120, 300, 600))
    const w = await mountPlayer('ep-1')
    const seek = vi.spyOn(usePlayerStore(), 'seek').mockImplementation(() => {})
    await w.get('[data-testid="player-step-next-rest"]').trigger('click')
    expect(seek).toHaveBeenLastCalledWith(120)
    // From 0:00 there is no previous insight to go to.
    seek.mockClear()
    await w.get('[data-testid="player-step-prev-rest"]').trigger('click')
    expect(seek).not.toHaveBeenCalled()
  })
})
