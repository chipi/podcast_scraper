import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import { batchViaGetEpisode } from '../test/apiViaSpies'
import * as contentCache from '../services/contentCache'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, PlaybackPosition } from '../services/types'
import RecentlyPlayedList from './RecentlyPlayedList.vue'

/**
 * Recently played — ported out of `QueuePanel.test.ts` when the panel was deleted (2026-09-27).
 *
 * The panel was this list's only test host, so deleting it would have taken six behaviours with it:
 * the played-at stamp, the demoted queue toggle, and the four halves of the offline cache contract.
 * None of those changed — only where they are mounted. `/queue` is the surface now, and the list is
 * its own component (it has been since 2026-09-23), so the tests belong on the component.
 *
 * What is NOT ported: the panel's own `emits close on the close button`. There is no panel and no
 * close button; that behaviour is gone rather than untested.
 */

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: stub },
    { path: '/episode/:slug', name: 'player', component: stub },
    { path: '/podcast/:feedId', name: 'podcast', component: stub },
  ],
})

function detail(slug: string, title: string): EpisodeDetail {
  return {
    slug,
    title,
    feed_id: 'f',
    podcast_title: 'Show',
    publish_date: '2024-01-01',
    duration_seconds: 1800,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: null,
    summary_text: null,
    summary_bullets: [],
    has_transcript: true,
    has_summary: false,
    has_gi: false,
    has_kg: false,
    has_bridge: false,
  } as unknown as EpisodeDetail
}

async function mountIt() {
  setActivePinia(createPinia())
  await router.push('/')
  await router.isReady()
  const w = mount(RecentlyPlayedList, {
    global: { plugins: [i18n, router], stubs: { teleport: true } },
  })
  await flushPromises()
  return w
}

afterEach(() => vi.restoreAllMocks())

beforeEach(() => {
  batchViaGetEpisode()
})

describe('RecentlyPlayedList (#1838, #1925)', () => {
  it('lists the playback history under its own heading', async () => {
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([
      { slug: 'r-1', position_seconds: 30, finished: false } as PlaybackPosition,
    ])
    vi.spyOn(api, 'getEpisode').mockImplementation(async () => detail('r-1', 'Recent Ep'))

    const w = await mountIt()
    expect(w.text()).toContain('Recently played')
    expect(w.get('[data-testid="queue-panel-recent"]').text()).toContain('Recent Ep')
  })

  it('states WHEN each recently-played episode was last played', async () => {
    // The list was already sorted by this timestamp and then refused to show it, so two sittings
    // looked identical (operator 2026-09-23).
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([
      { slug: 'r-1', position_seconds: 30, finished: true, updated_at: 1_760_000_000 } as PlaybackPosition,
    ])
    vi.spyOn(api, 'getEpisode').mockImplementation(async () => detail('r-1', 'Recent Ep'))
    const w = await mountIt()

    const recent = w.get('[data-testid="queue-panel-recent"]')
    const stamp = recent.find('time')
    expect(stamp.exists()).toBe(true)
    expect(stamp.text()).toMatch(/\d{1,2}:\d{2}/)
    // A machine-readable stamp beside the human one — the row is history, and `datetime` is what
    // makes it parseable rather than just legible.
    expect(stamp.attributes('datetime')).toBe(new Date(1_760_000_000_000).toISOString())
  })

  it('keeps the queue toggle OUT of the recently-played row', async () => {
    // Resuming is the point of this list, not re-queueing — and the compact column holds two
    // targets, so a third wrapped the ⋯ onto its own line.
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([
      { slug: 'r-1', position_seconds: 30, finished: true, updated_at: 1_760_000_000 } as PlaybackPosition,
    ])
    vi.spyOn(api, 'getEpisode').mockImplementation(async () => detail('r-1', 'Recent Ep'))
    const w = await mountIt()

    const labels = w
      .get('[data-testid="queue-panel-recent"]')
      .findAll('[data-testid="episode-actions"] button')
      .map((b) => b.attributes('aria-label'))
    expect(labels.length).toBeGreaterThan(0)
    expect(labels.some((l) => /queue/i.test(l ?? ''))).toBe(false)
  })

  it('shows the cached history when the network is gone', async () => {
    // Up next already survived offline (the queue store caches and flags `stale`); this half was
    // built straight from `GET /playback` with no cache, so the surface opened on a plane showing a
    // queue and an empty history — on the surface wanted precisely BECAUSE there is no network
    // (operator 2026-09-23).
    vi.spyOn(api, 'getPlaybackList').mockRejectedValue(new Error('offline'))
    vi.spyOn(contentCache, 'readCached').mockResolvedValue([
      { detail: detail('r-1', 'Cached Ep'), playedAt: 1_760_000_000 },
    ] as never)

    const w = await mountIt()
    expect(w.get('[data-testid="queue-panel-recent"]').text()).toContain('Cached Ep')
  })

  it('does not wipe the cached history when the request fails', async () => {
    // A FAILED request is not an empty history. Overwriting on it would throw away the only copy at
    // the moment it is the only copy. (`getPlaybackList` throws on a network failure; an empty
    // array is an answer — the case below.)
    vi.spyOn(api, 'getPlaybackList').mockRejectedValue(new Error('offline'))
    vi.spyOn(contentCache, 'readCached').mockResolvedValue([
      { detail: detail('r-1', 'Cached Ep'), playedAt: 1_760_000_000 },
    ] as never)
    const write = vi.spyOn(contentCache, 'writeCached').mockResolvedValue(undefined)

    const w = await mountIt()
    expect(w.get('[data-testid="queue-panel-recent"]').text()).toContain('Cached Ep')
    // By KEY: other callers cache through the same function, so a bare "was not called" would fail
    // on someone else's write and prove nothing about this one.
    expect(write.mock.calls.filter(([key]) => key === 'queue.recent')).toEqual([])
  })

  it('an EMPTY answer clears the cached history — e.g. after "Clear listening history" (2026-10-09)', async () => {
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    vi.spyOn(contentCache, 'readCached').mockResolvedValue([
      { detail: detail('r-1', 'Cached Ep'), playedAt: 1_760_000_000 },
    ] as never)
    const write = vi.spyOn(contentCache, 'writeCached').mockResolvedValue(undefined)

    const w = await mountIt()
    expect(w.find('[data-testid="queue-panel-recent"]').exists()).toBe(false)
    expect(w.text()).toContain(en.queue.recentEmpty)
    expect(write.mock.calls.filter(([key]) => key === 'queue.recent')).toEqual([['queue.recent', []]])
  })

  it('caches a successful history so the next offline open has one', async () => {
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([
      { slug: 'r-1', position_seconds: 30, finished: true, updated_at: 1_760_000_000 } as PlaybackPosition,
    ])
    vi.spyOn(api, 'getEpisode').mockImplementation(async () => detail('r-1', 'Recent Ep'))
    vi.spyOn(contentCache, 'readCached').mockResolvedValue(null)
    const write = vi.spyOn(contentCache, 'writeCached').mockResolvedValue(undefined)

    await mountIt()
    expect(write).toHaveBeenCalledWith('queue.recent', expect.any(Array))
  })
})
