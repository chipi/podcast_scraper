import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { useFollowedShows } from './useFollowedShows'
import { useAuthStore } from '../stores/auth'
import { useLibraryStore } from '../stores/library'
import type { LibraryItem, Podcast } from '../services/types'

/**
 * The composable behind Library → Following.
 *
 * It had no tests, and its own docblock stated the invariant it did not enforce: "Both must resolve
 * for 'you follow nothing' to be a truthful render." `library.ensureLoaded()` never throws — by
 * design, so an offline library cannot abort boot — so a failed `/library` resolved quietly, the
 * section went READY, and the view rendered "You're not following any shows yet" plus six shows to
 * follow. A user who WAS following shows saw that on prod.
 */

vi.mock('../services/contentCache', () => ({
  readCached: async () => null,
  writeCached: async () => {},
  clearCached: async () => {},
  setCacheNamespace: () => {},
  CACHE_KEYS: ['library'],
}))

function podcast(feedId: string): Podcast {
  return {
    feed_id: feedId,
    title: feedId,
    artwork_url: null,
    image_url: null,
    description: null,
    episode_count: 1,
  }
}

function follow(feedId: string): LibraryItem {
  return { feed_id: feedId, feed_url: null, title: feedId, added_at: null }
}

function signIn(): void {
  const auth = useAuthStore()
  auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
  auth.loaded = true
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.restoreAllMocks()
})

describe('useFollowedShows', () => {
  it('a failed library load THROWS, so the caller can render error + retry', async () => {
    // The whole bug in one assertion. Resolving quietly is what let the view claim, confidently and
    // wrongly, that the user follows nothing — and then offer them six shows to follow.
    signIn()
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([podcast('f1')])
    vi.spyOn(api, 'getLibrary').mockRejectedValue(new Error('offline'))
    const { load } = useFollowedShows()
    await expect(load()).rejects.toBeInstanceOf(Error)
  })

  it('a cached library is NOT a failure — stale still renders', async () => {
    // The distinction that matters: showing a slightly old list is right, claiming an empty one is
    // not. The store marks a cache hit `loaded`, so this must not throw.
    signIn()
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([podcast('f1')])
    vi.spyOn(api, 'getLibrary').mockRejectedValue(new Error('offline'))
    const lib = useLibraryStore()
    lib.items = [follow('f1')]
    lib.loaded = true
    const { load, shows } = useFollowedShows()
    await expect(load()).resolves.toBeUndefined()
    expect(shows.value).toHaveLength(1)
  })

  it('a genuinely empty library resolves — that empty state is truthful', async () => {
    signIn()
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([podcast('f1')])
    vi.spyOn(api, 'getLibrary').mockResolvedValue([])
    const { load, shows, suggested } = useFollowedShows()
    await expect(load()).resolves.toBeUndefined()
    expect(shows.value).toEqual([])
    expect(suggested.value.length, 'nothing to suggest from an empty catalogue').toBeGreaterThan(0)
  })

  it('suggestions exclude what you already follow', async () => {
    // Otherwise the empty-state grid offers "+ Follow show" for a show you follow — the incoherence
    // that got reported.
    signIn()
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([podcast('f1'), podcast('f2')])
    vi.spyOn(api, 'getLibrary').mockResolvedValue([follow('f1')])
    const { load, suggested } = useFollowedShows()
    await load()
    expect(suggested.value.map((p) => p.feed_id)).toEqual(['f2'])
  })

  it('signed out, there is nothing to show and nothing to fail', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([podcast('f1')])
    const spy = vi.spyOn(api, 'getLibrary')
    const { load, shows } = useFollowedShows()
    await expect(load()).resolves.toBeUndefined()
    expect(shows.value).toEqual([])
    expect(spy, 'asked for a signed-out user library').not.toHaveBeenCalled()
  })
})
