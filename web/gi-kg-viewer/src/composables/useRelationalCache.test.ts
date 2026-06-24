// @vitest-environment happy-dom
//
// #1075 chunk 4 — RFC-094 panel cache unit coverage. The composable
// wraps four relational endpoints with an LRU + TTL cache shared at
// module scope. Tests cover hit/miss/expiry/LRU eviction/invalidation
// flows.
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Mock the underlying fetches BEFORE importing the cache so the
// vi.mock factory replacement is picked up.
vi.mock('../api/relationalApi', () => ({
  fetchPersonTopics: vi.fn(),
  fetchCoSpeakers: vi.fn(),
  fetchPositions: vi.fn(),
}))
vi.mock('../api/cilApi', () => ({
  fetchPersonProfile: vi.fn(),
}))

import {
  fetchPersonTopics as rawFetchPersonTopics,
  fetchCoSpeakers as rawFetchCoSpeakers,
  fetchPositions as rawFetchPositions,
} from '../api/relationalApi'
import { fetchPersonProfile as rawFetchPersonProfile } from '../api/cilApi'

import {
  _cacheSizeForTest,
  _resetCacheParamsForTest,
  _setCacheParamsForTest,
  cachedFetchCoSpeakers,
  cachedFetchPersonProfile,
  cachedFetchPersonTopics,
  cachedFetchPositions,
  invalidateRelationalCache,
} from './useRelationalCache'

describe('useRelationalCache (#1075 chunk 4)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    invalidateRelationalCache()
    _resetCacheParamsForTest()
    vi.mocked(rawFetchPersonTopics).mockReset()
    vi.mocked(rawFetchCoSpeakers).mockReset()
    vi.mocked(rawFetchPositions).mockReset()
    vi.mocked(rawFetchPersonProfile).mockReset()
  })

  afterEach(() => {
    invalidateRelationalCache()
    _resetCacheParamsForTest()
  })

  it('cache miss forwards to the wrapped fn with the expected args', async () => {
    vi.mocked(rawFetchPersonTopics).mockResolvedValue({
      subject: 'person:a',
      results: [],
    })
    await cachedFetchPersonTopics('/corpus', 'person:a')
    expect(rawFetchPersonTopics).toHaveBeenCalledWith('/corpus', 'person:a')
    expect(rawFetchPersonTopics).toHaveBeenCalledTimes(1)
  })

  it('cache hit returns the prior payload without re-calling the wrapped fn', async () => {
    vi.mocked(rawFetchPersonTopics).mockResolvedValueOnce({
      subject: 'person:a',
      results: [{ id: 't:1', type: 'topic', text: 'AI', show_id: '', episode_id: '' }],
    })
    const first = await cachedFetchPersonTopics('/corpus', 'person:a')
    const second = await cachedFetchPersonTopics('/corpus', 'person:a')
    expect(rawFetchPersonTopics).toHaveBeenCalledTimes(1)
    expect(second).toEqual(first)
  })

  it('TTL expiry forces a re-fetch', async () => {
    _setCacheParamsForTest({ ttlMs: 1 })
    vi.mocked(rawFetchPersonTopics)
      .mockResolvedValueOnce({ subject: 'person:a', results: [] })
      .mockResolvedValueOnce({ subject: 'person:a', results: [] })
    await cachedFetchPersonTopics('/corpus', 'person:a')
    // Wait past the TTL.
    await new Promise((resolve) => setTimeout(resolve, 5))
    await cachedFetchPersonTopics('/corpus', 'person:a')
    expect(rawFetchPersonTopics).toHaveBeenCalledTimes(2)
  })

  it('different (corpus, person) tuples land in different cache slots', async () => {
    vi.mocked(rawFetchPersonTopics).mockResolvedValue({
      subject: 'x',
      results: [],
    })
    await cachedFetchPersonTopics('/corpus-a', 'person:a')
    await cachedFetchPersonTopics('/corpus-b', 'person:a')
    await cachedFetchPersonTopics('/corpus-a', 'person:b')
    // Each tuple is a miss → 3 underlying calls.
    expect(rawFetchPersonTopics).toHaveBeenCalledTimes(3)
    // Repeat the first tuple — hit, no new call.
    await cachedFetchPersonTopics('/corpus-a', 'person:a')
    expect(rawFetchPersonTopics).toHaveBeenCalledTimes(3)
  })

  it('LRU eviction drops the oldest entry when the cap is hit', async () => {
    _setCacheParamsForTest({ maxEntries: 2 })
    vi.mocked(rawFetchPersonTopics).mockResolvedValue({
      subject: 'x',
      results: [],
    })
    await cachedFetchPersonTopics('/corpus', 'person:a')
    await cachedFetchPersonTopics('/corpus', 'person:b')
    await cachedFetchPersonTopics('/corpus', 'person:c')
    expect(_cacheSizeForTest()).toBe(2)
    // person:a was evicted — re-querying it forces a re-fetch.
    await cachedFetchPersonTopics('/corpus', 'person:a')
    expect(rawFetchPersonTopics).toHaveBeenCalledTimes(4)
  })

  it('LRU recency: a hit refreshes the touched entry so it is not the next eviction candidate', async () => {
    _setCacheParamsForTest({ maxEntries: 2 })
    vi.mocked(rawFetchPersonTopics).mockResolvedValue({
      subject: 'x',
      results: [],
    })
    await cachedFetchPersonTopics('/corpus', 'person:a')
    await cachedFetchPersonTopics('/corpus', 'person:b')
    // Touch person:a → person:b is now the oldest.
    await cachedFetchPersonTopics('/corpus', 'person:a')
    // Adding a 3rd entry should evict person:b, not person:a.
    await cachedFetchPersonTopics('/corpus', 'person:c')
    // Re-querying person:a is a hit (no new fetch).
    const before = vi.mocked(rawFetchPersonTopics).mock.calls.length
    await cachedFetchPersonTopics('/corpus', 'person:a')
    expect(vi.mocked(rawFetchPersonTopics).mock.calls.length).toBe(before)
  })

  it('invalidateRelationalCache() empties the cache', async () => {
    vi.mocked(rawFetchPersonTopics).mockResolvedValue({
      subject: 'x',
      results: [],
    })
    await cachedFetchPersonTopics('/corpus', 'person:a')
    expect(_cacheSizeForTest()).toBe(1)
    invalidateRelationalCache()
    expect(_cacheSizeForTest()).toBe(0)
  })

  it('switching corpusPath produces a distinct cache slot (no stale return)', async () => {
    vi.mocked(rawFetchPersonTopics)
      .mockResolvedValueOnce({
        subject: 'x',
        results: [{ id: 't:a', type: 'topic', text: 'A', show_id: '', episode_id: '' }],
      })
      .mockResolvedValueOnce({
        subject: 'x',
        results: [{ id: 't:b', type: 'topic', text: 'B', show_id: '', episode_id: '' }],
      })
    const first = await cachedFetchPersonTopics('/corpus-a', 'person:a')
    const second = await cachedFetchPersonTopics('/corpus-b', 'person:a')
    // Different corpora → different cache slots → different payloads.
    expect(first.results[0].text).toBe('A')
    expect(second.results[0].text).toBe('B')
    expect(rawFetchPersonTopics).toHaveBeenCalledTimes(2)
  })

  it('separate fetch fns are cached independently', async () => {
    vi.mocked(rawFetchPersonTopics).mockResolvedValue({ subject: 'x', results: [] })
    vi.mocked(rawFetchCoSpeakers).mockResolvedValue({ subject: 'x', results: [] })
    vi.mocked(rawFetchPositions).mockResolvedValue({ subject: 'x', results: [] })
    vi.mocked(rawFetchPersonProfile).mockResolvedValue({
      path: '/c',
      person_id: 'person:a',
      topics: {},
      quotes: [],
    })
    await cachedFetchPersonTopics('/corpus', 'person:a')
    await cachedFetchCoSpeakers('/corpus', 'person:a')
    await cachedFetchPositions('/corpus', 'person:a')
    await cachedFetchPersonProfile('/corpus', 'person:a')
    // Repeat each — all hits, no new calls.
    await cachedFetchPersonTopics('/corpus', 'person:a')
    await cachedFetchCoSpeakers('/corpus', 'person:a')
    await cachedFetchPositions('/corpus', 'person:a')
    await cachedFetchPersonProfile('/corpus', 'person:a')
    expect(rawFetchPersonTopics).toHaveBeenCalledTimes(1)
    expect(rawFetchCoSpeakers).toHaveBeenCalledTimes(1)
    expect(rawFetchPositions).toHaveBeenCalledTimes(1)
    expect(rawFetchPersonProfile).toHaveBeenCalledTimes(1)
  })
})
