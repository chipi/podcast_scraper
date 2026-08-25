import { beforeEach, describe, expect, it } from 'vitest'
import {
  clearPlayerViewCache,
  getPlayerViewSnapshot,
  setPlayerViewSnapshot,
  type PlayerViewSnapshot,
} from './player-view-cache'

function snap(title: string): PlayerViewSnapshot {
  return {
    episode: { title } as PlayerViewSnapshot['episode'],
    segments: [],
    audioUrl: null,
    insights: [],
    topics: [],
    persons: [],
    relatedEpisodes: [],
    stats: null,
  }
}

describe('player-view-cache', () => {
  beforeEach(() => clearPlayerViewCache())

  it('round-trips a snapshot by slug', () => {
    setPlayerViewSnapshot('a', snap('A'))
    expect(getPlayerViewSnapshot('a')?.episode?.title).toBe('A')
    expect(getPlayerViewSnapshot('missing')).toBeUndefined()
  })

  it('evicts the oldest once past the bound (LRU by insertion)', () => {
    for (const s of ['a', 'b', 'c', 'd', 'e']) setPlayerViewSnapshot(s, snap(s))
    // MAX = 4 → the first inserted ('a') is gone, the last four survive.
    expect(getPlayerViewSnapshot('a')).toBeUndefined()
    for (const s of ['b', 'c', 'd', 'e']) expect(getPlayerViewSnapshot(s)?.episode?.title).toBe(s)
  })

  it('re-setting a slug refreshes it to most-recently-used, sparing it from eviction', () => {
    for (const s of ['a', 'b', 'c', 'd']) setPlayerViewSnapshot(s, snap(s))
    setPlayerViewSnapshot('a', snap('A2')) // touch 'a' → now MRU, 'b' becomes oldest
    setPlayerViewSnapshot('e', snap('e')) // evicts 'b', not 'a'
    expect(getPlayerViewSnapshot('a')?.episode?.title).toBe('A2')
    expect(getPlayerViewSnapshot('b')).toBeUndefined()
  })
})
