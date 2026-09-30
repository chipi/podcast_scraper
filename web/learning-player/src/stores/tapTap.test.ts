import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { serialWrites } from '../services/serialWrites'
import type { FavoritesResponse, Highlight, LibraryItem } from '../services/types'
import { useCaptureStore } from './capture'
import { useCompletedStore } from './completed'
import { useFavoritesStore } from './favorites'
import { useInterestsStore } from './interests'
import { useLibraryStore } from './library'
import { useQueueStore } from './queue'

/**
 * Tap-tap: every on/off control ends in the state the user left it (operator 2026-09-30:
 * "consistency above all").
 *
 * Each store's first write (turning the thing ON) is HELD BACK and released only after the second
 * tap (turning it OFF) has been made — the ordering that used to leave the control showing ON,
 * because the first write's response arrived last and was adopted. Each test asserts both halves:
 * the store shows OFF, and the server saw the writes in the user's order (on, then off).
 */

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

/** A promise the test resolves by hand, so a response can be made to arrive late. */
function held<T>(value: T): { promise: () => Promise<T>; release: () => void } {
  let release!: () => void
  const gate = new Promise<void>((r) => (release = r))
  return { promise: async () => (await gate, value), release }
}

describe('serialWrites', () => {
  it('runs writes one at a time, in order, and tells each whether it is still the latest', async () => {
    const w = serialWrites()
    const log: string[] = []
    const first = held('a')
    const p1 = w.run(async (isLatest) => {
      log.push('1:start')
      await first.promise()
      log.push(`1:end latest=${isLatest()}`)
    })
    const p2 = w.run(async (isLatest) => {
      log.push(`2:start latest=${isLatest()}`)
    })
    await Promise.resolve()
    first.release()
    await Promise.all([p1, p2])
    expect(log).toEqual(['1:start', '1:end latest=false', '2:start latest=true'])
  })

  it('keeps going after a write throws', async () => {
    const w = serialWrites()
    await expect(w.run(async () => Promise.reject(new Error('boom')))).rejects.toThrow('boom')
    await expect(w.run(async () => 'next')).resolves.toBe('next')
  })
})

describe('tap-tap leaves every toggle OFF, and the server saw ON then OFF', () => {
  it('follow (interests)', async () => {
    const order: string[] = []
    const add = held<string[]>(['topic:ai'])
    vi.spyOn(api, 'addInterest').mockImplementation(async () => (order.push('on'), add.promise()))
    vi.spyOn(api, 'removeInterest').mockImplementation(async () => (order.push('off'), []))
    const s = useInterestsStore()
    const a = s.toggle('topic:ai')
    const b = s.toggle('topic:ai')
    add.release()
    await Promise.all([a, b])
    expect(s.has('topic:ai')).toBe(false)
    expect(order).toEqual(['on', 'off'])
  })

  it('heart (favorites)', async () => {
    const order: string[] = []
    const saved: FavoritesResponse = { episodes: [{ slug: 'ep1' } as never], entities: [] }
    const add = held(saved)
    vi.spyOn(api, 'addFavorite').mockImplementation(async () => (order.push('on'), add.promise()))
    vi.spyOn(api, 'removeFavorite').mockImplementation(
      async () => (order.push('off'), { episodes: [], entities: [] }),
    )
    const s = useFavoritesStore()
    const a = s.toggle({ kind: 'episode', ref: 'ep1' })
    expect(s.has('episode', 'ep1'), 'the heart flips on the tap').toBe(true)
    const b = s.toggle({ kind: 'episode', ref: 'ep1' })
    expect(s.has('episode', 'ep1')).toBe(false)
    add.release()
    await Promise.all([a, b])
    expect(s.has('episode', 'ep1')).toBe(false)
    expect(order).toEqual(['on', 'off'])
  })

  it('follow show (library)', async () => {
    const order: string[] = []
    const row = { feed_id: 'f1', feed_url: null, title: 'Show', added_at: 1 } as LibraryItem
    const add = held([row])
    vi.spyOn(api, 'followShow').mockImplementation(async () => (order.push('on'), add.promise()))
    vi.spyOn(api, 'unfollowShow').mockImplementation(async () => (order.push('off'), []))
    const s = useLibraryStore()
    const a = s.toggle('f1')
    const b = s.toggle('f1')
    add.release()
    await Promise.all([a, b])
    expect(s.has('f1')).toBe(false)
    expect(order).toEqual(['on', 'off'])
  })

  it('played (completed)', async () => {
    const order: string[] = []
    vi.spyOn(api, 'getCompleted').mockResolvedValue([])
    const add = held(['ep1'])
    vi.spyOn(api, 'markCompleted').mockImplementation(async () => (order.push('on'), add.promise()))
    vi.spyOn(api, 'unmarkCompleted').mockImplementation(async () => (order.push('off'), []))
    const s = useCompletedStore()
    await s.ensureLoaded()
    const a = s.toggle('ep1')
    await vi.waitFor(() => expect(order).toEqual(['on']))
    const b = s.toggle('ep1')
    // This toggle awaits the store's load before it flips, so release the held ON response only
    // once the OFF tap is showing — otherwise ON lands before OFF is even sent.
    await vi.waitFor(() => expect(s.slugs.includes('ep1')).toBe(false))
    add.release()
    await Promise.all([a, b])
    expect(s.slugs.includes('ep1')).toBe(false)
    expect(order).toEqual(['on', 'off'])
  })

  it('queue', async () => {
    const order: string[] = []
    vi.spyOn(api, 'getQueue').mockResolvedValue([])
    const add = held(['ep1'])
    vi.spyOn(api, 'addQueueItem').mockImplementation(async () => (order.push('on'), add.promise()))
    vi.spyOn(api, 'removeQueueItem').mockImplementation(async () => (order.push('off'), []))
    const s = useQueueStore()
    await s.ensureLoaded()
    const a = s.toggle('ep1')
    await vi.waitFor(() => expect(order).toEqual(['on']))
    const b = s.toggle('ep1')
    // This toggle awaits the store's load before it flips, so release the held ON response only
    // once the OFF tap is showing — otherwise ON lands before OFF is even sent.
    await vi.waitFor(() => expect(s.items.includes('ep1')).toBe(false))
    add.release()
    await Promise.all([a, b])
    expect(s.items.includes('ep1')).toBe(false)
    expect(order).toEqual(['on', 'off'])
  })

  it('save insight (capture): the delete targets the id the SERVER gave the row', async () => {
    const order: string[] = []
    const row = { id: 'srv-1', source_insight_id: 'i1', kind: 'insight' } as unknown as Highlight
    const create = held(row)
    vi.spyOn(api, 'createHighlight').mockImplementation(
      async () => (order.push('on'), create.promise()),
    )
    vi.spyOn(api, 'deleteHighlight').mockImplementation(async (id: string) => {
      order.push(`off:${id}`)
      return []
    })
    const s = useCaptureStore()
    const a = s.captureInsight('ep1', { id: 'i1', text: 'An insight.' })
    const b = s.captureInsight('ep1', { id: 'i1', text: 'An insight.' })
    create.release()
    await Promise.all([a, b])
    expect(s.highlights.some((h) => h.source_insight_id === 'i1')).toBe(false)
    // Not DELETE /highlights/<client id> — that 404ed and the store put the row back.
    expect(order).toEqual(['on', 'off:srv-1'])
  })
})
