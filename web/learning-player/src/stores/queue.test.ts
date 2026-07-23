import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { useQueueStore } from './queue'

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(api, 'putQueue').mockResolvedValue()
  // Mutations now ensureLoaded() first (guards against a late load clobbering an optimistic
  // add). Default to an empty server queue; tests that need a pre-populated queue set
  // `loaded = true` so ensureLoaded is a no-op, or override this mock.
  vi.spyOn(api, 'getQueue').mockResolvedValue([])
})
afterEach(() => vi.restoreAllMocks())

describe('queue store', () => {
  it('load() pulls items from the API', async () => {
    vi.spyOn(api, 'getQueue').mockResolvedValue(['a', 'b'])
    const q = useQueueStore()
    await q.load()
    expect(q.items).toEqual(['a', 'b'])
    expect(q.count).toBe(2)
  })

  it('add() appends once and persists', async () => {
    const q = useQueueStore()
    await q.add('a')
    await q.add('a') // idempotent
    await q.add('b')
    expect(q.items).toEqual(['a', 'b'])
    expect(api.putQueue).toHaveBeenCalled()
  })

  it('toggle() adds then removes', async () => {
    const q = useQueueStore()
    await q.toggle('a')
    expect(q.has('a')).toBe(true)
    await q.toggle('a')
    expect(q.has('a')).toBe(false)
  })

  it('playNext() inserts right after the current slug', async () => {
    const q = useQueueStore()
    q.loaded = true // pre-loaded; ensureLoaded is a no-op so the seeded items survive
    q.items = ['a', 'b', 'c']
    await q.playNext('z', 'a')
    expect(q.items).toEqual(['a', 'z', 'b', 'c'])
  })

  it('move() reorders within bounds', async () => {
    const q = useQueueStore()
    q.loaded = true
    q.items = ['a', 'b', 'c']
    await q.move('c', -1)
    expect(q.items).toEqual(['a', 'c', 'b'])
    await q.move('a', -1) // no-op at the top
    expect(q.items).toEqual(['a', 'c', 'b'])
  })

  it('add() during an in-flight load is not clobbered by the late load (RFC-099 race)', async () => {
    // The initial load()'s GET resolves LATE. A mutation must wait for it (ensureLoaded) and
    // append to the loaded state — otherwise the late load overwrites items and drops the add
    // ("queue empty" after add). Regression for the queue-persist race.
    let resolveGet: (v: string[]) => void = () => {}
    vi.spyOn(api, 'getQueue').mockReturnValue(
      new Promise<string[]>((r) => {
        resolveGet = r
      }),
    )
    const q = useQueueStore()
    const addP = q.add('x') // triggers load(); blocks on ensureLoaded()
    resolveGet(['a']) // the server already had ['a']
    await addP
    expect(q.items).toEqual(['a', 'x']) // appended to loaded state, not clobbered to []
    expect(api.putQueue).toHaveBeenCalledWith(['a', 'x'])
  })

  it('nextAfter() returns the auto-advance target', () => {
    const q = useQueueStore()
    q.items = ['a', 'b', 'c']
    expect(q.nextAfter('a')).toBe('b')
    expect(q.nextAfter('c')).toBeNull()
    expect(q.nextAfter('zzz')).toBeNull()
  })
})
