import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const cached: Record<string, unknown> = {}
vi.mock('../services/contentCache', () => ({
  isArrayCache: (v: unknown) => Array.isArray(v),
  readCached: async (k: string) => cached[k] ?? null,
  writeCached: async () => {},
  clearCached: async () => {},
  setCacheNamespace: () => {},
  CACHE_KEYS: ['completed'],
}))
import * as api from '../services/api'
import { ApiError } from '../services/api'
import * as outbox from '../services/outbox'
import { useCompletedStore } from './completed'

// A fake server set — the store takes the response as truth.
let server: string[] = []

beforeEach(() => {
  setActivePinia(createPinia())
  server = []
  vi.spyOn(api, 'getCompleted').mockImplementation(async () => [...server])
  vi.spyOn(api, 'markCompleted').mockImplementation(async (slug) => {
    if (!server.includes(slug)) server.push(slug)
    return [...server]
  })
  vi.spyOn(api, 'unmarkCompleted').mockImplementation(async (slug) => {
    server = server.filter((s) => s !== slug)
    return [...server]
  })
  vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
})

afterEach(() => {
  vi.restoreAllMocks()
  for (const k of Object.keys(cached)) delete cached[k]
})

describe('completed store', () => {
  it('marks, reports has(), and is idempotent', async () => {
    const c = useCompletedStore()
    expect(c.has('ep-1')).toBe(false)
    await c.mark('ep-1')
    expect(c.has('ep-1')).toBe(true)
    await c.mark('ep-1') // idempotent
    expect(c.slugs.filter((s) => s === 'ep-1')).toHaveLength(1)
    expect(server).toEqual(['ep-1'])
  })

  it('toggle flips on then off', async () => {
    const c = useCompletedStore()
    await c.toggle('ep-1')
    expect(c.has('ep-1')).toBe(true)
    await c.toggle('ep-1')
    expect(c.has('ep-1')).toBe(false)
    expect(server).toEqual([])
  })

  it('keeps the optimistic mark and queues the write when the request never lands', async () => {
    vi.spyOn(api, 'markCompleted').mockRejectedValue(new TypeError('Failed to fetch'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const c = useCompletedStore()
    const ok = await c.mark('ep-9')
    expect(ok).toBe(true) // recorded — in the outbox
    expect(c.has('ep-9')).toBe(true) // optimistic mark stays
    expect(enqueue).toHaveBeenCalledWith({ op: 'completed.add', slug: 'ep-9' })
  })

  it('reverts a mark on a server REFUSAL (permanent), and does not queue it', async () => {
    vi.spyOn(api, 'markCompleted').mockRejectedValue(new ApiError(404, 'gone'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const c = useCompletedStore()
    const ok = await c.mark('ep-404')
    expect(ok).toBe(false)
    expect(c.has('ep-404')).toBe(false) // reverted
    expect(enqueue).not.toHaveBeenCalled()
  })
})
