import { afterEach, describe, expect, it, vi } from 'vitest'
import { __resetPodcastsCache, getPodcasts } from './api'

afterEach(() => {
  __resetPodcastsCache()
  vi.unstubAllGlobals()
})

describe('getPodcasts is shared (2026-10-08: Library → Following fetched it three times)', () => {
  it('callers in the same minute share ONE request', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200, json: async () => ({ items: [{ feed_id: 'f1' }] }) }))
    vi.stubGlobal('fetch', fetchMock)
    const [a, b] = await Promise.all([getPodcasts(), getPodcasts()])
    await getPodcasts()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(a).toEqual([{ feed_id: 'f1' }])
    expect(b).toBe(a)
  })

  it('a failed request is not kept: the next caller retries', async () => {
    const fail = vi.fn(async () => ({ ok: false, status: 503, json: async () => ({}) }))
    vi.stubGlobal('fetch', fail)
    await expect(getPodcasts()).rejects.toBeTruthy()
    const ok = vi.fn(async () => ({ ok: true, status: 200, json: async () => ({ items: [] }) }))
    vi.stubGlobal('fetch', ok)
    await expect(getPodcasts()).resolves.toEqual([])
    expect(ok).toHaveBeenCalledTimes(1)
  })
})
