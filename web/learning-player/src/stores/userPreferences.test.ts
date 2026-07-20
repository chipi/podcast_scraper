// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useUserPreferencesStore } from './userPreferences'

// Mock global fetch — every test controls the response shape.
const fetchMock = vi.fn()
vi.stubGlobal('fetch', fetchMock)

function makeResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response
}

describe('learning-player useUserPreferencesStore (USERPREFS-1 gh #1213)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchMock.mockReset()
  })

  it('hydrate() populates preferences from a 200 GET', async () => {
    fetchMock.mockResolvedValueOnce(
      makeResponse({ preferences: { 'lp.interests.dismissed': true, other: 'x' } }),
    )
    const store = useUserPreferencesStore()
    await store.hydrate()
    expect(store.hydrated).toBe(true)
    expect(store.available).toBe(true)
    expect(store.get<boolean>('lp.interests.dismissed')).toBe(true)
    expect(store.get<string>('other')).toBe('x')
  })

  it('hydrate() silently marks unavailable on non-2xx response', async () => {
    fetchMock.mockResolvedValueOnce(makeResponse(null, 401))
    const store = useUserPreferencesStore()
    await store.hydrate()
    expect(store.hydrated).toBe(true)
    expect(store.available).toBe(false)
    expect(store.get('anything')).toBeUndefined()
  })

  it('hydrate() silently marks unavailable on network error', async () => {
    fetchMock.mockRejectedValueOnce(new Error('offline'))
    const store = useUserPreferencesStore()
    await store.hydrate()
    expect(store.available).toBe(false)
  })

  it('hydrate() is idempotent — repeated calls are no-ops', async () => {
    fetchMock.mockResolvedValueOnce(makeResponse({ preferences: {} }))
    const store = useUserPreferencesStore()
    await store.hydrate()
    await store.hydrate()
    await store.hydrate()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('set() updates the local ref immediately (optimistic) even before the server responds', async () => {
    fetchMock.mockResolvedValueOnce(makeResponse({ preferences: {} }))
    const store = useUserPreferencesStore()
    await store.hydrate()

    // Set fires and-await; local value is updated synchronously in the store body.
    fetchMock.mockResolvedValueOnce(makeResponse({}))
    await store.set('k', 42)
    expect(store.get<number>('k')).toBe(42)
  })

  it('set() PATCHes /api/app/preferences with the single-key payload', async () => {
    fetchMock.mockResolvedValueOnce(makeResponse({ preferences: {} }))
    const store = useUserPreferencesStore()
    await store.hydrate()

    fetchMock.mockResolvedValueOnce(makeResponse({}))
    await store.set('lp.interests.dismissed', true)
    // The 2nd call is the PATCH.
    const [url, init] = fetchMock.mock.calls[1]
    expect(url).toBe('/api/app/preferences')
    expect((init as RequestInit).method).toBe('PATCH')
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      'lp.interests.dismissed': true,
    })
  })

  it('set() silently marks unavailable when PATCH fails', async () => {
    fetchMock.mockResolvedValueOnce(makeResponse({ preferences: {} }))
    const store = useUserPreferencesStore()
    await store.hydrate()

    fetchMock.mockRejectedValueOnce(new Error('offline'))
    await store.set('k', 1)
    expect(store.available).toBe(false)
    // Local value is still applied — the caller's UI shouldn't roll back.
    expect(store.get<number>('k')).toBe(1)
  })
})
