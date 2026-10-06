// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useAuthStore } from './auth'
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
    // Preferences are per-user and only hydrate when signed in; authenticate so these tests
    // exercise the fetch/set behaviour (the signed-out no-op is its own test below).
    useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
  })

  it('hydrate() is a no-op (no fetch) when signed out', async () => {
    useAuthStore().user = null
    const store = useUserPreferencesStore()
    await store.hydrate()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(store.hydrated).toBe(false)
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

  it('a hydrate that resolves AFTER a local write keeps the write — a stale snapshot does not undo it', async () => {
    // Save, un-Save, and a GET that was already in flight returns a snapshot taken between the two:
    // replacing wholesale put the query back and the button read "Saved ✓" (e2e flake 2026-10-04).
    let resolveGet!: (r: Response) => void
    fetchMock.mockImplementationOnce(() => new Promise<Response>((r) => (resolveGet = r)))
    fetchMock.mockResolvedValue(makeResponse({}, 200)) // the PATCHes
    const store = useUserPreferencesStore()
    const hydrating = store.hydrate()
    await store.set('lp.savedQueries', []) // un-saved while the GET is still out
    resolveGet(makeResponse({ preferences: { 'lp.savedQueries': [{ q: 'risk' }], other: 'x' } }))
    await hydrating
    expect(store.get('lp.savedQueries')).toEqual([]) // the user's last word stands
    expect(store.get('other')).toBe('x') // and the rest of the snapshot still arrives
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
    // The server's contract (UserPreferencesPatch): the keys go under `preferences`. This test used
    // to pin the bare `{ key: value }` shape — the bug — so it passed while every real write 422'd.
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      preferences: { 'lp.interests.dismissed': true },
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

  // #1906 — one offline blip must not write off preferences sync for the whole session.

  it('a transport failure stops sync, and resetAvailability lets it resume', async () => {
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    const store = useUserPreferencesStore()

    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    await store.hydrate()
    expect(store.available).toBe(false)

    // hydrate() early-returns once `hydrated` is true, so without a reset this is permanent.
    fetchMock.mockResolvedValueOnce(makeResponse({ preferences: { a: 1 } }))
    await store.hydrate()
    expect(store.preferences).toEqual({})

    store.resetAvailability()
    expect(store.available).toBe(true)
    fetchMock.mockResolvedValueOnce(makeResponse({ preferences: { a: 1 } }))
    await store.hydrate()
    expect(store.preferences).toEqual({ a: 1 })
  })
})
