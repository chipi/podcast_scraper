// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The native shell's api base is ABSOLUTE (the server is another origin from capacitor://localhost)
// and its sign-in rides a Bearer token, not a cookie.
const NATIVE_BASE = 'https://api.example.test/api/app'
vi.mock('../services/tier', async () => {
  const actual = await vi.importActual<typeof import('../services/tier')>('../services/tier')
  return { ...actual, resolveApiBase: () => NATIVE_BASE, resolveGateAuthHeader: () => null }
})

const { setAuthToken } = await import('../services/api')
const { useAuthStore } = await import('./auth')
const { useUserPreferencesStore } = await import('./userPreferences')

const fetchMock = vi.fn()
vi.stubGlobal('fetch', fetchMock)

function ok(body: unknown): Response {
  return { ok: true, status: 200, json: async () => body } as unknown as Response
}

/**
 * Preferences in the native app (2026-10-11). The store called a bare relative
 * `fetch('/api/app/preferences')`, which in the shell resolves against capacitor://localhost and
 * never reaches the server — the dev api logged 356 requests from a phone and not one to
 * /preferences. So nothing synced on iOS or Android: the guided start's state and snooze, Trends'
 * scope, saved searches.
 */
describe('userPreferences in the native shell', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchMock.mockReset()
    useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
    setAuthToken('native-token')
  })

  it('reads from the server at the api base, signed in with the bearer token', async () => {
    fetchMock.mockResolvedValueOnce(ok({ preferences: { 'lp.guidedStart': 'done' } }))
    const store = useUserPreferencesStore()
    await store.hydrate()
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toBe(`${NATIVE_BASE}/preferences`)
    expect(new Headers((init as RequestInit).headers).get('Authorization')).toBe('Bearer native-token')
    expect(store.get('lp.guidedStart')).toBe('done')
  })

  it('writes to the server the same way', async () => {
    fetchMock.mockResolvedValueOnce(ok({ preferences: {} }))
    fetchMock.mockResolvedValueOnce(ok({}))
    const store = useUserPreferencesStore()
    await store.hydrate()
    await store.set('lp.guidedStart', 'active')
    const [url, init] = fetchMock.mock.calls[1]
    expect(String(url)).toBe(`${NATIVE_BASE}/preferences`)
    expect((init as RequestInit).method).toBe('PATCH')
    expect(new Headers((init as RequestInit).headers).get('Authorization')).toBe('Bearer native-token')
  })
})
