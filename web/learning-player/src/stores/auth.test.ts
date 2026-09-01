import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import * as deviceStore from '../services/deviceStore'
import { useAuthStore } from './auth'

const ME = { user_id: 'u_1', email: 'dev@localhost', name: 'Dev' }
/** Faithful fake of device storage: a read returns what the last write stored. */
let disk: Record<string, unknown> = {}

beforeEach(() => {
  setActivePinia(createPinia())
  disk = {}
  vi.spyOn(deviceStore, 'setDeviceJson').mockImplementation(async (k, v) => {
    disk[k] = v
  })
  vi.spyOn(deviceStore, 'getDeviceJson').mockImplementation(async (k) => (disk[k] ?? null) as never)
  vi.spyOn(deviceStore, 'removeDeviceKey').mockImplementation(async (k) => {
    delete disk[k]
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('auth store', () => {
  it('refresh() populates the user and marks loaded', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue({ user_id: 'u_1', email: 'dev@localhost', name: 'Dev' })
    const auth = useAuthStore()
    expect(auth.isAuthenticated).toBe(false)
    await auth.refresh()
    expect(auth.isAuthenticated).toBe(true)
    expect(auth.user?.email).toBe('dev@localhost')
    expect(auth.loaded).toBe(true)
  })

  it('refresh() leaves user null when signed out', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue(null)
    const auth = useAuthStore()
    await auth.refresh()
    expect(auth.isAuthenticated).toBe(false)
    expect(auth.loaded).toBe(true)
  })

  it('ensureLoaded() only refreshes once', async () => {
    const spy = vi.spyOn(api, 'getMe').mockResolvedValue(null)
    const auth = useAuthStore()
    await auth.ensureLoaded()
    await auth.ensureLoaded()
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('logout() clears the user via the API', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue({ user_id: 'u_1', email: 'd@l', name: 'D' })
    const logoutSpy = vi.spyOn(api, 'logout').mockResolvedValue()
    const auth = useAuthStore()
    await auth.refresh()
    await auth.logout()
    expect(logoutSpy).toHaveBeenCalledOnce()
    expect(auth.isAuthenticated).toBe(false)
  })

  it('login() redirects into the OAuth flow', () => {
    const assign = vi.fn()
    vi.stubGlobal('location', { assign } as unknown as Location)
    useAuthStore().login()
    expect(assign).toHaveBeenCalledWith(api.loginUrl())
    vi.unstubAllGlobals()
  })

  // #1906 — offline survivability. Governing rule: only a 401/403 may destroy cached auth
  // state; a transport error never may.

  it('refresh() persists the identity to the device', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue(ME)
    await useAuthStore().refresh()
    expect(disk['auth.me']).toEqual(ME)
  })

  it('refresh() clears the snapshot when the credential is dead (401)', async () => {
    disk['auth.me'] = ME
    vi.spyOn(api, 'getMe').mockResolvedValue(null)
    const auth = useAuthStore()
    await auth.refresh()
    expect(auth.isAuthenticated).toBe(false)
    expect(disk['auth.me']).toBeUndefined()
  })

  it('refresh() does not throw offline, and signs in from the device snapshot', async () => {
    // This is the bug that made the app unusable rather than merely stale: the rejection aborted
    // App.vue's onMounted and threw out of the router guard on every navigation.
    disk['auth.me'] = ME
    vi.spyOn(api, 'getMe').mockRejectedValue(new TypeError('Failed to fetch'))
    const auth = useAuthStore()
    await expect(auth.refresh()).resolves.toBeUndefined()
    expect(auth.isAuthenticated).toBe(true)
    expect(auth.stale).toBe(true)
    // Latches, or the guard retries the failing call forever.
    expect(auth.loaded).toBe(true)
  })

  it('refresh() keeps a live user when the network drops', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue(ME)
    const auth = useAuthStore()
    await auth.refresh()
    vi.spyOn(api, 'getMe').mockRejectedValue(new TypeError('Failed to fetch'))
    await auth.refresh()
    expect(auth.isAuthenticated).toBe(true)
    expect(auth.stale).toBe(true)
    // A transport error must never destroy the snapshot.
    expect(disk['auth.me']).toEqual(ME)
  })

  it('refresh() stays signed out offline when there is no snapshot', async () => {
    vi.spyOn(api, 'getMe').mockRejectedValue(new TypeError('Failed to fetch'))
    const auth = useAuthStore()
    await auth.refresh()
    expect(auth.isAuthenticated).toBe(false)
    expect(auth.loaded).toBe(true)
  })

  it('hydrateFromDevice() paints the cached identity and marks it stale', async () => {
    disk['auth.me'] = ME
    const auth = useAuthStore()
    await auth.hydrateFromDevice()
    expect(auth.user?.email).toBe('dev@localhost')
    expect(auth.stale).toBe(true)
  })

  it('hydrateFromDevice() never overwrites an already-resolved user', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue(ME)
    const auth = useAuthStore()
    await auth.refresh()
    disk['auth.me'] = { ...ME, email: 'stale@old' }
    await auth.hydrateFromDevice()
    expect(auth.user?.email).toBe('dev@localhost')
    expect(auth.stale).toBe(false)
  })

  it('logout() drops the snapshot even when the server call fails', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue(ME)
    vi.spyOn(api, 'logout').mockRejectedValue(new TypeError('Failed to fetch'))
    const auth = useAuthStore()
    await auth.refresh()
    // Otherwise an offline sign-out leaves the snapshot and the next launch signs back in.
    await expect(auth.logout()).rejects.toThrow()
    expect(auth.isAuthenticated).toBe(false)
    expect(disk['auth.me']).toBeUndefined()
  })
})
