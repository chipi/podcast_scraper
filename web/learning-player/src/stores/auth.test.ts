import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import * as deviceStore from '../services/deviceStore'
import * as native from '../services/native'
import * as online from '../composables/useOnline'
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
    // The server must be stated HEALTHY, or this asserts nothing about a dead credential.
    // It used to pass without saying so, because the snapshot was removed unconditionally ahead of
    // the `serverAtFault` check — so the test held for a 401 of ANY origin, including one the code
    // had judged to be the server's fault. Under the guard, jsdom's `offlineReason()` reports a
    // degraded server and the identity is (correctly) kept. The intent is unchanged; the
    // precondition that intent needs is now supplied rather than left to chance.
    vi.spyOn(api, 'getHealth').mockResolvedValue({ auth_ready: true } as never)
    vi.spyOn(online, 'offlineReason').mockReturnValue(null)
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

  it('ensureLoaded() resolves from the device snapshot without awaiting a hanging network', async () => {
    // Blank-screen-after-splash bug (offline cold-start): the router guard awaits ensureLoaded().
    // When getMe() HANGS offline (no request timeout — api.ts apiFetch), awaiting refresh() left
    // the INITIAL navigation pending forever, so RouterView stayed empty behind the lifted splash.
    // It recovered only when a nav tap re-ran the guard after onMounted's hydrateFromDevice had
    // set `loaded`. Fix: resolve identity from the instant device snapshot first, revalidate in
    // the background — never block the first paint on the network.
    disk['auth.me'] = ME
    let settled = false
    // A promise that never settles within the test — models the offline connection hang.
    const getMeSpy = vi.spyOn(api, 'getMe').mockReturnValue(
      new Promise<typeof ME | null>(() => {
        /* intentionally never resolves */
      }),
    )
    const auth = useAuthStore()

    await auth.ensureLoaded().then(() => {
      settled = true
    })

    expect(settled).toBe(true)
    expect(auth.isAuthenticated).toBe(true)
    expect(auth.loaded).toBe(true)
    expect(auth.stale).toBe(true)
    // Revalidation still fires in the background — it just must not be awaited. Guards against a
    // future "simplification" that drops the background refresh and leaves the snapshot unverified.
    expect(getMeSpy).toHaveBeenCalled()
  }, 2000)

  it('logout() drops the snapshot even when the server call fails', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue(ME)
    vi.spyOn(api, 'logout').mockRejectedValue(new TypeError('Failed to fetch'))
    const auth = useAuthStore()
    await auth.refresh()
    // Must NOT reject: App.vue awaits this and then redirects. A rejection stranded the user on
    // an authed view with the header already flipped to signed-out.
    await expect(auth.logout()).resolves.toBeUndefined()
    expect(auth.isAuthenticated).toBe(false)
    expect(disk['auth.me']).toBeUndefined()
  })
})


describe('a dead credential takes the TOKEN with it (2026-09-24)', () => {
  /*
   * `hasSession` on native is `user !== null || (isNative() && token)`. `refresh()` dropped the
   * user, the snapshot and the cached content on a genuine 401 — and left the bearer token. The app
   * then looked signed in with no identity: the router guard admitted every authed route, each call
   * 401'd, and offline there was no snapshot to paint. A session that exists for the guard and for
   * nobody else.
   *
   * Reachable in the field, not a lab case: the server rotates its signing key while the app is
   * online, and the next launch is stranded.
   */
  /** Native, with a token, and the server healthy unless a test says otherwise. */
  function nativeWithToken() {
    vi.spyOn(native, 'isNative').mockReturnValue(true)
    // `serverAtFault` has THREE inputs; a test that controls one and leaves two to chance is
    // asserting about whatever jsdom happens to report. Pin the connectivity one here.
    vi.spyOn(online, 'offlineReason').mockReturnValue(null)
    return vi.spyOn(native, 'storeAuthToken').mockImplementation(() => {})
  }

  it('clears the token when the 401 is OURS and the server is healthy', async () => {
    const store = nativeWithToken()
    vi.spyOn(api, 'getMe').mockResolvedValue(null)
    vi.spyOn(api, 'getHealth').mockResolvedValue({ auth_ready: true } as never)

    await useAuthStore().refresh()

    expect(store).toHaveBeenCalledWith(null)
  })

  it('KEEPS the token when the server is the one at fault', async () => {
    // Signing everyone out over an outage is the 2026-09-16 incident in a different shape. The
    // token follows the same `serverAtFault` guard the cached content already had.
    const store = nativeWithToken()
    vi.spyOn(api, 'getMe').mockResolvedValue(null)
    vi.spyOn(api, 'getHealth').mockResolvedValue({ auth_ready: false } as never)

    await useAuthStore().refresh()

    expect(store).not.toHaveBeenCalled()
  })

  it('KEEPS the token on a transport failure — offline is not a dead credential', async () => {
    const store = nativeWithToken()
    vi.spyOn(api, 'getMe').mockRejectedValue(new Error('offline'))

    await useAuthStore().refresh()

    expect(store).not.toHaveBeenCalled()
  })
})


describe('a server-fault 401 keeps the IDENTITY, not just the token (2026-09-24)', () => {
  /*
   * Measured on device: across a full run the bearer token was present in all 34 samples and
   * `auth.me` in none, while the downloads registry — written through the same Preferences path —
   * persisted fine. The snapshot was not failing to write. It was written, then deleted by a 401
   * that this code had ALREADY judged not to be the user's fault.
   *
   * `removeDeviceKey(SNAPSHOT_KEY)` sat ahead of the `serverAtFault` check, so a rotated signing
   * key kept the token (right) and destroyed the identity anyway (wrong) — leaving the worst of
   * both: a session the router admits that cannot name its user, with nothing left to paint
   * offline. That is exactly the offline promise this app makes to someone on a plane.
   */
  function nativeSession() {
    vi.spyOn(native, 'isNative').mockReturnValue(true)
    vi.spyOn(online, 'offlineReason').mockReturnValue(null)
    return vi.spyOn(native, 'storeAuthToken').mockImplementation(() => {})
  }

  it('keeps the snapshot when the server rotated its signing key', async () => {
    nativeSession()
    const auth = useAuthStore()
    // A real identity lands first, exactly as a successful login would.
    vi.spyOn(api, 'getMe').mockResolvedValue(ME)
    await auth.refresh()
    expect(disk['auth.me']).toBeTruthy()

    // Now the server invalidates every token at once and answers 401.
    vi.spyOn(api, 'getMe').mockResolvedValue(null)
    vi.spyOn(api, 'getHealth').mockResolvedValue({ auth_ready: false } as never)
    await auth.refresh()

    expect(disk['auth.me'], 'the identity was destroyed over a fault that was not the user\'s').toBeTruthy()
  })

  it('still drops the snapshot when the credential is genuinely ours and dead', async () => {
    // The other half: a real expiry must still sign you out cleanly, or the guard becomes a
    // licence to keep stale identities forever.
    nativeSession()
    const auth = useAuthStore()
    vi.spyOn(api, 'getMe').mockResolvedValue(ME)
    await auth.refresh()
    expect(disk['auth.me']).toBeTruthy()

    vi.spyOn(api, 'getMe').mockResolvedValue(null)
    vi.spyOn(api, 'getHealth').mockResolvedValue({ auth_ready: true } as never)
    await auth.refresh()

    expect(disk['auth.me']).toBeFalsy()
  })
})

