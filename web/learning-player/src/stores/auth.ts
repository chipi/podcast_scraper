/**
 * Auth store (Pinia). Resolves the signed-in user from the session cookie via GET /api/app/me.
 * Reads are open, so the app works signed-out; per-user features gate on `isAuthenticated`.
 */

import { defineStore } from 'pinia'
import { getMe, loginUrl, logout as apiLogout } from '../services/api'
import { getDeviceJson, removeDeviceKey, setDeviceJson } from '../services/deviceStore'
import { isNative, startNativeLogin, storeAuthToken } from '../services/native'
import type { Me } from '../services/types'

/**
 * Device-persisted identity (#1906). The bearer token already survives a restart
 * (`services/native.ts`), but the `Me` it resolves to did not — so an offline launch had no user,
 * every `requiresAuth` route was unreachable, and `refresh()` REJECTING on a transport error
 * aborted boot entirely (`App.vue` onMounted) and threw out of the router guard.
 *
 * The governing rule, which every cached read in this app should follow:
 * **only a 401/403 may destroy cached auth state; a transport error never may.**
 */
const SNAPSHOT_KEY = 'auth.me'

interface AuthState {
  user: Me | null
  loaded: boolean
  /** `user` came from the device snapshot and has not been revalidated against the server yet. */
  stale: boolean
}

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({ user: null, loaded: false, stale: false }),
  getters: {
    isAuthenticated: (s): boolean => s.user !== null,
  },
  actions: {
    /**
     * Paint the last known identity from the device before touching the network, so an offline
     * launch is signed in immediately instead of after a connect timeout.
     */
    async hydrateFromDevice(): Promise<void> {
      if (this.user) return
      const cached = await getDeviceJson<Me>(SNAPSHOT_KEY)
      if (!cached) return
      this.user = cached
      this.stale = true
      this.loaded = true
    },

    /**
     * Revalidate against the server. NEVER throws: a rejection here used to abort boot and the
     * router guard, which is why the app was unusable rather than merely stale when offline.
     */
    async refresh(): Promise<void> {
      try {
        const me = await getMe()
        this.user = me
        this.stale = false
        // `getMe` maps 401 -> null, so a null answer means the credential is genuinely dead and
        // the snapshot must go with it. Anything else that resolves is a real identity.
        if (me) await setDeviceJson(SNAPSHOT_KEY, me)
        else await removeDeviceKey(SNAPSHOT_KEY)
      } catch {
        // Transport or server failure — NOT an auth failure. Keep whatever identity we have, and
        // fall back to the device snapshot if this is a cold offline start.
        await this.hydrateFromDevice()
        this.stale = true
      } finally {
        // Latches either way: while this stayed false the router guard re-ran the failing call on
        // every navigation, forever.
        this.loaded = true
      }
    },
    /** Resolve auth once (no-op if already loaded) — used by the router guard. */
    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.refresh()
    },
    login(as?: string): void {
      if (isNative()) {
        // Native (#1310): iOS uses ASWebAuthenticationSession (prompt-free), Android the system
        // browser + intent-filter callback; both return the signed token → refresh() via
        // initNativeAuth's onAuthed. A full-page redirect here would strand the WebView.
        void startNativeLogin(loginUrl(as, true))
        return
      }
      // Web: full-page redirect into the OAuth flow (Google in prod, mock provider in dev/e2e).
      // `as` is the dev-picker identity hint (mock provider only).
      window.location.assign(loginUrl(as))
    },
    async logout(): Promise<void> {
      // Drop the local identity even if the server call fails — otherwise a sign-out with no
      // network leaves the snapshot behind and the next launch is silently signed back in.
      try {
        await apiLogout()
      } finally {
        if (isNative()) storeAuthToken(null) // stateless token → client-side discard
        await removeDeviceKey(SNAPSHOT_KEY)
        this.user = null
        this.stale = false
      }
    },
  },
})
