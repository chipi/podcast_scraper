/**
 * Auth store (Pinia). Resolves the signed-in user from the session cookie via GET /api/app/me.
 * Reads are open, so the app works signed-out; per-user features gate on `isAuthenticated`.
 */

import { defineStore } from 'pinia'
import { getAuthToken, getHealth, getMe, loginUrl, logout as apiLogout } from '../services/api'
import { CACHE_KEYS, clearCached } from '../services/contentCache'
import { getDeviceJson, removeDeviceKey, setDeviceJson } from '../services/deviceStore'
import { isNative, startNativeLogin, storeAuthToken } from '../services/native'
import { offlineReason } from '../composables/useOnline'
import { clearAuthEpoch, noteAuthEpoch } from '../services/authEpoch'
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
    /**
     * The ONE definition of "this device has a session" — for routing AND for chrome.
     *
     * The router guard and the masthead used to answer this differently. The guard counted a stored
     * native bearer as signed-in (so a cold-start transport failure would not strand a returning
     * user on the landing), while the masthead gated the avatar, the bell and the Library icon on
     * `isAuthenticated` alone. With a token present but no user resolved, the guard admitted you to
     * Home while the masthead showed "Sign in" and offered no way to reach Profile — reproduced on
     * the simulator 2026-09-16, and the reported "all profile screens are missing".
     *
     * Both now read THIS, so the two cannot disagree again.
     */
    hasSession(): boolean {
      // Read the token UNCONDITIONALLY, before any short-circuit can skip it. This is a Pinia
      // getter — a Vue computed — so it only re-runs when a reactive dependency changes, and a
      // dependency is only recorded if it is actually READ. Written as
      // `this.user !== null || (isNative() && !!getAuthToken())`, the `isNative()` check
      // short-circuits on web (and on the first native evaluation before a token exists), the
      // token ref is never touched, nothing is tracked, and the getter then caches its answer
      // forever — the masthead would never gain an avatar when a login stored a token.
      const token = getAuthToken()
      return this.user !== null || (isNative() && !!token)
    },
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
        if (me) {
          await setDeviceJson(SNAPSHOT_KEY, me)
        } else {
          // A null answer means the credential is genuinely dead (getMe maps 401 -> null), so the
          // cached CONTENT for that account goes with the snapshot — contentCache's own docstring
          // promised this and nothing did it (#1925 review C13).
          //
          // ...but ONLY when the server is actually healthy. A degraded server used to answer 401
          // for its OWN missing signing secret, and this line then deleted the user's entire
          // offline library over a fault that was not theirs and not their credential's — the
          // 2026-09-16 incident. The API now reports that case as 503 (so `getMe` rethrows and we
          // land in `catch`), and this guard is the belt to that braces: never destroy cached
          // content while we believe the server is unwell.
          await removeDeviceKey(SNAPSHOT_KEY)
          // Is this 401 OURS, or the platform's? A 401 alone cannot say: it is emitted both when
          // one user's token has aged out and when the server has rotated or lost its signing key
          // and invalidated EVERY token at once. Only the second is a server fault, and treating it
          // as the first is what made the 2026-09-16 incident destructive — it deleted the user's
          // offline library over an outage that was not theirs.
          //
          // Three independent signals, any of which means "not the user's fault": the connectivity
          // layer already considers the server degraded, health says it cannot authenticate at all,
          // or the session-key fingerprint has CHANGED since we last saw it (a mass invalidation).
          const health = await getHealth().catch(() => null)
          const keysRotated = noteAuthEpoch(health?.auth_epoch)
          const serverAtFault =
            offlineReason() === 'server' || health?.auth_ready === false || keysRotated
          if (!serverAtFault) await clearCached(CACHE_KEYS)
        }
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
    /**
     * Resolve auth once (no-op if already loaded) — used by the router guard.
     *
     * The FIRST paint must never block on the network. A returning user has a device snapshot
     * (#1906): resolve identity from it INSTANTLY and revalidate in the background, so the guard's
     * initial navigation renders immediately. Awaiting `refresh()` here meant awaiting `getMe()`,
     * which has no request timeout (`api.ts` apiFetch) and HANGS offline until the OS connection
     * timeout — the initial navigation stayed pending, the splash lifted over an empty RouterView,
     * and only a nav tap (re-running the guard after onMounted's hydrate set `loaded`) recovered it.
     */
    async ensureLoaded(): Promise<void> {
      if (this.loaded) return
      await this.hydrateFromDevice()
      if (this.loaded) {
        // Snapshot painted — revalidate without blocking the guard. `refresh()` never throws.
        void this.refresh()
        return
      }
      // No snapshot (a genuinely first, never-online launch): must resolve. `refresh()` latches
      // `loaded`, and login-first then routes to the lure landing.
      await this.refresh()
    },
    login(as?: string, returnTo?: string): void {
      if (isNative()) {
        // Native (#1310): iOS uses ASWebAuthenticationSession (prompt-free), Android the system
        // browser + intent-filter callback; both return the signed token → refresh() via
        // initNativeAuth's onAuthed. The in-app LoginView watch handles ?redirect after the token
        // lands, so native doesn't need return_to. A full-page redirect here would strand the WebView.
        void startNativeLogin(loginUrl(as, true))
        return
      }
      // Web: full-page redirect into the OAuth flow (Google in prod, mock provider in dev/e2e).
      // `as` is the dev-picker identity hint (mock provider only). `returnTo` carries the login-first
      // `?redirect` across the full-page OAuth bounce (RFC-120 #2009).
      window.location.assign(loginUrl(as, false, returnTo))
    },
    async logout(): Promise<void> {
      // Drop the local identity even if the server call fails — otherwise a sign-out with no
      // network leaves the snapshot behind and the next launch is silently signed back in.
      try {
        await apiLogout()
      } catch {
        // Offline, the server call cannot land — but the local identity still must go, and a
        // rejection here left the caller (App.vue's onSignOut) with an unhandled rejection and no
        // redirect: the header flipped to signed-out while the user sat on an authed view.
        // The session is stateless, so a client-side discard is the whole operation anyway.
      } finally {
        if (isNative()) storeAuthToken(null) // stateless token → client-side discard
        clearAuthEpoch()
        await removeDeviceKey(SNAPSHOT_KEY)
        this.user = null
        this.stale = false
      }
    },
    /**
     * Local-only sign-out for an EXPIRED session (RFC-120 #2009). The server session is already
     * dead (a 401 triggered this), so skip the API logout — just drop the local identity so the UI
     * and the login-first guard reflect signed-out and the 401 interceptor can route to /welcome.
     */
    markSignedOut(): void {
      if (isNative()) storeAuthToken(null)
      void removeDeviceKey(SNAPSHOT_KEY)
      this.user = null
      this.stale = false
    },
  },
})
