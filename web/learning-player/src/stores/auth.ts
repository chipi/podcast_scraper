/**
 * Auth store (Pinia). Resolves the signed-in user from the session cookie via GET /api/app/me.
 * Reads are open, so the app works signed-out; per-user features gate on `isAuthenticated`.
 */

import { defineStore } from 'pinia'
import { getMe, loginUrl, logout as apiLogout } from '../services/api'
import { isNative, openOAuth, storeAuthToken } from '../services/native'
import type { Me } from '../services/types'

interface AuthState {
  user: Me | null
  loaded: boolean
}

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({ user: null, loaded: false }),
  getters: {
    isAuthenticated: (s): boolean => s.user !== null,
  },
  actions: {
    async refresh(): Promise<void> {
      this.user = await getMe()
      this.loaded = true
    },
    /** Resolve auth once (no-op if already loaded) — used by the router guard. */
    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.refresh()
    },
    login(as?: string): void {
      if (isNative()) {
        // Native (#1310): open OAuth in the system browser; the backend returns the token via the
        // `closelistening://auth` deep link, handled by initNativeAuth() → refresh(). A full-page
        // redirect here would navigate the WebView away with no way back.
        void openOAuth(loginUrl(as, true))
        return
      }
      // Web: full-page redirect into the OAuth flow (Google in prod, mock provider in dev/e2e).
      // `as` is the dev-picker identity hint (mock provider only).
      window.location.assign(loginUrl(as))
    },
    async logout(): Promise<void> {
      await apiLogout()
      if (isNative()) storeAuthToken(null) // stateless token → client-side discard
      this.user = null
    },
  },
})
