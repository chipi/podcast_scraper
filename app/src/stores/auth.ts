/**
 * Auth store (Pinia). Resolves the signed-in user from the session cookie via GET /api/app/me.
 * Reads are open, so the app works signed-out; per-user features gate on `isAuthenticated`.
 */

import { defineStore } from 'pinia'
import { getMe, loginUrl } from '../services/api'
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
    login(): void {
      // Full-page redirect into the OAuth flow (Google in prod, mock provider in dev/e2e).
      window.location.assign(loginUrl())
    },
  },
})
