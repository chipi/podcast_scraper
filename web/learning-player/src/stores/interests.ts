/**
 * Interests store (Pinia ↔ /api/app/interests) — the user's "profile of interests" that shapes
 * personalized discovery. Tokens are a mixed set: topic-clusters (`tc:`) from the picker, plus
 * topics (`topic:`) and people (`person:`) followed from entity cards. Mirrors the favorites store:
 * auth-gated (empty + no-op signed out), every mutation persists and refreshes from the server.
 */
import { defineStore } from 'pinia'
import { addInterest, getUserInterests, removeInterest } from '../services/api'

interface InterestsState {
  ids: string[]
  loaded: boolean
}

export const useInterestsStore = defineStore('interests', {
  state: (): InterestsState => ({ ids: [], loaded: false }),
  getters: {
    /** Whether a token (cluster / topic / person id) is currently followed. */
    has:
      (s) =>
      (token: string): boolean =>
        s.ids.includes(token),
  },
  actions: {
    async load(): Promise<void> {
      this.ids = await getUserInterests()
      this.loaded = true
    },
    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.load()
    },
    /** Follow / unfollow a token; the server response is authoritative (no optimistic drift). */
    async toggle(token: string): Promise<void> {
      try {
        this.ids = this.has(token) ? await removeInterest(token) : await addInterest(token)
        this.loaded = true
      } catch {
        /* signed out / transient — leave state; next load reconciles with the server */
      }
    },
  },
})
