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
    /**
     * Best-effort hydration — it does NOT reject.
     *
     * Six call sites treat it as fire-and-forget; four remembered `.catch(() => {})` and two did
     * not, so offline those two raised unhandled rejections. Patching the two call sites would
     * leave the trap armed for the seventh. Whether a follow-list is loaded is not something a
     * caller can act on, and every one of them already renders fine without it — so the failure
     * belongs here, swallowed once, rather than in each caller's memory. `load()` still throws for
     * anyone who genuinely wants to know.
     */
    async ensureLoaded(): Promise<void> {
      if (this.loaded) return
      try {
        await this.load()
      } catch {
        /* a follow-list we could not fetch is an empty one for now; the next load reconciles */
      }
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
