/**
 * Interests store (Pinia ↔ /api/app/interests) — the user's "profile of interests" that shapes
 * personalized discovery. Tokens are a mixed set: topic-clusters (`tc:`) from the picker, plus
 * topics (`topic:`) and people (`person:`) followed from entity cards. Mirrors the favorites store:
 * auth-gated (empty + no-op signed out), every mutation persists and refreshes from the server.
 */
import { defineStore } from 'pinia'
import { addInterest, getUserInterests, removeInterest } from '../services/api'
import { identityChangedSince, identityEpoch } from '../services/identity'
import { enqueue, isPermanent } from '../services/outbox'

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
    /**
     * Follow / unfollow a token. The server response is authoritative on success; a transient
     * failure keeps the optimistic flip AND queues the write to replay on reconnect, so the Follow
     * button is not a silent dead control offline (#2004 #7) — same contract as favourites. Only a
     * REFUSAL (4xx that is not 401/403) reverts, because replaying it would just fail again.
     */
    async toggle(token: string): Promise<void> {
      const wasFollowing = this.has(token)
      const generation = identityEpoch()
      // Optimistic flip so the tap is never swallowed.
      this.ids = wasFollowing ? this.ids.filter((t) => t !== token) : [...this.ids, token]
      try {
        const ids = wasFollowing ? await removeInterest(token) : await addInterest(token)
        if (identityChangedSince(generation)) return
        this.ids = ids
        this.loaded = true
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return
        if (isPermanent(err)) {
          // A refusal is an answer: undo the optimistic flip.
          this.ids = wasFollowing ? [...this.ids, token] : this.ids.filter((t) => t !== token)
          return
        }
        enqueue(wasFollowing ? { op: 'interest.remove', token } : { op: 'interest.add', token })
      }
    },
  },
})
