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
import { serialWrites } from '../services/serialWrites'

/** Every toggle write goes through the shared serializer (see `services/serialWrites`). */
const writes = serialWrites()

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
      // `fresh`: a tap made while this is in flight must not be undone by it (serialWrites).
      this.ids = await writes.fresh(getUserInterests)
      this.loaded = true
    },
    /**
     * Adopt the authoritative set returned by an absolute write.
     *
     * `InterestsPicker` PUTs the whole list rather than toggling, so it cannot use `toggle` — and
     * before this existed it updated NOTHING here. Each parent was left to cope: `ProfileView`
     * assigned a local ref (so Profile looked right) and `HomeView` set its DISMISSED flag (so Home
     * looked right for the wrong reason, and only on the path that starts from Home). Choose
     * interests from Profile and Home went on showing "Personalize your Home", because this store —
     * which is what `showInterestsCard` reads — still held the empty list it loaded at boot.
     * `ensureLoaded()` short-circuits on `loaded`, so nothing ever refetched it either.
     *
     * Present since #1111 (2026-06-28) and caught by `PersonalisationTests.test10` on device.
     */
    replaceAll(ids: string[]): void {
      this.ids = ids
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
      await writes.run(async (isLatest) => {
        try {
          const ids = wasFollowing ? await removeInterest(token) : await addInterest(token)
          if (identityChangedSince(generation)) return
          // A later tap is already showing its own optimistic state; its response will settle it.
          if (!isLatest()) return
          this.ids = ids
          this.loaded = true
        } catch (err: unknown) {
          if (identityChangedSince(generation)) return
          if (isPermanent(err)) {
            // A refusal is an answer: undo the optimistic flip — unless a later tap superseded it.
            if (!isLatest()) return
            this.ids = wasFollowing
              ? this.ids.includes(token)
                ? this.ids
                : [...this.ids, token]
              : this.ids.filter((t) => t !== token)
            return
          }
          enqueue(wasFollowing ? { op: 'interest.remove', token } : { op: 'interest.add', token })
        }
      })
    },
  },
})
