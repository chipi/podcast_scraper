/**
 * Library store (Pinia ↔ /api/app/library) — the shows the user follows (feed subscriptions).
 * This is what the "Your Week" digest reads for its "new in your follows" section; it is a
 * SEPARATE store from interests (topic:/person: tokens), which feed "Recommended for you".
 *
 * Mirrors the interests/favorites stores — auth-gated (empty + no-op signed out) — but the toggle
 * is optimistic: the button flips immediately, then reconciles with the authoritative server list,
 * and reverts if the call fails.
 */
import { defineStore } from 'pinia'
import { track } from '../services/analytics'
import { sourceForRoute } from '../services/provenance'
import { router } from '../router'
import { followShow, getLibrary, unfollowShow } from '../services/api'
import { isArrayCache, readCached, writeCached } from '../services/contentCache'
import { identityChangedSince, identityEpoch } from '../services/identity'
import { enqueue, isPermanent } from '../services/outbox'
import { serialWrites } from '../services/serialWrites'
import type { LibraryItem } from '../services/types'

interface LibraryState {
  items: LibraryItem[]
  loaded: boolean
  /** Showing a cached copy that has not been revalidated against the server (#1909). */
  stale: boolean
}

/** Every toggle write goes through the shared serializer (see `services/serialWrites`). */
const writes = serialWrites()

export const useLibraryStore = defineStore('library', {
  state: (): LibraryState => ({ items: [], loaded: false, stale: false }),
  getters: {
    /** Whether a show is followed (drives the Follow / Following button state). */
    has:
      (s) =>
      (feedId: string): boolean =>
        s.items.some((i) => i.feed_id === feedId),
    feedIds: (s): string[] => s.items.map((i) => i.feed_id),
  },
  actions: {
    /**
     * Revalidate, and fall back to the cached copy when the request never lands (#1909).
     * Never throws: offline this used to reject into hydrateUser and abort the rest of boot.
     */
    async load(): Promise<void> {
      const generation = identityEpoch()
      try {
        // `fresh`: a tap made while this is in flight must not be undone by it (serialWrites).
        const fresh = await writes.fresh(getLibrary)
        if (identityChangedSince(generation)) return
        this.items = fresh
        this.loaded = true
        this.stale = false
        void writeCached('library', this.items)
      } catch {
        if (identityChangedSince(generation)) return
        const cached = await readCached<LibraryItem[]>('library', isArrayCache)
        if (cached) {
          this.items = cached
          this.loaded = true
          this.stale = true
        }
        // NO cache and no answer: `loaded` deliberately stays FALSE.
        //
        // That flag is the only thing separating "you follow nothing" from "we could not ask", and
        // Library reads it to decide between its follows grid and an empty state that then offers
        // six shows to follow. Latching `loaded` here is what turned a failed fetch into a
        // confident, wrong answer about the user's own data.
      }
    },
    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.load()
    },
    /**
     * Follow / unfollow a show. Flips locally first so the button responds instantly, then swaps in
     * the server's list; on failure the pre-flip state is restored so the UI never claims a
     * subscription the server didn't take.
     */
    async toggle(feedId: string, meta: { title?: string | null } = {}): Promise<void> {
      const before = this.items
      const wasFollowing = this.has(feedId)
      // #2267. Reported on INTENT, next to the optimistic state change, not after the request:
      // `follow` is one of the spec's four Umami goals, and tying it to the response would drop
      // every follow made offline — which this store deliberately keeps, optimistically, precisely
      // because they are real. The surface comes from the route, since a follow can be tapped from
      // Home, an entity card, an entity page or the library and the spec asks which.
      track(wasFollowing ? 'unfollow' : 'follow', {
        kind: 'show',
        source: sourceForRoute(router.currentRoute.value.name),
      })
      this.items = wasFollowing
        ? before.filter((i) => i.feed_id !== feedId)
        : [...before, { feed_id: feedId, feed_url: null, title: meta.title ?? null, added_at: null }]
      const generation = identityEpoch()
      await writes.run(async (isLatest) => {
        try {
          const items = wasFollowing
            ? await unfollowShow(feedId)
            : await followShow(feedId, { title: meta.title })
          // A response landing after an account switch belongs to nobody now (advisor 1.4).
          if (identityChangedSince(generation)) return
          // A later tap is already showing its own state; its response will settle it.
          if (!isLatest()) return
          this.items = items
          this.loaded = true
        } catch (err: unknown) {
          if (identityChangedSince(generation)) return
          // `isPermanent`, not `instanceof ApiError` (advisor 2.2). Reverting on ANY ApiError
          // meant a single 502 destroyed the follow — and the whole reason that helper is exported
          // is that a bad gateway is not the server saying no. 401/403 is not a refusal either: a
          // dead session is repaired by signing in, so the intent is queued rather than thrown away.
          if (isPermanent(err)) {
            // The server ANSWERED and refused. Undo THIS tap's flip — the button must not claim a
            // subscription the server rejected — unless a later tap has superseded it.
            if (!isLatest()) return
            this.items = wasFollowing
              ? [...this.items, ...before.filter((i) => i.feed_id === feedId)]
              : this.items.filter((i) => i.feed_id !== feedId)
            return
          }
          // The request never landed. KEEP the flip and queue it (#1910): follow/unfollow is
          // item-level and idempotent, so a replay cannot corrupt anything, and reverting would
          // tell the user their action failed when it is merely delayed.
          enqueue(
            wasFollowing ? { op: 'unfollow', feedId } : { op: 'follow', feedId, title: meta.title },
          )
        }
      })
    },
  },
})
