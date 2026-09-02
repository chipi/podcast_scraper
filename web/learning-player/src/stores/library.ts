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
import { ApiError, followShow, getLibrary, unfollowShow } from '../services/api'
import { readCached, writeCached } from '../services/contentCache'
import { enqueue } from '../services/outbox'
import type { LibraryItem } from '../services/types'

interface LibraryState {
  items: LibraryItem[]
  loaded: boolean
  /** Showing a cached copy that has not been revalidated against the server (#1909). */
  stale: boolean
}

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
      try {
        this.items = await getLibrary()
        this.loaded = true
        this.stale = false
        void writeCached('library', this.items)
      } catch {
        const cached = await readCached<LibraryItem[]>('library')
        if (cached) {
          this.items = cached
          this.loaded = true
          this.stale = true
        }
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
      this.items = wasFollowing
        ? before.filter((i) => i.feed_id !== feedId)
        : [...before, { feed_id: feedId, feed_url: null, title: meta.title ?? null, added_at: null }]
      try {
        this.items = wasFollowing
          ? await unfollowShow(feedId)
          : await followShow(feedId, { title: meta.title })
        this.loaded = true
      } catch (err: unknown) {
        if (err instanceof ApiError) {
          // The server ANSWERED and refused (401, 4xx). Revert — the button must not claim a
          // subscription the server rejected, and queueing a refused write would replay a failure.
          this.items = before
          return
        }
        // The request never landed. KEEP the flip and queue it (#1910): follow/unfollow is
        // item-level and idempotent, so a replay cannot corrupt anything, and reverting would tell
        // the user their action failed when it is merely delayed.
        enqueue(wasFollowing ? { op: 'unfollow', feedId } : { op: 'follow', feedId, title: meta.title })
      }
    },
  },
})
