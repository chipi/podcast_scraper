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
import { followShow, getLibrary, unfollowShow } from '../services/api'
import type { LibraryItem } from '../services/types'

interface LibraryState {
  items: LibraryItem[]
  loaded: boolean
}

export const useLibraryStore = defineStore('library', {
  state: (): LibraryState => ({ items: [], loaded: false }),
  getters: {
    /** Whether a show is followed (drives the Follow / Following button state). */
    has:
      (s) =>
      (feedId: string): boolean =>
        s.items.some((i) => i.feed_id === feedId),
    feedIds: (s): string[] => s.items.map((i) => i.feed_id),
  },
  actions: {
    async load(): Promise<void> {
      this.items = await getLibrary()
      this.loaded = true
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
      } catch {
        // Signed out / transient — revert so the button flips back (that IS the user feedback);
        // the next load reconciles with the server. Swallowed like the interests/favorites stores
        // so `void store.toggle(...)` call sites can't raise an unhandled rejection.
        this.items = before
      }
    },
  },
})
