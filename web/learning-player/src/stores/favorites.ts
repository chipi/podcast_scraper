/**
 * Favorites store (Pinia ↔ GET/PUT/DELETE /api/app/favorites) — the polymorphic "saved things"
 * the user collects (episodes, insights, … later people/topics). Mirrors the queue store: auth-gated
 * (empty + no-op signed out), every mutation persists and refreshes from the server response.
 */
import { defineStore } from 'pinia'
import { addFavorite, getFavorites, removeFavorite } from '../services/api'
import { readCached, writeCached } from '../services/contentCache'
import { identityChangedSince, identityEpoch } from '../services/identity'
import { enqueue, isPermanent } from '../services/outbox'
import type { EpisodeSummary, FavoriteAdd, FavoriteInsight } from '../services/types'

interface FavoritesState {
  episodes: EpisodeSummary[]
  insights: FavoriteInsight[]
  loaded: boolean
  /** Showing a cached copy not yet revalidated (#1909). */
  stale: boolean
  /**
   * Offline toggles the server has not confirmed, keyed `kind:ref` → the state the user asked for.
   *
   * The heart used to sit unchanged after an offline tap: the write went to the outbox and local
   * state was left alone, so the control read as a dead button while the app had in fact recorded
   * the intent (#1925 review). This carries the intent WITHOUT fabricating a list entry — an
   * offline `add` has no EpisodeSummary to show, and inventing one would put a card with a blank
   * title in the favourites list. So: the heart flips, the list waits for the server.
   */
  pendingFlips: Record<string, boolean>
}

function flipKey(kind: string, ref: string): string {
  return `${kind}:${ref}`
}

export const useFavoritesStore = defineStore('favorites', {
  state: (): FavoritesState => ({
    episodes: [],
    insights: [],
    loaded: false,
    stale: false,
    pendingFlips: {},
  }),
  getters: {
    /** Whether a given item is saved (drives the heart toggle state). */
    has:
      (s) =>
      (kind: string, ref: string): boolean => {
        // An unconfirmed offline toggle wins over the list: it is the newer of the two truths.
        const pending = s.pendingFlips[flipKey(kind, ref)]
        if (pending !== undefined) return pending
        return kind === 'episode'
          ? s.episodes.some((e) => e.slug === ref)
          : kind === 'insight'
            ? s.insights.some((i) => i.ref === ref)
            : false
      },
    count: (s): number => s.episodes.length + s.insights.length,
  },
  actions: {
    /** Revalidate, falling back to the cached copy when the request never lands (#1909). */
    async load(): Promise<void> {
      const generation = identityEpoch()
      try {
        const f = await getFavorites()
        if (identityChangedSince(generation)) return
        this.episodes = f.episodes
        this.insights = f.insights
        this.loaded = true
        this.stale = false
        // A successful read is the server's answer, and the outbox is flushed BEFORE the reconnect
        // revalidation (App.vue), so anything still pending here has already been applied.
        this.pendingFlips = {}
        void writeCached('favorites', { episodes: f.episodes, insights: f.insights })
      } catch {
        const cached = await readCached<Pick<FavoritesState, 'episodes' | 'insights'>>('favorites')
        if (cached) {
          this.episodes = cached.episodes
          this.insights = cached.insights
          this.loaded = true
          this.stale = true
        }
      }
    },
    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.load()
    },
    /** Toggle a favorite; the server response is authoritative (no optimistic drift). */
    async toggle(item: FavoriteAdd): Promise<void> {
      const wasFavorite = this.has(item.kind, item.ref)
      try {
        const f = wasFavorite
          ? await removeFavorite(item.kind, item.ref)
          : await addFavorite(item)
        this.episodes = f.episodes
        this.insights = f.insights
        this.loaded = true
        delete this.pendingFlips[flipKey(item.kind, item.ref)]
      } catch (err: unknown) {
        // Only a request that never LANDED is queued. A server REFUSAL is an answer, and
        // replaying it would just fail again — but a 502/408/429 is not a refusal, so it queues
        // like any other unanswered write (#1925).
        if (isPermanent(err)) return
        // Add/remove of one favourite is item-level and idempotent, so a replay lands on the same
        // state (#1910). The heart flips now so the tap is not silently swallowed; the LIST waits
        // for the server, except for a removal, which we can represent exactly.
        this.pendingFlips[flipKey(item.kind, item.ref)] = !wasFavorite
        if (wasFavorite) {
          this.episodes = this.episodes.filter((e) => e.slug !== item.ref)
          this.insights = this.insights.filter((i) => i.ref !== item.ref)
        }
        enqueue(
          wasFavorite
            ? { op: 'favorite.remove', kind: item.kind, ref: item.ref }
            : { op: 'favorite.add', kind: item.kind, ref: item.ref },
        )
      }
    },
  },
})
