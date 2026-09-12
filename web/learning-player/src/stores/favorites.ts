/**
 * Favorites store (Pinia ↔ GET/PUT/DELETE /api/app/favorites) — the "saved things" the user
 * collects (episodes; later people/topics/shows/storylines per RFC-121). Insights are NOT favorites
 * — they save via the highlights/capture path. Mirrors the queue store: auth-gated (empty + no-op
 * signed out), every mutation persists and refreshes from the server response.
 */
import { defineStore } from 'pinia'
import { addFavorite, getFavorites, removeFavorite, setFavoriteColor } from '../services/api'
import { hasArrayFields, readCached, writeCached } from '../services/contentCache'
import { identityChangedSince, identityEpoch } from '../services/identity'
import { enqueue, isPermanent } from '../services/outbox'
import type { EpisodeSummary, FavoriteAdd, FavoriteEntity, FavoriteKind } from '../services/types'

interface FavoritesState {
  episodes: EpisodeSummary[]
  /** Saved non-episode favorites (show / topic / person / storyline). */
  entities: FavoriteEntity[]
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
    entities: [],
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
          : s.entities.some((e) => e.kind === kind && e.ref === ref)
      },
    count: (s): number => s.episodes.length + s.entities.length,
  },
  actions: {
    /** Revalidate, falling back to the cached copy when the request never lands (#1909). */
    async load(): Promise<void> {
      const generation = identityEpoch()
      try {
        const f = await getFavorites()
        if (identityChangedSince(generation)) return
        this.episodes = f.episodes
        this.entities = f.entities ?? []
        this.loaded = true
        this.stale = false
        // A successful read is the server's answer, and the outbox is flushed BEFORE the reconnect
        // revalidation (App.vue), so anything still pending here has already been applied.
        this.pendingFlips = {}
        void writeCached('favorites', { episodes: f.episodes, entities: this.entities })
      } catch {
        const cached = await readCached<Pick<FavoritesState, 'episodes' | 'entities'>>(
          'favorites',
          hasArrayFields('episodes', 'entities'),
        )
        if (cached) {
          this.episodes = cached.episodes
          this.entities = cached.entities ?? []
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
      const generation = identityEpoch()
      try {
        const f = wasFavorite
          ? await removeFavorite(item.kind, item.ref)
          : await addFavorite(item)
        // A response that lands after an account switch belongs to nobody now (advisor 1.4).
        if (identityChangedSince(generation)) return
        this.episodes = f.episodes
        this.entities = f.entities ?? []
        this.loaded = true
        delete this.pendingFlips[flipKey(item.kind, item.ref)]
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return
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
          this.entities = this.entities.filter((e) => !(e.kind === item.kind && e.ref === item.ref))
        }
        enqueue(
          wasFavorite
            ? { op: 'favorite.remove', kind: item.kind, ref: item.ref }
            : { op: 'favorite.add', kind: item.kind, ref: item.ref },
        )
      }
    },
    /** Paint a colour onto the matching in-memory row (optimistic; server response reconciles). */
    _paintColor(kind: FavoriteKind, ref: string, color: string | null): void {
      if (kind === 'episode') {
        const ep = this.episodes.find((e) => e.slug === ref)
        if (ep) ep.color = color
      } else {
        const ent = this.entities.find((e) => e.kind === kind && e.ref === ref)
        if (ent) ent.color = color
      }
    },
    /** Set (token) or clear (null) a saved item's colour (RFC-121 ph. 4).
     *
     * Optimistic like the rest of the store's edits: paint locally, then let the server response be
     * authoritative. A permanent refusal (404 — the favorite is gone) can only be reconciled by a
     * reload; a request that never landed queues, and the local paint stands until reconnect. */
    async setColor(kind: FavoriteKind, ref: string, color: string | null): Promise<void> {
      this._paintColor(kind, ref, color)
      const generation = identityEpoch()
      try {
        const f = await setFavoriteColor(kind, ref, color)
        if (identityChangedSince(generation)) return
        this.episodes = f.episodes
        this.entities = f.entities ?? []
        this.loaded = true
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return
        if (isPermanent(err)) {
          await this.load()
          return
        }
        enqueue({ op: 'favorite.color', kind, ref, color })
      }
    },
  },
})
