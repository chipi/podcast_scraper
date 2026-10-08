/**
 * Favorites store (Pinia ↔ /api/app/favorites) — WHICH things the user saved (episodes, shows,
 * topics, people, storylines, themes). Insights are NOT favorites — they save via the
 * highlights/capture path. Mirrors the queue store: auth-gated (empty + no-op signed out), every
 * mutation persists and refreshes from the server response.
 *
 * Identity only (kind + ref + colour), since 2026-10-08: the hearts need to know what is saved, not
 * every saved episode's card. The Saved tab reads its own pages from the server
 * (`useFavoritesPage`) and refetches when `version` moves.
 */
import { defineStore } from 'pinia'
import {
  addFavorite,
  favoriteRefsOf,
  getFavoriteRefs,
  removeFavorite,
  setFavoriteColor,
} from '../services/api'
import { hasArrayFields, readCached, writeCached } from '../services/contentCache'
import { identityChangedSince, identityEpoch } from '../services/identity'
import { enqueue, isPermanent } from '../services/outbox'
import { serialWrites } from '../services/serialWrites'
import type { FavoriteAdd, FavoriteKind, FavoriteRef } from '../services/types'

interface FavoritesState {
  /** Every saved item's identity, newest first. */
  items: FavoriteRef[]
  /** Moves on every change the server confirms (or a removal applied locally) — lists refetch. */
  version: number
  loaded: boolean
  /** Showing a cached copy not yet revalidated (#1909). */
  stale: boolean
  /**
   * Toggles the server has not confirmed, keyed `kind:ref` → the state the user asked for. Set on
   * EVERY tap since 2026-09-30 (it used to be offline taps only), cleared by the server's answer.
   *
   * The heart used to sit unchanged after an offline tap: the write went to the outbox and local
   * state was left alone, so the control read as a dead button while the app had in fact recorded
   * the intent (#1925 review). This carries the intent WITHOUT fabricating a list entry — an
   * offline `add` has no EpisodeSummary to show, and inventing one would put a card with a blank
   * title in the favourites list. So: the heart flips, the list waits for the server.
   */
  pendingFlips: Record<string, boolean>
}

/** Every toggle write goes through the shared serializer (see `services/serialWrites`). */
const writes = serialWrites()
/**
 * Flips whose write is sitting in the OFFLINE outbox. The newest successful answer is the server's
 * whole list, so it retires every other flip — including one whose write was refused while a later
 * tap was in flight — but these must survive until the outbox has replayed them.
 */
const queuedFlips = new Set<string>()

function flipKey(kind: string, ref: string): string {
  return `${kind}:${ref}`
}

export const useFavoritesStore = defineStore('favorites', {
  state: (): FavoritesState => ({
    items: [],
    version: 0,
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
        return s.items.some((i) => i.kind === kind && i.ref === ref)
      },
    count: (s): number => s.items.length,
    /** How many of one kind are saved (drives which Saved sections and type chips exist). */
    countOf:
      (s) =>
      (kind: FavoriteKind): number =>
        s.items.filter((i) => i.kind === kind).length,
  },
  actions: {
    /** Revalidate, falling back to the cached copy when the request never lands (#1909). */
    async load(): Promise<void> {
      const generation = identityEpoch()
      try {
        // `fresh`: a tap made while this is in flight must not be undone by it (serialWrites).
        const items = await writes.fresh(getFavoriteRefs)
        if (identityChangedSince(generation)) return
        this._set(items)
        this.loaded = true
        this.stale = false
        // A successful read is the server's answer, and the outbox is flushed BEFORE the reconnect
        // revalidation (App.vue), so anything still pending here has already been applied.
        this.pendingFlips = {}
        queuedFlips.clear()
        void writeCached('favorite-refs', { items })
      } catch {
        const cached = await readCached<Pick<FavoritesState, 'items'>>(
          'favorite-refs',
          hasArrayFields('items'),
        )
        if (cached) {
          this._set(cached.items)
          this.loaded = true
          this.stale = true
        }
      }
    },
    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.load()
    },
    /**
     * Toggle a favorite. The heart flips on the tap (`pendingFlips`, the newer truth `has` reads
     * first); the LIST waits for the server, whose answer is authoritative and clears the flip.
     * Writes are serialised and only the newest tap's answer is applied (`services/serialWrites`).
     */
    async toggle(item: FavoriteAdd): Promise<void> {
      const key = flipKey(item.kind, item.ref)
      const wasFavorite = this.has(item.kind, item.ref)
      const generation = identityEpoch()
      this.pendingFlips[key] = !wasFavorite
      await writes.run(async (isLatest) => {
        try {
          const f = wasFavorite
            ? await removeFavorite(item.kind, item.ref)
            : await addFavorite(item)
          // A response that lands after an account switch belongs to nobody now (advisor 1.4).
          if (identityChangedSince(generation)) return
          // A later tap is already showing its own state; its answer will settle it.
          if (!isLatest()) return
          this._set(favoriteRefsOf(f))
          this.loaded = true
          // Every earlier write has finished (they are serialised), so this list reflects them all;
          // only flips still waiting in the outbox say more than it does.
          this.pendingFlips = Object.fromEntries(
            Object.entries(this.pendingFlips).filter(([k]) => queuedFlips.has(k)),
          )
        } catch (err: unknown) {
          if (identityChangedSince(generation)) return
          // Only a request that never LANDED is queued. A server REFUSAL is an answer, and
          // replaying it would just fail again — but a 502/408/429 is not a refusal, so it queues
          // like any other unanswered write (#1925).
          if (isPermanent(err)) {
            // The refusal stands: drop the flip so the heart shows the server's state again —
            // unless a later tap has superseded this one.
            if (isLatest()) delete this.pendingFlips[key]
            return
          }
          // Add/remove of one favourite is item-level and idempotent, so a replay lands on the same
          // state (#1910). The flip stays; the LIST waits for the server, except for a removal,
          // which we can represent exactly.
          if (wasFavorite) {
            this._set(this.items.filter((i) => !(i.kind === item.kind && i.ref === item.ref)))
          }
          queuedFlips.add(key)
          enqueue(
            wasFavorite
              ? { op: 'favorite.remove', kind: item.kind, ref: item.ref }
              : { op: 'favorite.add', kind: item.kind, ref: item.ref },
          )
        }
      })
    },
    /** Replace the identities and tell every Saved list to refetch. */
    _set(items: FavoriteRef[]): void {
      this.items = items
      this.version++
    },
    /** Paint a colour onto the matching item (optimistic; the server response reconciles). */
    _paintColor(kind: FavoriteKind, ref: string, color: string | null): void {
      this._set(this.items.map((i) => (i.kind === kind && i.ref === ref ? { ...i, color } : i)))
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
        this._set(favoriteRefsOf(f))
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
