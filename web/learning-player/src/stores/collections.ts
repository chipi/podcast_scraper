/**
 * The user's collections (RFC-119) — built like every other per-user feature (#2013).
 *
 * ## Why this exists
 *
 * Collections was the only per-user data with no store, no cached read and no outbox replay.
 * Favourites, queue, highlights, notes and follows all have the full set. So a transient failure
 * that hits every feature was invisible in all of them and permanent data loss in this one: the
 * create appeared to work, the read came back empty, and the collection was gone. That asymmetry —
 * not a collections-specific bug — is what shipped.
 *
 * The write half was fixed first (outbox ops `collection.create` / `collection.addItem`). This is
 * the read half plus identity handling: a failed read now falls back to the cached copy instead of
 * rendering "you have no collections", which is the specific lie that made the data look lost.
 *
 * ## `stale` is part of the contract
 *
 * A cached answer is not the server's answer. Callers that care can tell the difference; callers
 * that do not still get a list rather than a false empty state. Same shape as `favorites`.
 */

import { defineStore } from 'pinia'
import { getCollections } from '../services/api'
import { readCached, writeCached } from '../services/contentCache'
import { identityChangedSince, identityEpoch } from '../services/identity'
import type { Collection } from '../services/types'

interface CollectionsState {
  items: Collection[]
  loaded: boolean
  /** True when `items` came from cache because the request did not land. */
  stale: boolean
  /** True when the last read failed AND there was no cache to fall back to. */
  unavailable: boolean
}

export const useCollectionsStore = defineStore('collections', {
  state: (): CollectionsState => ({ items: [], loaded: false, stale: false, unavailable: false }),

  getters: {
    count: (s): number => s.items.length,
    byId:
      (s) =>
      (id: string): Collection | undefined =>
        s.items.find((c) => c.id === id),
  },

  actions: {
    /**
     * Revalidate, falling back to the cached copy when the request never lands.
     *
     * `loaded` latches only on a real answer or a cache hit — never on a bare failure, because a
     * latched empty list is exactly the "you have none" lie.
     */
    async load(): Promise<void> {
      const generation = identityEpoch()
      try {
        const items = await getCollections()
        // A response landing after an account switch belongs to nobody now.
        if (identityChangedSince(generation)) return
        this.items = items
        this.loaded = true
        this.stale = false
        this.unavailable = false
        void writeCached('collections', { items })
      } catch {
        if (identityChangedSince(generation)) return
        const cached = await readCached<{ items: Collection[] }>('collections')
        if (cached) {
          this.items = cached.items
          this.loaded = true
          this.stale = true
          this.unavailable = false
        } else {
          // Nothing to show and no answer — the caller must render an ERROR, not an empty state.
          this.items = []
          this.unavailable = true
        }
      }
    },

    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.load()
    },

    /** Reflect a collection the UI just created or changed, without a round-trip. */
    upsert(collection: Collection): void {
      const i = this.items.findIndex((c) => c.id === collection.id)
      if (i >= 0) this.items[i] = collection
      else this.items = [collection, ...this.items]
      this.loaded = true
      void writeCached('collections', { items: this.items })
    },

    remove(id: string): void {
      this.items = this.items.filter((c) => c.id !== id)
      void writeCached('collections', { items: this.items })
    },
  },
})
