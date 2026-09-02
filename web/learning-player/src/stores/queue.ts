/**
 * Play-queue store (Pinia ↔ GET/PUT /api/app/queue, RFC-099 §4). Ordered episode slugs;
 * auth-gated (empty + no-op when signed out). Every mutation mirrors to the server.
 * Auto-advance is driven by the player calling `nextAfter(slug)` on `ended`.
 */

import { defineStore } from 'pinia'
import { getQueue, putQueue } from '../services/api'

interface QueueState {
  items: string[]
  loaded: boolean
}

// Single in-flight load promise (module-scoped: the store is an app singleton). Guards
// against a SECOND GET /queue racing the first — e.g. the initial mount load() vs a mutation's
// ensureLoaded(). Without it a late-resolving load() overwrites `items` with stale server data
// and silently drops an optimistic add ("queue empty" after add; RFC-099 §4).
let inflightLoad: Promise<boolean> | null = null

export const useQueueStore = defineStore('queue', {
  state: (): QueueState => ({ items: [], loaded: false }),
  getters: {
    has:
      (s) =>
      (slug: string): boolean =>
        s.items.includes(slug),
    count: (s): number => s.items.length,
  },
  actions: {
    /**
     * Returns whether the queue is now loaded. NEVER throws (#1906): offline this rejected and the
     * rejection travelled through every mutation into `void store.toggle()` call sites.
     */
    async load(): Promise<boolean> {
      // Coalesce concurrent loads onto one GET so nothing clobbers an in-progress mutation.
      if (inflightLoad) return inflightLoad
      inflightLoad = (async (): Promise<boolean> => {
        try {
          this.items = await getQueue()
          this.loaded = true
          return true
        } catch {
          return false
        }
      })().finally(() => {
        inflightLoad = null
      })
      return inflightLoad
    },
    async ensureLoaded(): Promise<boolean> {
      return this.loaded ? true : this.load()
    },
    /**
     * Mirror to the server, reverting to `prev` if the write fails.
     *
     * Before this, a rejected PUT left the optimistic mutation in place: the app showed a queued
     * episode the server never received, and it was gone at the next launch. That is the store
     * telling the user something untrue, and it also violated this directory's own convention —
     * "the store flips local state, fires the request, and reverts on rejection — deliberately
     * never throwing" (stores/README.md §2). It threw, too, into `void store.toggle()` call sites.
     *
     * Returns whether the change survived, because the user can perceive this one.
     */
    async _persist(prev: string[]): Promise<boolean> {
      try {
        await putQueue(this.items)
        return true
      } catch {
        this.items = prev
        return false
      }
    },
    /** Append to the end if not already queued. */
    async add(slug: string): Promise<boolean> {
      // Mutations operate on the LOADED queue: without this, an add() that runs before the
      // initial load() finishes gets overwritten when that load resolves (dropped add).
      // Without a loaded queue we do not know the baseline, and _persist sends the WHOLE list —
      // writing from an empty one would delete the user's queue on the server.
      if (!(await this.ensureLoaded())) return false
      if (this.items.includes(slug)) return true
      const prev = [...this.items]
      this.items.push(slug)
      return this._persist(prev)
    },
    /** Insert right after `afterSlug` (or at the front if it's not in the queue). */
    async playNext(slug: string, afterSlug: string | null): Promise<boolean> {
      if (!(await this.ensureLoaded())) return false
      const prev = [...this.items]
      this.items = this.items.filter((s) => s !== slug)
      const idx = afterSlug ? this.items.indexOf(afterSlug) : -1
      this.items.splice(idx + 1, 0, slug)
      return this._persist(prev)
    },
    async remove(slug: string): Promise<boolean> {
      if (!(await this.ensureLoaded())) return false
      const next = this.items.filter((s) => s !== slug)
      if (next.length === this.items.length) return true
      const prev = [...this.items]
      this.items = next
      return this._persist(prev)
    },
    async toggle(slug: string): Promise<boolean> {
      if (!(await this.ensureLoaded())) return false
      return this.items.includes(slug) ? this.remove(slug) : this.add(slug)
    },
    /** Move a slug one step up (-1) or down (+1). */
    async move(slug: string, delta: -1 | 1): Promise<boolean> {
      if (!(await this.ensureLoaded())) return false
      const i = this.items.indexOf(slug)
      const j = i + delta
      if (i < 0 || j < 0 || j >= this.items.length) return false
      const prev = [...this.items]
      ;[this.items[i], this.items[j]] = [this.items[j], this.items[i]]
      return this._persist(prev)
    },
    /** The slug after `slug` (auto-advance target), or null at the end / not queued. */
    nextAfter(slug: string): string | null {
      const i = this.items.indexOf(slug)
      return i >= 0 && i < this.items.length - 1 ? this.items[i + 1] : null
    },
    prevBefore(slug: string): string | null {
      const i = this.items.indexOf(slug)
      return i > 0 ? this.items[i - 1] : null
    },
  },
})
