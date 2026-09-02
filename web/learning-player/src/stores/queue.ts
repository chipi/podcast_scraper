/**
 * Play-queue store (Pinia ↔ GET/PUT /api/app/queue, RFC-099 §4). Ordered episode slugs;
 * auth-gated (empty + no-op when signed out). Every mutation mirrors to the server.
 * Auto-advance is driven by the player calling `nextAfter(slug)` on `ended`.
 */

import { defineStore } from 'pinia'
import { ApiError, addQueueItem, getQueue, putQueue, removeQueueItem } from '../services/api'
import { readCached, writeCached } from '../services/contentCache'
import { identityChangedSince, identityEpoch } from '../services/identity'
import { enqueue } from '../services/outbox'
import type { OutboxOp } from '../services/outbox'

interface QueueState {
  items: string[]
  loaded: boolean
  /** Showing a cached copy that was never revalidated — readable, but NOT safe to write from. */
  stale: boolean
}

// Single in-flight load promise (module-scoped: the store is an app singleton). Guards
// against a SECOND GET /queue racing the first — e.g. the initial mount load() vs a mutation's
// ensureLoaded(). Without it a late-resolving load() overwrites `items` with stale server data
// and silently drops an optimistic add ("queue empty" after add; RFC-099 §4).
let inflightLoad: Promise<boolean> | null = null

export const useQueueStore = defineStore('queue', {
  state: (): QueueState => ({ items: [], loaded: false, stale: false }),
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
      const generation = identityEpoch()
      inflightLoad = (async (): Promise<boolean> => {
        try {
          const fresh = await getQueue()
          // The account changed while this was in flight — this result belongs to nobody now,
          // and writing it would put A's queue in B's store and B's cache file.
          if (identityChangedSince(generation)) return false
          this.items = fresh
          this.loaded = true
          this.stale = false
          void writeCached('queue', this.items)
          return true
        } catch {
          // Fall back to the cached copy so the queue is READABLE offline (#1909). It is not
          // WRITABLE: `stale` keeps the mutations refusing, because _persist sends the whole
          // list and writing from a stale baseline would delete the server's queue.
          const cached = await readCached<string[]>('queue')
          if (cached) {
            this.items = cached
            this.loaded = true
            this.stale = true
          }
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
     * Mirror the WHOLE list to the server, reverting to `prev` if the write fails.
     *
     * Only `move` uses this now. Reordering has no item-level form — "swap these two" replayed
     * against a list someone else has since changed means something different from what the user
     * did — so it keeps the whole-list PUT, and with it the stale refusal. The arrows disable
     * while the list is stale so that refusal is visible rather than silent.
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
      // Never write from a cached baseline: the PUT replaces the whole list server-side, so a
      // stale one would delete whatever the server actually has. The caller gets `false` and is
      // expected to tell the user — silently doing nothing is what made this look like a dead
      // button after any offline boot (#1909).
      if (this.stale) {
        this.items = prev
        return false
      }
      try {
        await putQueue(this.items)
        return true
      } catch {
        this.items = prev
        return false
      }
    },
    /**
     * Send ONE item-level intent, keeping the optimistic list on a transport failure and queuing
     * the write for replay. Returns whether the user's intent is safely recorded — which offline
     * means "in the outbox", not "on the server".
     *
     * This is what item-level operations bought (#1925): add/remove/play-next are idempotent, so
     * replaying them cannot clobber an edit made on another device in between, and they no longer
     * need a fresh baseline. Only `move` still does.
     */
    async _sendItem(op: OutboxOp, send: () => Promise<string[]>, prev: string[]): Promise<boolean> {
      try {
        const items = await send()
        // The server's answer is the truth, and it may differ from our optimistic guess (another
        // device reordered, the anchor moved).
        this.items = items
        this.stale = false
        void writeCached('queue', this.items)
        return true
      } catch (err: unknown) {
        // A server REFUSAL is an answer — a 404 for a removed episode, a 401 for a dead session.
        // Replaying it would only fail again, so revert and report it.
        if (err instanceof ApiError) {
          this.items = prev
          return false
        }
        // The request never landed. The optimistic list STAYS — the user's tap was real and the
        // write is queued — which is the whole difference from the old whole-list write.
        enqueue(op)
        void writeCached('queue', this.items)
        return true
      }
    },
    /** Append to the end if not already queued. */
    async add(slug: string): Promise<boolean> {
      // Mutations operate on the LOADED queue: without this, an add() that runs before the
      // initial load() finishes gets overwritten when that load resolves (dropped add).
      await this.ensureLoaded()
      if (this.items.includes(slug)) return true
      const prev = [...this.items]
      this.items.push(slug)
      return this._sendItem({ op: 'queue.add', slug }, () => addQueueItem(slug), prev)
    },
    /** Insert right after `afterSlug` (or at the front if it's not in the queue). */
    async playNext(slug: string, afterSlug: string | null): Promise<boolean> {
      await this.ensureLoaded()
      const prev = [...this.items]
      this.items = this.items.filter((s) => s !== slug)
      const idx = afterSlug ? this.items.indexOf(afterSlug) : -1
      this.items.splice(idx + 1, 0, slug)
      return this._sendItem(
        { op: 'queue.add', slug, after: afterSlug },
        () => addQueueItem(slug, afterSlug),
        prev,
      )
    },
    async remove(slug: string): Promise<boolean> {
      await this.ensureLoaded()
      if (!this.items.includes(slug)) return true
      const prev = [...this.items]
      this.items = this.items.filter((s) => s !== slug)
      return this._sendItem({ op: 'queue.remove', slug }, () => removeQueueItem(slug), prev)
    },
    async toggle(slug: string): Promise<boolean> {
      if (!(await this.ensureLoaded())) return false
      return this.items.includes(slug) ? this.remove(slug) : this.add(slug)
    },
    /** Move a slug one step up (-1) or down (+1). Requires a live queue — see `_persist`. */
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
