/**
 * Completed-episodes store (Pinia ↔ GET/PUT/DELETE /api/app/completed) — the set of episodes the
 * user has manually marked played (PL.6). Auth-gated (empty + no-op signed out). Item-level writes
 * replay through the outbox, so an offline mark/unmark lands when the network returns. Drives the
 * player's mark-as-played toggle and hides completed episodes from Continue-listening / Revisit.
 */
import { defineStore } from 'pinia'
import { getCompleted, markCompleted, unmarkCompleted } from '../services/api'
import { isArrayCache, readCached, writeCached } from '../services/contentCache'
import { identityChangedSince, identityEpoch } from '../services/identity'
import { enqueue, isPermanent } from '../services/outbox'
import type { OutboxOp } from '../services/outbox'

interface CompletedState {
  slugs: string[]
  loaded: boolean
  /** Showing a cached copy not yet revalidated — readable offline; item-level writes still queue. */
  stale: boolean
}

// One in-flight load (module singleton), so a mutation's ensureLoaded() can't race the mount load().
let inflightLoad: Promise<boolean> | null = null

export const useCompletedStore = defineStore('completed', {
  state: (): CompletedState => ({ slugs: [], loaded: false, stale: false }),
  getters: {
    has:
      (s) =>
      (slug: string): boolean =>
        s.slugs.includes(slug),
    count: (s): number => s.slugs.length,
  },
  actions: {
    /** Returns whether it is now loaded. NEVER throws (offline falls back to cache). */
    async load(): Promise<boolean> {
      if (inflightLoad) return inflightLoad
      const generation = identityEpoch()
      inflightLoad = (async (): Promise<boolean> => {
        try {
          const fresh = await getCompleted()
          if (identityChangedSince(generation)) return false
          this.slugs = fresh
          this.loaded = true
          this.stale = false
          void writeCached('completed', this.slugs)
          return true
        } catch {
          const cached = await readCached<string[]>('completed', isArrayCache)
          if (cached) {
            this.slugs = cached
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
     * Send ONE item-level intent, keeping the optimistic set on a transport failure and queuing the
     * write for replay. Both mark and unmark are idempotent (a set), so a replay is always safe.
     */
    async _sendItem(op: OutboxOp, send: () => Promise<string[]>, prev: string[]): Promise<boolean> {
      const generation = identityEpoch()
      try {
        const slugs = await send()
        if (identityChangedSince(generation)) return false
        this.slugs = slugs
        this.stale = false
        void writeCached('completed', this.slugs)
        return true
      } catch (err: unknown) {
        if (identityChangedSince(generation)) return false
        if (isPermanent(err)) {
          this.slugs = prev
          return false
        }
        enqueue(op)
        void writeCached('completed', this.slugs)
        return true
      }
    },
    async mark(slug: string): Promise<boolean> {
      // Guard the whole op, not just the send: identity can switch during the async ensureLoaded(),
      // and the optimistic write below must not land in the next user's set.
      const generation = identityEpoch()
      await this.ensureLoaded()
      if (identityChangedSince(generation)) return false
      if (this.slugs.includes(slug)) return true
      const prev = [...this.slugs]
      this.slugs = [...this.slugs, slug]
      return this._sendItem({ op: 'completed.add', slug }, () => markCompleted(slug), prev)
    },
    async unmark(slug: string): Promise<boolean> {
      const generation = identityEpoch()
      await this.ensureLoaded()
      if (identityChangedSince(generation)) return false
      if (!this.slugs.includes(slug)) return true
      const prev = [...this.slugs]
      this.slugs = this.slugs.filter((s) => s !== slug)
      return this._sendItem({ op: 'completed.remove', slug }, () => unmarkCompleted(slug), prev)
    },
    async toggle(slug: string): Promise<boolean> {
      if (!(await this.ensureLoaded())) return false
      return this.slugs.includes(slug) ? this.unmark(slug) : this.mark(slug)
    },
  },
})
