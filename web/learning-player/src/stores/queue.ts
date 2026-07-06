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
    async load(): Promise<void> {
      this.items = await getQueue()
      this.loaded = true
    },
    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.load()
    },
    async _persist(): Promise<void> {
      await putQueue(this.items)
    },
    /** Append to the end if not already queued. */
    async add(slug: string): Promise<void> {
      if (!this.items.includes(slug)) {
        this.items.push(slug)
        await this._persist()
      }
    },
    /** Insert right after `afterSlug` (or at the front if it's not in the queue). */
    async playNext(slug: string, afterSlug: string | null): Promise<void> {
      this.items = this.items.filter((s) => s !== slug)
      const idx = afterSlug ? this.items.indexOf(afterSlug) : -1
      this.items.splice(idx + 1, 0, slug)
      await this._persist()
    },
    async remove(slug: string): Promise<void> {
      const next = this.items.filter((s) => s !== slug)
      if (next.length !== this.items.length) {
        this.items = next
        await this._persist()
      }
    },
    async toggle(slug: string): Promise<void> {
      if (this.items.includes(slug)) await this.remove(slug)
      else await this.add(slug)
    },
    /** Move a slug one step up (-1) or down (+1). */
    async move(slug: string, delta: -1 | 1): Promise<void> {
      const i = this.items.indexOf(slug)
      const j = i + delta
      if (i < 0 || j < 0 || j >= this.items.length) return
      ;[this.items[i], this.items[j]] = [this.items[j], this.items[i]]
      await this._persist()
    },
    /** The slug after `slug` (auto-advance target), or null at the end / not queued. */
    nextAfter(slug: string): string | null {
      const i = this.items.indexOf(slug)
      return i >= 0 && i < this.items.length - 1 ? this.items[i + 1] : null
    },
  },
})
