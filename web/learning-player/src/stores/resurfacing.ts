/**
 * How many highlights are due to resurface — the number the Library nav badge shows (#1592).
 *
 * ## Why a store rather than a call from the nav
 *
 * The badge renders in TWO places: `NavIconLink` on desktop (`sm:` and up) and `BottomNav` on
 * phones. They are separate components with separate markup, so without a shared owner the count
 * would be fetched twice per navigation and the two could disagree. `ResurfacingInbox` already
 * calls `getResurfacing()` directly for the full list; this store owns only the COUNT.
 *
 * ## When it refreshes — and why there is no polling
 *
 * Resurfacing is a spaced-repetition ladder measured in days. A count that is minutes stale is
 * indistinguishable from a fresh one, so polling would spend a request per interval to change
 * nothing. It loads on the three events that can actually move it:
 *
 *   - **sign-in** — there is no count for a signed-out visitor
 *   - **visiting the Library** — the inbox is one tap away, so the badge must not disagree with
 *     what the user is about to see
 *   - **after a successful capture** — the only in-app action that adds to the ladder
 *   - **visiting Home** — since 2026-09-18 `RevisitRail` renders the due ITEMS there, and waiting
 *     for a Library visit to populate them would mean the rail is blank until the user makes the
 *     trip it exists to save
 *
 * ## Paused suppresses the badge entirely
 *
 * A user who paused resurfacing has said "stop asking". Showing them a count is the app asking
 * anyway, so `dueCount` reports 0 while paused rather than the badge being hidden by each caller —
 * two callers means two chances to forget.
 */

import { defineStore } from 'pinia'
import { getResurfacing, markSurfaced } from '../services/api'
import type { ResurfacingItem } from '../services/types'

interface State {
  /** Raw due count from the server, before the paused rule is applied. */
  due: number
  /**
   * The due items themselves, for the Home rail (operator 2026-09-18).
   *
   * The response ALREADY carries them — this store was throwing everything but `.length` away —
   * so the rail costs no extra request, and it cannot disagree with the badge because both read
   * the same fetch.
   */
  items: ResurfacingItem[]
  paused: boolean
  loaded: boolean
}

export const useResurfacingStore = defineStore('resurfacing', {
  state: (): State => ({ due: 0, items: [], paused: false, loaded: false }),

  getters: {
    /** What the badge shows: 0 when paused, so neither caller has to remember the rule. */
    dueCount: (s): number => (s.paused ? 0 : s.due),

    /**
     * Up to four due captures for the Home rail, at most one PER EPISODE (operator 2026-09-18).
     *
     * `select_due` groups by episode, so the first four would often be four captures from one
     * episode — a rail meant to show the breadth of what is waiting would show one episode's
     * session instead. One per episode until four are found, then fill from what is left so a
     * user with a single active episode still sees a populated rail.
     *
     * Empty while paused, for the reason the badge is: the user said stop asking.
     */
    railItems: (s): ResurfacingItem[] => {
      if (s.paused) return []
      const seen = new Set<string>()
      const spread = s.items.filter((i) => {
        const slug = i.highlight.episode_slug
        if (seen.has(slug)) return false
        seen.add(slug)
        return true
      })
      const rest = s.items.filter((i) => !spread.includes(i))
      return [...spread, ...rest].slice(0, 4)
    },
  },

  actions: {
    /**
     * Refresh the count. Never throws and never leaves a stale number standing after a failure:
     * signed out is a 401, which the API layer turns into an empty response, and any other error
     * means we do not know the count — and a badge is a claim, so an unknown count shows nothing.
     */
    async load(): Promise<void> {
      try {
        const resp = await getResurfacing()
        this.due = resp.items.length
        this.items = resp.items
        this.paused = resp.paused
      } catch {
        this.due = 0
        this.items = []
        this.paused = false
      } finally {
        this.loaded = true
      }
    },

    /**
     * Mark one capture reviewed and drop it from the stack (operator 2026-09-18).
     *
     * Optimistic and local: the card leaves immediately and `railItems` recomputes, so the next
     * due capture — from a different episode, per that getter — takes the empty slot at once. The
     * whole point of the rail is the number of reviews answered, and a spinner between answers is
     * a reason to stop answering.
     *
     * On failure the item goes back. A card that vanished and silently did not count would leave
     * the user believing they had reviewed something they had not.
     */
    async review(id: string): Promise<void> {
      const before = this.items
      this.items = this.items.filter((i) => i.highlight.id !== id)
      this.due = Math.max(0, this.due - 1)
      try {
        await markSurfaced(id)
      } catch {
        this.items = before
        this.due = before.length
      }
    },

    /** Drop the count on sign-out — one user's due items must never be shown to the next. */
    reset(): void {
      this.due = 0
      this.items = []
      this.paused = false
      this.loaded = false
    },
  },
})
