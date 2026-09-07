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
 *
 * ## Paused suppresses the badge entirely
 *
 * A user who paused resurfacing has said "stop asking". Showing them a count is the app asking
 * anyway, so `dueCount` reports 0 while paused rather than the badge being hidden by each caller —
 * two callers means two chances to forget.
 */

import { defineStore } from 'pinia'
import { getResurfacing } from '../services/api'

interface State {
  /** Raw due count from the server, before the paused rule is applied. */
  due: number
  paused: boolean
  loaded: boolean
}

export const useResurfacingStore = defineStore('resurfacing', {
  state: (): State => ({ due: 0, paused: false, loaded: false }),

  getters: {
    /** What the badge shows: 0 when paused, so neither caller has to remember the rule. */
    dueCount: (s): number => (s.paused ? 0 : s.due),
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
        this.paused = resp.paused
      } catch {
        this.due = 0
        this.paused = false
      } finally {
        this.loaded = true
      }
    },

    /** Drop the count on sign-out — one user's due items must never be shown to the next. */
    reset(): void {
      this.due = 0
      this.paused = false
      this.loaded = false
    },
  },
})
