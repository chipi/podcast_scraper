/**
 * The in-app notification inbox — what backs the header bell (wave-I).
 *
 * ## Why a store
 *
 * The bell renders the unread badge and the dropdown reads the same list; a store keeps the count
 * and the items in one owner so the badge can't disagree with the panel. The inbox is the `in_app`
 * delivery channel — distinct from OS push (which reaches you while the app is closed).
 *
 * ## When it refreshes — no polling
 *
 * Like the resurfacing badge, the count is not time-sensitive to the second, so there is no poll.
 * It loads on sign-in and whenever the user opens the bell (so the panel never shows a stale list),
 * and updates locally on mark-read so the badge responds instantly without a round-trip.
 */

import { defineStore } from 'pinia'

import {
  getNotifications,
  markAllNotificationsRead,
  markNotificationRead,
} from '../services/api'
import type { NotificationItem } from '../services/types'

interface State {
  items: NotificationItem[]
  unread: number
  loaded: boolean
}

export const useNotificationsStore = defineStore('notifications', {
  state: (): State => ({ items: [], unread: 0, loaded: false }),

  actions: {
    /**
     * Refresh the inbox. Never throws: signed out is a 401 the API turns into an empty inbox, and
     * any other error leaves nothing standing — a badge is a claim, so an unknown count shows none.
     */
    async load(): Promise<void> {
      try {
        const resp = await getNotifications()
        this.items = resp.items
        this.unread = resp.unread
      } catch {
        this.items = []
        this.unread = 0
      } finally {
        this.loaded = true
      }
    },

    /** Mark one read — update locally first for an instant badge, then persist. */
    async markRead(id: string): Promise<void> {
      const item = this.items.find((n) => n.id === id)
      if (item && !item.read) {
        item.read = true
        this.unread = Math.max(0, this.unread - 1)
      }
      try {
        const resp = await markNotificationRead(id)
        this.unread = resp.unread
      } catch {
        // The optimistic local state stands; the next load() reconciles.
      }
    },

    /** Mark everything read. */
    async markAllRead(): Promise<void> {
      this.items.forEach((n) => (n.read = true))
      this.unread = 0
      try {
        await markAllNotificationsRead()
      } catch {
        // Optimistic; next load() reconciles.
      }
    },

    /** Drop everything on sign-out — one user's inbox must never show to the next. */
    reset(): void {
      this.items = []
      this.unread = 0
      this.loaded = false
    },
  },
})
