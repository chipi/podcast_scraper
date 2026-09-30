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

/**
 * Reads the user made while a load() was in flight. That load's answer was computed BEFORE them,
 * so applying it as-is un-reads what the user just read.
 *
 * MEASURED (operator 2026-09-30, on device): opening the bell reloads the inbox, and the server runs
 * a corpus-wide new-episode sweep before it answers, so the GET can take seconds. Tapping "Mark all
 * read" in that window cleared the dots, the GET then landed with every item still unread, and the
 * POST's reply zeroed the count — so the badge said "nothing unread" over a list of unread dots.
 */
let loadSeq = 0
let readDuringLoad = new Set<string>()
let allReadDuringLoad = false

export const useNotificationsStore = defineStore('notifications', {
  state: (): State => ({ items: [], unread: 0, loaded: false }),

  actions: {
    /**
     * Refresh the inbox. Never throws: signed out is a 401 the API turns into an empty inbox, and
     * any other error leaves nothing standing — a badge is a claim, so an unknown count shows none.
     */
    async load(): Promise<void> {
      const seq = ++loadSeq
      readDuringLoad = new Set()
      allReadDuringLoad = false
      try {
        const resp = await getNotifications()
        if (seq !== loadSeq) return // a newer load owns the list now
        const items = resp.items.map((n) =>
          allReadDuringLoad || readDuringLoad.has(n.id) ? { ...n, read: true } : n,
        )
        this.items = items
        this.unread = allReadDuringLoad || readDuringLoad.size
          ? items.filter((n) => !n.read).length
          : resp.unread
      } catch {
        if (seq !== loadSeq) return // a late failure must not wipe a newer load's list
        this.items = []
        this.unread = 0
      } finally {
        if (seq === loadSeq) this.loaded = true
      }
    },

    /** Mark one read — update locally first for an instant badge, then persist. */
    async markRead(id: string): Promise<void> {
      readDuringLoad.add(id)
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

    /** Mark everything read — optimistic, then reconcile to the server's count (a concurrent
     *  sweep on GET /notifications could have added one between the optimistic zero and the write). */
    async markAllRead(): Promise<void> {
      allReadDuringLoad = true
      this.items.forEach((n) => (n.read = true))
      this.unread = 0
      try {
        const resp = await markAllNotificationsRead()
        this.unread = resp.unread
      } catch {
        // Optimistic zero stands; next load() reconciles.
      }
    },

    /** Drop everything on sign-out — one user's inbox must never show to the next. */
    reset(): void {
      loadSeq++ // a load still in flight belongs to the previous user; never let it land
      this.items = []
      this.unread = 0
      this.loaded = false
    },
  },
})
