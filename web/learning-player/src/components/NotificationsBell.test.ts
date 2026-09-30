import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { NotificationItem } from '../services/types'
import { useNotificationsStore } from '../stores/notifications'
import NotificationsBell from './NotificationsBell.vue'

function items(): NotificationItem[] {
  return [
    { id: 'n1', type: 'digest', title: 'Your Week', read: false, created_at: 1 },
    { id: 'n2', type: 'new_episodes', title: 'New episode', read: false, created_at: 2 },
  ]
}

async function mountOpen() {
  setActivePinia(createPinia())
  const router = createRouter({ history: createMemoryHistory(), routes: [{ path: '/', component: { template: '<div/>' } }] })
  const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
  const w = mount(NotificationsBell, { global: { plugins: [router, i18n] }, attachTo: document.body })
  await w.get('[data-testid="notifications-bell"]').trigger('click')
  await flushPromises()
  return w
}

/** Unread dots are the per-item `bg-accent` spans inside each notification row. */
function unreadDots(w: ReturnType<typeof mount>): number {
  return w.findAll('[data-testid="notification-item"] span.bg-accent').length
}

describe('NotificationsBell — mark all read', () => {
  afterEach(() => vi.restoreAllMocks())

  it('clears every per-item unread dot, not only the bell badge', async () => {
    vi.spyOn(api, 'getNotifications').mockResolvedValue({ items: items(), unread: 2 })
    vi.spyOn(api, 'markAllNotificationsRead').mockResolvedValue({ unread: 0 })
    const w = await mountOpen()
    expect(unreadDots(w)).toBe(2)

    await w.get('[data-testid="notifications-mark-all"]').trigger('click')
    await flushPromises()

    expect(w.find('[data-testid="notifications-badge"]').exists()).toBe(false)
    expect(unreadDots(w)).toBe(0)
    w.unmount()
  })

  it('a reload that was already in flight cannot bring the dots back', async () => {
    // Opening the panel reloads the inbox, and the server runs a corpus-wide new-episode sweep
    // before answering, so on prod that GET can take seconds. Tapping "Mark all read" in that
    // window cleared the dots, then the GET landed with its PRE-tap answer (every item unread) and
    // the POST's reply zeroed the count: badge gone, every dot back.
    const get = vi.spyOn(api, 'getNotifications').mockResolvedValue({ items: items(), unread: 2 })
    let postDone!: (v: { unread: number }) => void
    vi.spyOn(api, 'markAllNotificationsRead').mockReturnValue(
      new Promise((r) => (postDone = r)),
    )
    const w = await mountOpen()
    await w.get('[data-testid="notifications-bell"]').trigger('click') // close

    let getDone!: (v: { items: NotificationItem[]; unread: number }) => void
    get.mockReturnValue(new Promise((r) => (getDone = r)))
    await w.get('[data-testid="notifications-bell"]').trigger('click') // reopen: reload in flight
    await w.get('[data-testid="notifications-mark-all"]').trigger('click')
    await flushPromises()

    getDone({ items: items(), unread: 2 }) // the stale, pre-tap answer
    await flushPromises()
    postDone({ unread: 0 })
    await flushPromises()

    expect(w.find('[data-testid="notifications-badge"]').exists()).toBe(false)
    expect(unreadDots(w)).toBe(0)
    w.unmount()
  })

  it('a load still in flight at sign-out does not land in the next session', async () => {
    setActivePinia(createPinia())
    let getDone!: (v: { items: NotificationItem[]; unread: number }) => void
    vi.spyOn(api, 'getNotifications').mockReturnValue(new Promise((r) => (getDone = r)))
    const store = useNotificationsStore()
    const pending = store.load()
    store.reset() // sign-out while the previous user's GET is still out
    getDone({ items: items(), unread: 2 })
    await pending
    expect(store.items).toEqual([])
    expect(store.unread).toBe(0)
  })

  it('stays cleared when the panel is reopened and the inbox reloads', async () => {
    const get = vi.spyOn(api, 'getNotifications').mockResolvedValue({ items: items(), unread: 2 })
    vi.spyOn(api, 'markAllNotificationsRead').mockResolvedValue({ unread: 0 })
    const w = await mountOpen()
    await w.get('[data-testid="notifications-mark-all"]').trigger('click')
    await flushPromises()

    // The server marked every record read, so the next GET says so.
    get.mockResolvedValue({ items: items().map((n) => ({ ...n, read: true })), unread: 0 })
    await w.get('[data-testid="notifications-bell"]').trigger('click') // close
    await w.get('[data-testid="notifications-bell"]').trigger('click') // reopen → load()
    await flushPromises()

    expect(unreadDots(w)).toBe(0)
    w.unmount()
  })
})
