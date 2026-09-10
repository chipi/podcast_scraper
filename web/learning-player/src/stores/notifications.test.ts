import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { useNotificationsStore } from './notifications'
import type { NotificationItem } from '../services/types'

function item(id: string, read = false): NotificationItem {
  return { id, type: 'new_episodes', title: `n-${id}`, read, created_at: 1000 }
}

describe('notifications store (wave-I)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.restoreAllMocks()
  })

  it('loads items + unread count', async () => {
    vi.spyOn(api, 'getNotifications').mockResolvedValue({
      items: [item('a'), item('b', true)],
      unread: 1,
    })
    const s = useNotificationsStore()
    await s.load()
    expect(s.items.map((n) => n.id)).toEqual(['a', 'b'])
    expect(s.unread).toBe(1)
  })

  it('shows an empty inbox and does not throw when the fetch fails', async () => {
    vi.spyOn(api, 'getNotifications').mockRejectedValue(new Error('offline'))
    const s = useNotificationsStore()
    await expect(s.load()).resolves.toBeUndefined()
    expect(s.items).toEqual([])
    expect(s.unread).toBe(0)
  })

  it('marks one read optimistically then reconciles the count', async () => {
    vi.spyOn(api, 'getNotifications').mockResolvedValue({
      items: [item('a'), item('b')],
      unread: 2,
    })
    const markSpy = vi.spyOn(api, 'markNotificationRead').mockResolvedValue({ unread: 1 })
    const s = useNotificationsStore()
    await s.load()
    await s.markRead('a')
    expect(s.items.find((n) => n.id === 'a')?.read).toBe(true)
    expect(s.unread).toBe(1)
    expect(markSpy).toHaveBeenCalledWith('a')
  })

  it('keeps optimistic state when the mark-read call fails', async () => {
    vi.spyOn(api, 'getNotifications').mockResolvedValue({ items: [item('a')], unread: 1 })
    vi.spyOn(api, 'markNotificationRead').mockRejectedValue(new Error('offline'))
    const s = useNotificationsStore()
    await s.load()
    await expect(s.markRead('a')).resolves.toBeUndefined()
    expect(s.items[0].read).toBe(true)
    expect(s.unread).toBe(0)
  })

  it('marks all read', async () => {
    vi.spyOn(api, 'getNotifications').mockResolvedValue({
      items: [item('a'), item('b')],
      unread: 2,
    })
    const allSpy = vi.spyOn(api, 'markAllNotificationsRead').mockResolvedValue({ unread: 0 })
    const s = useNotificationsStore()
    await s.load()
    await s.markAllRead()
    expect(s.items.every((n) => n.read)).toBe(true)
    expect(s.unread).toBe(0)
    expect(allSpy).toHaveBeenCalled()
  })

  it('resets on identity change, so one account never shows another account inbox', async () => {
    vi.spyOn(api, 'getNotifications').mockResolvedValue({ items: [item('a')], unread: 1 })
    const s = useNotificationsStore()
    await s.load()
    s.reset()
    expect(s.items).toEqual([])
    expect(s.unread).toBe(0)
    expect(s.loaded).toBe(false)
  })
})
