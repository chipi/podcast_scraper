import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { useResurfacingStore } from './resurfacing'
import type { ResurfacingItem } from '../services/types'

function items(n: number): ResurfacingItem[] {
  return Array.from({ length: n }, (_, i) => ({ id: `h${i}` }) as unknown as ResurfacingItem)
}

describe('resurfacing store (#1592)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.restoreAllMocks()
  })

  it('counts the due items', async () => {
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: items(3), paused: false })
    const s = useResurfacingStore()
    await s.load()
    expect(s.dueCount).toBe(3)
  })

  it('reports ZERO while paused, however many are due', async () => {
    // A user who paused resurfacing said "stop asking". A badge is the app asking anyway. The rule
    // lives in the getter rather than in each caller, because two callers means two chances to
    // forget it — and the mobile nav is the one that would have been forgotten.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: items(7), paused: true })
    const s = useResurfacingStore()
    await s.load()
    expect(s.dueCount).toBe(0)
  })

  it('shows nothing when the count cannot be fetched, and does not throw', async () => {
    // A badge is a claim. An unknown count must not render a stale or invented one, and `load` is
    // called fire-and-forget from three places, so it must never reject into an unhandled rejection.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: items(4), paused: false })
    const s = useResurfacingStore()
    await s.load()
    expect(s.dueCount).toBe(4)

    vi.spyOn(api, 'getResurfacing').mockRejectedValue(new Error('offline'))
    await expect(s.load()).resolves.toBeUndefined()
    expect(s.dueCount).toBe(0)
  })

  it('resets on identity change, so one account never shows another account count', async () => {
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: items(2), paused: false })
    const s = useResurfacingStore()
    await s.load()
    expect(s.dueCount).toBe(2)

    s.reset()
    expect(s.dueCount).toBe(0)
    expect(s.loaded).toBe(false)
  })
})
