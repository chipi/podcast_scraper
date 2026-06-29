import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getMe = vi.fn()
vi.mock('../api/authApi', () => ({
  getMe: (...a: unknown[]) => getMe(...a),
  loginUrl: (grant: string | null) => `/api/app/auth/login${grant ? `?grant=${grant}` : ''}`,
  logout: vi.fn(),
}))

import { useAuthStore } from './auth'

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => getMe.mockReset())

describe('auth store', () => {
  it('refresh loads the user and sets loaded', async () => {
    getMe.mockResolvedValue({ user_id: 'u1', email: 'a@b.c', name: 'A', role: 'admin', disabled: false })
    const s = useAuthStore()
    expect(s.loaded).toBe(false)
    await s.refresh()
    expect(s.loaded).toBe(true)
    expect(s.isAuthenticated).toBe(true)
    expect(s.isAdmin).toBe(true)
    expect(s.canUseViewer).toBe(true)
  })

  it('a creator can use the viewer but is not admin', async () => {
    getMe.mockResolvedValue({ user_id: 'u2', email: 'c@b.c', name: 'C', role: 'creator', disabled: false })
    const s = useAuthStore()
    await s.refresh()
    expect(s.canUseViewer).toBe(true)
    expect(s.isAdmin).toBe(false)
  })

  it('a listener is authenticated but cannot use the viewer', async () => {
    getMe.mockResolvedValue({ user_id: 'u3', email: 'l@b.c', name: 'L', role: 'listener', disabled: false })
    const s = useAuthStore()
    await s.refresh()
    expect(s.isAuthenticated).toBe(true)
    expect(s.canUseViewer).toBe(false)
  })

  it('an anonymous visitor (null /me) is unauthenticated', async () => {
    getMe.mockResolvedValue(null)
    const s = useAuthStore()
    await s.refresh()
    expect(s.isAuthenticated).toBe(false)
    expect(s.canUseViewer).toBe(false)
    expect(s.role).toBeNull()
  })

  it('ensureLoaded only fetches once', async () => {
    getMe.mockResolvedValue(null)
    const s = useAuthStore()
    await s.ensureLoaded()
    await s.ensureLoaded()
    expect(getMe).toHaveBeenCalledTimes(1)
  })
})
