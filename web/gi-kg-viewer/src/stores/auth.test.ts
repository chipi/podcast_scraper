import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getAuthStatus = vi.fn()
vi.mock('../api/authApi', () => ({
  getAuthStatus: (...a: unknown[]) => getAuthStatus(...a),
  loginUrl: (grant: string | null) => `/api/app/auth/login${grant ? `?grant=${grant}` : ''}`,
  logout: vi.fn(),
}))

import { useAuthStore } from './auth'

const u = (role: string) => ({ user_id: 'u', email: `${role}@x`, name: role, role, disabled: false })

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => getAuthStatus.mockReset())

describe('auth store', () => {
  it('refresh loads the user + enabled, sets loaded', async () => {
    getAuthStatus.mockResolvedValue({ enabled: true, user: u('admin') })
    const s = useAuthStore()
    expect(s.loaded).toBe(false)
    await s.refresh()
    expect(s.loaded).toBe(true)
    expect(s.enabled).toBe(true)
    expect(s.isAuthenticated).toBe(true)
    expect(s.isAdmin).toBe(true)
    expect(s.canUseViewer).toBe(true)
    expect(s.gated).toBe(false)
  })

  it('a creator can use the viewer but is not admin', async () => {
    getAuthStatus.mockResolvedValue({ enabled: true, user: u('creator') })
    const s = useAuthStore()
    await s.refresh()
    expect(s.canUseViewer).toBe(true)
    expect(s.isAdmin).toBe(false)
    expect(s.gated).toBe(false)
  })

  it('a signed-in listener is gated (auth enabled, no viewer access)', async () => {
    getAuthStatus.mockResolvedValue({ enabled: true, user: u('listener') })
    const s = useAuthStore()
    await s.refresh()
    expect(s.isAuthenticated).toBe(true)
    expect(s.canUseViewer).toBe(false)
    expect(s.gated).toBe(true)
  })

  it('an anonymous visitor with auth enabled is gated', async () => {
    getAuthStatus.mockResolvedValue({ enabled: true, user: null })
    const s = useAuthStore()
    await s.refresh()
    expect(s.isAuthenticated).toBe(false)
    expect(s.gated).toBe(true)
    expect(s.role).toBeNull()
  })

  it('when auth is NOT enabled the viewer is open (never gated)', async () => {
    getAuthStatus.mockResolvedValue({ enabled: false, user: null })
    const s = useAuthStore()
    await s.refresh()
    expect(s.enabled).toBe(false)
    expect(s.gated).toBe(false) // open — no login/no-access screen
    expect(s.isAdmin).toBe(false)
  })

  it('ensureLoaded only fetches once', async () => {
    getAuthStatus.mockResolvedValue({ enabled: false, user: null })
    const s = useAuthStore()
    await s.ensureLoaded()
    await s.ensureLoaded()
    expect(getAuthStatus).toHaveBeenCalledTimes(1)
  })
})
