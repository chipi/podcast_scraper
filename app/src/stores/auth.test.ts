import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { useAuthStore } from './auth'

beforeEach(() => {
  setActivePinia(createPinia())
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('auth store', () => {
  it('refresh() populates the user and marks loaded', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue({ user_id: 'u_1', email: 'dev@localhost', name: 'Dev' })
    const auth = useAuthStore()
    expect(auth.isAuthenticated).toBe(false)
    await auth.refresh()
    expect(auth.isAuthenticated).toBe(true)
    expect(auth.user?.email).toBe('dev@localhost')
    expect(auth.loaded).toBe(true)
  })

  it('refresh() leaves user null when signed out', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue(null)
    const auth = useAuthStore()
    await auth.refresh()
    expect(auth.isAuthenticated).toBe(false)
    expect(auth.loaded).toBe(true)
  })

  it('ensureLoaded() only refreshes once', async () => {
    const spy = vi.spyOn(api, 'getMe').mockResolvedValue(null)
    const auth = useAuthStore()
    await auth.ensureLoaded()
    await auth.ensureLoaded()
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('logout() clears the user via the API', async () => {
    vi.spyOn(api, 'getMe').mockResolvedValue({ user_id: 'u_1', email: 'd@l', name: 'D' })
    const logoutSpy = vi.spyOn(api, 'logout').mockResolvedValue()
    const auth = useAuthStore()
    await auth.refresh()
    await auth.logout()
    expect(logoutSpy).toHaveBeenCalledOnce()
    expect(auth.isAuthenticated).toBe(false)
  })

  it('login() redirects into the OAuth flow', () => {
    const assign = vi.fn()
    vi.stubGlobal('location', { assign } as unknown as Location)
    useAuthStore().login()
    expect(assign).toHaveBeenCalledWith(api.loginUrl())
    vi.unstubAllGlobals()
  })
})
