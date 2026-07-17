// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  fetchUserPreferences: vi.fn(),
  patchUserPreferences: vi.fn(),
}))

vi.mock('../api/userPreferencesApi', () => ({
  fetchUserPreferences: api.fetchUserPreferences,
  patchUserPreferences: api.patchUserPreferences,
}))

async function loadStore() {
  const mod = await import('./userPreferences')
  return mod.useUserPreferencesStore
}

describe('useUserPreferencesStore (USERPREFS-1)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    api.fetchUserPreferences.mockReset()
    api.patchUserPreferences.mockReset()
  })

  it('starts empty, unhydrated, and marked available', async () => {
    const useStore = await loadStore()
    const s = useStore()
    expect(s.local).toEqual({})
    expect(s.hydrated).toBe(false)
    expect(s.available).toBe(true)
  })

  it('hydrate() populates local from a successful fetch and stays available', async () => {
    api.fetchUserPreferences.mockResolvedValueOnce({ preferences: { theme: 'dark' } })
    const useStore = await loadStore()
    const s = useStore()
    await s.hydrate()
    expect(s.hydrated).toBe(true)
    expect(s.local).toEqual({ theme: 'dark' })
    expect(s.available).toBe(true)
  })

  it('hydrate() marks available=false when the fetch returns null', async () => {
    api.fetchUserPreferences.mockResolvedValueOnce(null)
    const useStore = await loadStore()
    const s = useStore()
    await s.hydrate()
    expect(s.available).toBe(false)
    expect(s.local).toEqual({})
  })

  it('hydrate() is idempotent (in-flight and completed)', async () => {
    api.fetchUserPreferences.mockResolvedValue({ preferences: {} })
    const useStore = await loadStore()
    const s = useStore()
    await Promise.all([s.hydrate(), s.hydrate()])
    await s.hydrate()
    expect(api.fetchUserPreferences).toHaveBeenCalledTimes(1)
  })

  it('set(key, value) updates the local mirror immediately and PATCHes the server', async () => {
    api.fetchUserPreferences.mockResolvedValueOnce({ preferences: {} })
    api.patchUserPreferences.mockResolvedValue({ preferences: { theme: 'light' } })
    const useStore = await loadStore()
    const s = useStore()
    await s.hydrate()
    await s.set('theme', 'light')
    expect(s.local).toEqual({ theme: 'light' })
    expect(api.patchUserPreferences).toHaveBeenCalledWith({ theme: 'light' })
  })

  it('set(key, null) deletes locally and sends null to the server', async () => {
    api.fetchUserPreferences.mockResolvedValueOnce({ preferences: { theme: 'dark' } })
    api.patchUserPreferences.mockResolvedValueOnce({ preferences: {} })
    const useStore = await loadStore()
    const s = useStore()
    await s.hydrate()
    await s.set('theme', null)
    expect(s.local).toEqual({})
    expect(api.patchUserPreferences).toHaveBeenCalledWith({ theme: null })
  })

  it('set() marks available=false permanently when the PATCH returns null', async () => {
    api.fetchUserPreferences.mockResolvedValueOnce({ preferences: {} })
    api.patchUserPreferences.mockResolvedValueOnce(null)
    const useStore = await loadStore()
    const s = useStore()
    await s.hydrate()
    await s.set('theme', 'dark')
    expect(s.available).toBe(false)
    // Subsequent sets don't re-attempt the network — just local update.
    api.patchUserPreferences.mockClear()
    await s.set('theme', 'light')
    expect(api.patchUserPreferences).not.toHaveBeenCalled()
    expect(s.local).toEqual({ theme: 'light' })
  })

  it('setMany() ships one PATCH with normalised nulls for undefined values', async () => {
    api.fetchUserPreferences.mockResolvedValueOnce({ preferences: {} })
    api.patchUserPreferences.mockResolvedValueOnce({ preferences: { a: 1, b: 2 } })
    const useStore = await loadStore()
    const s = useStore()
    await s.hydrate()
    await s.setMany({ a: 1, b: 2, c: undefined })
    expect(s.local).toEqual({ a: 1, b: 2 })
    // undefined normalises to null (delete) for the server.
    expect(api.patchUserPreferences).toHaveBeenCalledWith({ a: 1, b: 2, c: null })
  })

  it('get() returns undefined for unset keys', async () => {
    const useStore = await loadStore()
    const s = useStore()
    expect(s.get('never-set')).toBeUndefined()
  })

  it('get<T>() returns the typed value when set', async () => {
    api.fetchUserPreferences.mockResolvedValueOnce({
      preferences: { flag: true, size: 42, nested: { key: 'v' } },
    })
    const useStore = await loadStore()
    const s = useStore()
    await s.hydrate()
    expect(s.get<boolean>('flag')).toBe(true)
    expect(s.get<number>('size')).toBe(42)
    expect(s.get<{ key: string }>('nested')).toEqual({ key: 'v' })
  })
})
