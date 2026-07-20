// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

const storage = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => storage.set(k, v),
  removeItem: (k: string) => storage.delete(k),
  clear: () => storage.clear(),
})

async function useStore() {
  const { useLibraryViewModeStore } = await import('./libraryViewMode')
  return useLibraryViewModeStore()
}

describe('useLibraryViewModeStore (USERPREFS-1 adoption)', () => {
  beforeEach(() => {
    storage.clear()
    setActivePinia(createPinia())
  })

  it("defaults to 'episodes' when no local storage value is set", async () => {
    const s = await useStore()
    expect(s.mode).toBe('episodes')
  })

  it("hydrates 'shows' from localStorage", async () => {
    storage.set('gikg.library.mode', 'shows')
    const s = await useStore()
    expect(s.mode).toBe('shows')
  })

  it('ignores garbage localStorage values and falls back to default', async () => {
    storage.set('gikg.library.mode', 'nonsense')
    const s = await useStore()
    expect(s.mode).toBe('episodes')
  })

  it('setMode(v) persists to localStorage under the write-through key', async () => {
    const s = await useStore()
    s.setMode('shows')
    await nextTick()
    expect(storage.get('gikg.library.mode')).toBe('shows')
    s.setMode('episodes')
    await nextTick()
    expect(storage.get('gikg.library.mode')).toBe('episodes')
  })

  it('setMode(same value) is a no-op — no write-through fires', async () => {
    const s = await useStore()
    s.setMode('episodes') // Same as default
    await nextTick()
    expect(storage.get('gikg.library.mode') ?? null).toBeNull()
  })
})
