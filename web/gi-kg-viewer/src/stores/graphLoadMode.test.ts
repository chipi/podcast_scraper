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
  const { useGraphLoadModeStore } = await import('./graphLoadMode')
  return useGraphLoadModeStore()
}

describe('useGraphLoadModeStore (graph-v3 tier 8-5)', () => {
  beforeEach(() => {
    storage.clear()
    setActivePinia(createPinia())
  })

  it("defaults to 'everything' when no local storage value is set", async () => {
    const s = await useStore()
    expect(s.mode).toBe('everything')
    expect(s.isTopDown).toBe(false)
  })

  it('hydrates from localStorage when the key is set', async () => {
    storage.set('ps_graph_load_mode', 'topDown')
    const s = await useStore()
    expect(s.mode).toBe('topDown')
    expect(s.isTopDown).toBe(true)
  })

  it('ignores garbage localStorage values and falls back to default', async () => {
    storage.set('ps_graph_load_mode', 'nonsense')
    const s = await useStore()
    expect(s.mode).toBe('everything')
  })

  it('toggleMode() flips between topDown and everything', async () => {
    const s = await useStore()
    expect(s.mode).toBe('everything')
    s.toggleMode()
    expect(s.mode).toBe('topDown')
    s.toggleMode()
    expect(s.mode).toBe('everything')
  })

  it('setMode(v) persists to localStorage under the write-through key', async () => {
    const s = await useStore()
    s.setMode('topDown')
    await nextTick()
    expect(storage.get('ps_graph_load_mode')).toBe('topDown')
    s.setMode('everything')
    await nextTick()
    expect(storage.get('ps_graph_load_mode')).toBe('everything')
  })

  it('setMode(same value) is a no-op', async () => {
    const s = await useStore()
    s.setMode('everything')
    await nextTick()
    // Should not have written to storage — setMode short-circuits when the
    // requested mode already matches the current one; no watcher fires.
    expect(storage.get('ps_graph_load_mode') ?? null).toBeNull()
  })
})
