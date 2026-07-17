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
  const { useGraphTopDownStore } = await import('./graphTopDown')
  return useGraphTopDownStore()
}

describe('useGraphTopDownStore (graph-v3 tier 8-2)', () => {
  beforeEach(() => {
    storage.clear()
    setActivePinia(createPinia())
  })

  it('starts empty (hasExpansions=false, isExpanded=false everywhere)', async () => {
    const s = await useStore()
    expect(s.expandedSuperThemeIds.size).toBe(0)
    expect(s.hasExpansions).toBe(false)
    expect(s.isExpanded('sth:a')).toBe(false)
  })

  it('hydrates from localStorage when the JSON array is set', async () => {
    storage.set('ps_graph_topdown_expanded_supers', JSON.stringify(['sth:a', 'sth:b']))
    const s = await useStore()
    expect(s.expandedSuperThemeIds.size).toBe(2)
    expect(s.isExpanded('sth:a')).toBe(true)
    expect(s.hasExpansions).toBe(true)
  })

  it('ignores garbage localStorage values and falls back to empty', async () => {
    storage.set('ps_graph_topdown_expanded_supers', 'nonsense{')
    const s = await useStore()
    expect(s.expandedSuperThemeIds.size).toBe(0)
  })

  it('expandSuperTheme adds; collapseSuperTheme removes; both are idempotent', async () => {
    const s = await useStore()
    s.expandSuperTheme('sth:a')
    expect(s.isExpanded('sth:a')).toBe(true)
    /* re-expand is a no-op — should not mutate the underlying set reference. */
    const setRefBefore = s.expandedSuperThemeIds
    s.expandSuperTheme('sth:a')
    expect(s.expandedSuperThemeIds).toBe(setRefBefore)

    s.collapseSuperTheme('sth:a')
    expect(s.isExpanded('sth:a')).toBe(false)
    /* re-collapse is a no-op. */
    const setRefAfter = s.expandedSuperThemeIds
    s.collapseSuperTheme('sth:a')
    expect(s.expandedSuperThemeIds).toBe(setRefAfter)
  })

  it('toggleSuperTheme flips membership', async () => {
    const s = await useStore()
    s.toggleSuperTheme('sth:a')
    expect(s.isExpanded('sth:a')).toBe(true)
    s.toggleSuperTheme('sth:a')
    expect(s.isExpanded('sth:a')).toBe(false)
  })

  it('clearExpanded empties the set', async () => {
    const s = await useStore()
    s.expandSuperTheme('sth:a')
    s.expandSuperTheme('sth:b')
    s.clearExpanded()
    expect(s.expandedSuperThemeIds.size).toBe(0)
    expect(s.hasExpansions).toBe(false)
  })

  it('write-through persists to localStorage as a JSON array', async () => {
    const s = await useStore()
    s.expandSuperTheme('sth:a')
    await nextTick()
    const raw = storage.get('ps_graph_topdown_expanded_supers')
    expect(JSON.parse(raw!)).toEqual(['sth:a'])
    s.expandSuperTheme('sth:b')
    await nextTick()
    expect(JSON.parse(storage.get('ps_graph_topdown_expanded_supers')!).sort()).toEqual(['sth:a', 'sth:b'])
  })

  it('write-through removes the localStorage key when the set becomes empty', async () => {
    const s = await useStore()
    s.expandSuperTheme('sth:a')
    await nextTick()
    expect(storage.has('ps_graph_topdown_expanded_supers')).toBe(true)
    s.clearExpanded()
    await nextTick()
    expect(storage.has('ps_graph_topdown_expanded_supers')).toBe(false)
  })
})
