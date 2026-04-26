// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const storage = new Map<string, string>()

vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => storage.set(k, v),
  removeItem: (k: string) => storage.delete(k),
  clear: () => storage.clear(),
})

describe('useGraphLensesStore (RFC-080)', () => {
  beforeEach(() => {
    storage.clear()
    setActivePinia(createPinia())
  })

  it('defaults match RFC rollout (V1 + V5 off)', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(false)
    expect(s.nodeSizeByDegree).toBe(false)
  })

  it('persists toggles to localStorage as a single JSON blob', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    s.setAggregatedEdges(true)
    s.setNodeSizeByDegree(true)
    await nextTick()
    const raw = storage.get('ps_graph_lenses')
    expect(raw).toBeTruthy()
    const parsed = JSON.parse(raw!) as Record<string, boolean>
    expect(parsed.aggregatedEdges).toBe(true)
    expect(parsed.nodeSizeByDegree).toBe(true)
  })

  it('rehydrates from localStorage on store creation', async () => {
    storage.set(
      'ps_graph_lenses',
      JSON.stringify({ aggregatedEdges: true, nodeSizeByDegree: false }),
    )
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(true)
    expect(s.nodeSizeByDegree).toBe(false)
  })

  it('falls back to defaults when localStorage payload is malformed', async () => {
    storage.set('ps_graph_lenses', 'not-json')
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(false)
    expect(s.nodeSizeByDegree).toBe(false)
  })

  it('falls back to default for missing keys in a partial blob', async () => {
    // Forward-compat: future flag added; missing keys must not crash.
    storage.set('ps_graph_lenses', JSON.stringify({ aggregatedEdges: true }))
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(true)
    expect(s.nodeSizeByDegree).toBe(false)
  })

  it('resetToDefaults clears all flags', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    s.setAggregatedEdges(true)
    s.setNodeSizeByDegree(true)
    s.resetToDefaults()
    expect(s.aggregatedEdges).toBe(false)
    expect(s.nodeSizeByDegree).toBe(false)
  })

  it('exposes a flags computed that reads both flags atomically', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    s.setAggregatedEdges(true)
    expect(s.flags).toEqual({ aggregatedEdges: true, nodeSizeByDegree: false })
  })
})
