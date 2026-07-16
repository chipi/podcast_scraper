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

  it('defaults: V1 off, V5 on (graph-v3 C), V6 off, V7 on, Tier 5C/5D off', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(false)
    expect(s.nodeSizeByDegree).toBe(true)
    expect(s.themeClusterRegions).toBe(false)
    expect(s.bridgeRing).toBe(true)
    // graph-v3 Tier 5C — enricher-based decoration lenses default off.
    expect(s.velocityHalo).toBe(false)
    expect(s.personCredibility).toBe(false)
    // graph-v3 Tier 5D — enricher-based edge overlays default off.
    expect(s.consensusEdges).toBe(false)
    expect(s.coGuestEdges).toBe(false)
  })

  it('persists all four toggles to localStorage as a single JSON blob', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    s.setAggregatedEdges(true)
    s.setNodeSizeByDegree(false)
    s.setThemeClusterRegions(true)
    s.setBridgeRing(false)
    await nextTick()
    const raw = storage.get('ps_graph_lenses')
    expect(raw).toBeTruthy()
    const parsed = JSON.parse(raw!) as Record<string, boolean>
    expect(parsed.aggregatedEdges).toBe(true)
    expect(parsed.nodeSizeByDegree).toBe(false)
    expect(parsed.themeClusterRegions).toBe(true)
    expect(parsed.bridgeRing).toBe(false)
  })

  it('rehydrates all four flags from localStorage on store creation', async () => {
    storage.set(
      'ps_graph_lenses',
      JSON.stringify({
        aggregatedEdges: true,
        nodeSizeByDegree: false,
        themeClusterRegions: true,
        bridgeRing: false,
      }),
    )
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(true)
    expect(s.nodeSizeByDegree).toBe(false)
    expect(s.themeClusterRegions).toBe(true)
    expect(s.bridgeRing).toBe(false)
  })

  it('migrates legacy communityColours key into themeClusterRegions', async () => {
    // graph-v3 R — the MCL iteration used communityColours; when we
    // pivoted to theme-cluster regions the flag was renamed. Users who
    // opted into the earlier flag keep their opt-in through the rename.
    storage.set('ps_graph_lenses', JSON.stringify({ communityColours: true }))
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.themeClusterRegions).toBe(true)
  })

  it('themeClusterRegions in localStorage takes precedence over legacy communityColours', async () => {
    storage.set(
      'ps_graph_lenses',
      JSON.stringify({ communityColours: true, themeClusterRegions: false }),
    )
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.themeClusterRegions).toBe(false)
  })

  it('falls back to defaults when localStorage payload is malformed', async () => {
    storage.set('ps_graph_lenses', 'not-json')
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(false)
    expect(s.nodeSizeByDegree).toBe(true)
    expect(s.themeClusterRegions).toBe(false)
    expect(s.bridgeRing).toBe(true)
  })

  it('falls back to default for missing keys in a partial blob', async () => {
    // Forward-compat: future flag added; missing keys must not crash.
    storage.set('ps_graph_lenses', JSON.stringify({ aggregatedEdges: true }))
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(true)
    expect(s.nodeSizeByDegree).toBe(true)
    expect(s.themeClusterRegions).toBe(false)
    expect(s.bridgeRing).toBe(true)
  })

  it('resetToDefaults restores V5 + V7 on, V1 + V6 off', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    s.setAggregatedEdges(true)
    s.setNodeSizeByDegree(false)
    s.setThemeClusterRegions(true)
    s.setBridgeRing(false)
    s.resetToDefaults()
    expect(s.aggregatedEdges).toBe(false)
    expect(s.nodeSizeByDegree).toBe(true)
    expect(s.themeClusterRegions).toBe(false)
    expect(s.bridgeRing).toBe(true)
  })

  it('exposes a flags computed that reads all eight flags atomically', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    s.setAggregatedEdges(true)
    expect(s.flags).toEqual({
      aggregatedEdges: true,
      nodeSizeByDegree: true,
      themeClusterRegions: false,
      bridgeRing: true,
      velocityHalo: false,
      personCredibility: false,
      consensusEdges: false,
      coGuestEdges: false,
    })
  })
})
