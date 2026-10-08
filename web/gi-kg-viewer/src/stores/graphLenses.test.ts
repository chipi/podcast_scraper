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

/* USERPREFS-1 — the graphLenses store now wires into useUserPreferencesStore
   for cross-device sync (write-through PATCH to /api/app/preferences). Stub
   the API here so the flag-mutation watcher doesn't fire real network calls
   at happy-dom's default localhost:3000. */
vi.mock('../api/userPreferencesApi', () => ({
  fetchUserPreferences: vi.fn().mockResolvedValue(null),
  patchUserPreferences: vi.fn().mockResolvedValue(null),
  replaceUserPreferences: vi.fn().mockResolvedValue(null),
}))

describe('useGraphLensesStore (RFC-080)', () => {
  beforeEach(() => {
    storage.clear()
    setActivePinia(createPinia())
  })

  it('defaults: V1 off, V5 on (graph-v3 C), V7 on, Tier 5C/5D off', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(false)
    expect(s.nodeSizeByDegree).toBe(true)
    expect(s.bridgeRing).toBe(true)
    // graph-v3 Tier 5C/5D — enricher-based lenses default off.
    expect(s.personCredibility).toBe(false)
    expect(s.coGuestEdges).toBe(false)
    expect(s.personCommunities).toBe(false)
  })

  it('persists the toggles to localStorage as a single JSON blob', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    s.setAggregatedEdges(true)
    s.setNodeSizeByDegree(false)
    s.setPersonCredibility(true)
    s.setBridgeRing(false)
    await nextTick()
    const raw = storage.get('ps_graph_lenses')
    expect(raw).toBeTruthy()
    const parsed = JSON.parse(raw!) as Record<string, boolean>
    expect(parsed.aggregatedEdges).toBe(true)
    expect(parsed.nodeSizeByDegree).toBe(false)
    expect(parsed.personCredibility).toBe(true)
    expect(parsed.bridgeRing).toBe(false)
  })

  it('rehydrates the flags from localStorage on store creation', async () => {
    storage.set(
      'ps_graph_lenses',
      JSON.stringify({
        aggregatedEdges: true,
        nodeSizeByDegree: false,
        personCredibility: true,
        bridgeRing: false,
      }),
    )
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(true)
    expect(s.nodeSizeByDegree).toBe(false)
    expect(s.personCredibility).toBe(true)
    expect(s.bridgeRing).toBe(false)
  })

  it('ignores stored flags for lenses this viewer does not have', async () => {
    // ADR-158: theme regions, velocity halo and consensus edges are private lenses. A blob saved by
    // a viewer that had them must not resurrect them here.
    storage.set(
      'ps_graph_lenses',
      JSON.stringify({ storylineRegions: true, velocityHalo: true, consensusEdges: true }),
    )
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(Object.keys(s.flags)).not.toContain('storylineRegions')
    expect(Object.keys(s.flags)).not.toContain('velocityHalo')
    expect(Object.keys(s.flags)).not.toContain('consensusEdges')
  })

  it('falls back to defaults when localStorage payload is malformed', async () => {
    storage.set('ps_graph_lenses', 'not-json')
    setActivePinia(createPinia())
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    expect(s.aggregatedEdges).toBe(false)
    expect(s.nodeSizeByDegree).toBe(true)
    expect(s.personCredibility).toBe(false)
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
    expect(s.personCredibility).toBe(false)
    expect(s.bridgeRing).toBe(true)
  })

  it('resetToDefaults restores V5 + V7 on, V1 + Tier 5C off', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    s.setAggregatedEdges(true)
    s.setNodeSizeByDegree(false)
    s.setPersonCredibility(true)
    s.setBridgeRing(false)
    s.resetToDefaults()
    expect(s.aggregatedEdges).toBe(false)
    expect(s.nodeSizeByDegree).toBe(true)
    expect(s.personCredibility).toBe(false)
    expect(s.bridgeRing).toBe(true)
  })

  it('exposes a flags computed that reads every flag atomically', async () => {
    const { useGraphLensesStore } = await import('./graphLenses')
    const s = useGraphLensesStore()
    s.setAggregatedEdges(true)
    expect(s.flags).toEqual({
      aggregatedEdges: true,
      nodeSizeByDegree: true,
      bridgeRing: true,
      personCredibility: false,
      coGuestEdges: false,
      personCommunities: false,
    })
  })
})
