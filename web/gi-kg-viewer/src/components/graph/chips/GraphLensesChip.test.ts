// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
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

// Stub the popover composable — we only care about lens toggling here, not the
// open/close mechanics of the popover positioning.
vi.mock('../../../composables/useFilterChipPopover', () => ({
  useFilterChipPopover: () => ({
    open: { value: true },
    toggle: vi.fn(),
    close: vi.fn(),
  }),
}))

async function loadChip() {
  const mod = await import('./GraphLensesChip.vue')
  return mod.default
}

async function useStores() {
  const { useGraphLensesStore } = await import('../../../stores/graphLenses')
  const { useArtifactsStore } = await import('../../../stores/artifacts')
  return { lenses: useGraphLensesStore(), artifacts: useArtifactsStore() }
}

describe('GraphLensesChip (graph-v3 Q + Tier 5A-4)', () => {
  beforeEach(() => {
    storage.clear()
    setActivePinia(createPinia())
  })

  it('renders 3 rows when themeClustersDoc is null (enricher-gated: theme regions hidden)', async () => {
    const { artifacts } = await useStores()
    artifacts.themeClustersDoc = null
    const Chip = await loadChip()
    const w = mount(Chip)
    await nextTick()
    expect(w.find('[data-testid="lens-node-size-by-degree"]').exists()).toBe(true)
    expect(w.find('[data-testid="lens-bridge-ring"]').exists()).toBe(true)
    expect(w.find('[data-testid="lens-aggregated-edges"]').exists()).toBe(true)
    // Theme regions row hidden when the artifact isn't loaded.
    expect(w.find('[data-testid="lens-theme-cluster-regions"]').exists()).toBe(false)
  })

  it('renders 4 rows including theme regions when the artifact is present', async () => {
    const { artifacts } = await useStores()
    artifacts.themeClustersDoc = {
      clusters: [{ graph_compound_parent_id: 'thc:x', canonical_label: 'X' }],
    }
    const Chip = await loadChip()
    const w = mount(Chip)
    await nextTick()
    expect(w.find('[data-testid="lens-theme-cluster-regions"]').exists()).toBe(true)
  })

  it('toggling a checkbox writes to the lens store', async () => {
    const { lenses, artifacts } = await useStores()
    artifacts.themeClustersDoc = { clusters: [{ graph_compound_parent_id: 'thc:x' }] }
    // Defaults: themeClusterRegions=false, bridgeRing=true.
    expect(lenses.themeClusterRegions).toBe(false)
    const Chip = await loadChip()
    const w = mount(Chip)
    await nextTick()
    const themeCheckbox = w.find('[data-testid="lens-theme-cluster-regions"]')
    expect(themeCheckbox.exists()).toBe(true)
    await themeCheckbox.setValue(true)
    expect(lenses.themeClusterRegions).toBe(true)
  })

  it('reset link calls resetToDefaults', async () => {
    const { lenses, artifacts } = await useStores()
    artifacts.themeClustersDoc = null
    lenses.setAggregatedEdges(true)
    lenses.setNodeSizeByDegree(false)
    const Chip = await loadChip()
    const w = mount(Chip)
    await nextTick()
    await w.find('[data-testid="lens-reset"]').trigger('click')
    expect(lenses.aggregatedEdges).toBe(false)
    expect(lenses.nodeSizeByDegree).toBe(true) // default-on
    expect(lenses.bridgeRing).toBe(true) // default-on
  })
})
