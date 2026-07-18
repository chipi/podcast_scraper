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

/* USERPREFS-1 — the chip probes enricher availability via
   fetchCachedCorpusEnvelope AND uses the graphLenses store which
   write-through PATCHes to /api/app/preferences. Stub both so the tests
   stay hermetic (happy-dom otherwise attempts localhost:3000 for every
   fetch). */
vi.mock('../../../composables/useEnrichmentEnvelopeCache', () => ({
  fetchCachedCorpusEnvelope: vi.fn().mockResolvedValue(null),
}))

vi.mock('../../../api/userPreferencesApi', () => ({
  fetchUserPreferences: vi.fn().mockResolvedValue(null),
  patchUserPreferences: vi.fn().mockResolvedValue(null),
  replaceUserPreferences: vi.fn().mockResolvedValue(null),
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

/* Type dance so `seedFullArtifactWithEdges` can type the argument
   without a top-level dynamic import (vitest hoists mocks; static imports
   would run before setActivePinia). */
type UseArtifactsStoreCtor = Awaited<ReturnType<typeof useStores>>['artifacts']
const useArtifactsStoreCtor = null as unknown as (
  ...args: unknown[]
) => UseArtifactsStoreCtor

describe('GraphLensesChip (graph-v3 Q + Tier 5A-4)', () => {
  beforeEach(() => {
    storage.clear()
    setActivePinia(createPinia())
  })

  function seedFullArtifactWithEdges(
    artifacts: ReturnType<typeof useArtifactsStoreCtor>,
    edgeTypes: string[],
  ): void {
    artifacts.fullArtifact = {
      name: 'stub',
      kind: 'gi',
      episodeId: 'e1',
      nodes: 0,
      edges: 0,
      nodeTypes: {},
      data: {
        nodes: [],
        edges: edgeTypes.map((t, i) => ({ id: `e${i}`, from: 'a', to: 'b', type: t })),
      },
    } as unknown as typeof artifacts.fullArtifact
  }

  it('renders 2 rows when no artifact is loaded (all enricher/data-gated rows hidden)', async () => {
    const { artifacts } = await useStores()
    artifacts.themeClustersDoc = null
    artifacts.fullArtifact = null
    const Chip = await loadChip()
    const w = mount(Chip)
    await nextTick()
    expect(w.find('[data-testid="lens-node-size-by-degree"]').exists()).toBe(true)
    expect(w.find('[data-testid="lens-bridge-ring"]').exists()).toBe(true)
    // Theme regions row hidden when the enricher artifact isn't loaded.
    expect(w.find('[data-testid="lens-theme-cluster-regions"]').exists()).toBe(false)
    // Aggregated-edges row hidden when the full artifact has no ABOUT /
    // SPOKEN_BY source edges to roll up (FU2 gate).
    expect(w.find('[data-testid="lens-aggregated-edges"]').exists()).toBe(false)
  })

  it('renders theme regions row when the artifact is present', async () => {
    const { artifacts } = await useStores()
    artifacts.themeClustersDoc = {
      clusters: [{ graph_compound_parent_id: 'thc:x', canonical_label: 'X' }],
    }
    const Chip = await loadChip()
    const w = mount(Chip)
    await nextTick()
    expect(w.find('[data-testid="lens-theme-cluster-regions"]').exists()).toBe(true)
  })

  it('renders aggregated-edges row when the artifact has ABOUT edges', async () => {
    const { artifacts } = await useStores()
    seedFullArtifactWithEdges(artifacts, ['ABOUT'])
    const Chip = await loadChip()
    const w = mount(Chip)
    await nextTick()
    expect(w.find('[data-testid="lens-aggregated-edges"]').exists()).toBe(true)
  })

  it('renders aggregated-edges row when the artifact has SPOKEN_BY edges', async () => {
    const { artifacts } = await useStores()
    seedFullArtifactWithEdges(artifacts, ['SPOKEN_BY'])
    const Chip = await loadChip()
    const w = mount(Chip)
    await nextTick()
    expect(w.find('[data-testid="lens-aggregated-edges"]').exists()).toBe(true)
  })

  it('hides aggregated-edges row when the artifact only has unrelated edge types', async () => {
    const { artifacts } = await useStores()
    seedFullArtifactWithEdges(artifacts, ['HAS_INSIGHT', 'RELATED_TO'])
    const Chip = await loadChip()
    const w = mount(Chip)
    await nextTick()
    expect(w.find('[data-testid="lens-aggregated-edges"]').exists()).toBe(false)
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
