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

async function loadComponent() {
  const mod = await import('./GraphThemeLegend.vue')
  return mod.default
}

async function useStores() {
  const { useArtifactsStore } = await import('../../stores/artifacts')
  const { useGraphLensesStore } = await import('../../stores/graphLenses')
  const { useGraphThemeFocusStore } = await import('../../stores/graphThemeFocus')
  const artifacts = useArtifactsStore()
  const lenses = useGraphLensesStore()
  const themeFocus = useGraphThemeFocusStore()
  // Legend renders only when this lens is on + a doc is loaded.
  lenses.setThemeClusterRegions(true)
  return { artifacts, lenses, themeFocus }
}

/** graph-v3 tier 7 shape: 8 flat clusters, 2 rolled up under sth:alpha,
 *  the rest each their own super-theme. */
function tier7Doc() {
  return {
    clusters: [
      { graph_compound_parent_id: 'thc:c1', canonical_label: 'C1', member_count: 3,
        super_theme_id: 'sth:alpha', super_theme_label: 'Alpha' },
      { graph_compound_parent_id: 'thc:c2', canonical_label: 'C2', member_count: 2,
        super_theme_id: 'sth:alpha', super_theme_label: 'Alpha' },
      { graph_compound_parent_id: 'thc:c3', canonical_label: 'C3', member_count: 5,
        super_theme_id: 'sth:beta', super_theme_label: 'Beta' },
      { graph_compound_parent_id: 'thc:c4', canonical_label: 'C4', member_count: 4,
        super_theme_id: 'sth:gamma', super_theme_label: 'Gamma' },
    ],
  }
}

describe('GraphThemeLegend (graph-v3 tier 7)', () => {
  beforeEach(() => {
    storage.clear()
    setActivePinia(createPinia())
  })

  it('hides itself when the themeClusterRegions lens is off', async () => {
    const { artifacts, lenses } = await useStores()
    lenses.setThemeClusterRegions(false)
    artifacts.themeClustersDoc = tier7Doc()
    const Legend = await loadComponent()
    const w = mount(Legend)
    await nextTick()
    expect(w.find('[data-testid="graph-theme-legend"]').exists()).toBe(false)
  })

  it('renders single-child super-themes flat (no expand chevron)', async () => {
    const { artifacts } = await useStores()
    artifacts.themeClustersDoc = tier7Doc()
    const Legend = await loadComponent()
    const w = mount(Legend)
    await nextTick()
    // sth:beta has 1 child (C3) — no super header; just the flat cluster row.
    expect(w.find('[data-testid="graph-theme-legend-super-sth:beta"]').exists()).toBe(false)
    expect(w.find('[data-testid="graph-theme-legend-row-thc:c3"]').exists()).toBe(true)
  })

  it('renders a header + expand chevron for multi-child super-themes', async () => {
    const { artifacts } = await useStores()
    artifacts.themeClustersDoc = tier7Doc()
    const Legend = await loadComponent()
    const w = mount(Legend)
    await nextTick()
    // sth:alpha has 2 children (C1, C2) — header + toggle live; children
    // hidden until expanded.
    expect(w.find('[data-testid="graph-theme-legend-super-sth:alpha"]').exists()).toBe(true)
    expect(w.find('[data-testid="graph-theme-legend-super-toggle-sth:alpha"]').exists()).toBe(true)
    expect(w.find('[data-testid="graph-theme-legend-row-thc:c1"]').exists()).toBe(false)
    expect(w.find('[data-testid="graph-theme-legend-row-thc:c2"]').exists()).toBe(false)
    // Expand.
    await w.find('[data-testid="graph-theme-legend-super-toggle-sth:alpha"]').trigger('click')
    await nextTick()
    expect(w.find('[data-testid="graph-theme-legend-row-thc:c1"]').exists()).toBe(true)
    expect(w.find('[data-testid="graph-theme-legend-row-thc:c2"]').exists()).toBe(true)
  })

  it('clicking a super-theme label focuses all its children in the theme-focus store', async () => {
    const { artifacts, themeFocus } = await useStores()
    artifacts.themeClustersDoc = tier7Doc()
    const Legend = await loadComponent()
    const w = mount(Legend)
    await nextTick()
    await w.find('[data-testid="graph-theme-legend-super-focus-sth:alpha"]').trigger('click')
    expect(themeFocus.focusedThemeIds.has('thc:c1')).toBe(true)
    expect(themeFocus.focusedThemeIds.has('thc:c2')).toBe(true)
    expect(themeFocus.focusedThemeIds.size).toBe(2)
    // Re-click toggles off.
    await w.find('[data-testid="graph-theme-legend-super-focus-sth:alpha"]').trigger('click')
    expect(themeFocus.focusedThemeIds.size).toBe(0)
  })

  it('clicking a single-child (flat) row focuses just that cluster', async () => {
    const { artifacts, themeFocus } = await useStores()
    artifacts.themeClustersDoc = tier7Doc()
    const Legend = await loadComponent()
    const w = mount(Legend)
    await nextTick()
    // sth:beta is a single-child group — the flat row's focus button is the whole row.
    await w.find('[data-testid="graph-theme-legend-row-focus-thc:c3"]').trigger('click')
    expect(themeFocus.focusedThemeIds.size).toBe(1)
    expect(themeFocus.focusedThemeIds.has('thc:c3')).toBe(true)
  })

  it('filter input is hidden when there are <=6 super-themes', async () => {
    const { artifacts } = await useStores()
    artifacts.themeClustersDoc = tier7Doc() // 3 super-themes → filter hidden
    const Legend = await loadComponent()
    const w = mount(Legend)
    await nextTick()
    expect(w.find('[data-testid="graph-theme-legend-filter"]').exists()).toBe(false)
  })

  it('filter input filters rows by substring against super + child labels', async () => {
    const { artifacts, lenses } = await useStores()
    lenses.setThemeClusterRegions(true)
    // 7 super-themes — trips the filter-visible threshold (>6).
    artifacts.themeClustersDoc = {
      clusters: Array.from({ length: 7 }, (_, i) => ({
        graph_compound_parent_id: `thc:c${i}`,
        canonical_label: i === 3 ? 'quantum computing' : `cluster ${i}`,
        member_count: 1,
        super_theme_id: `sth:c${i}`,
        super_theme_label: i === 3 ? 'quantum computing' : `cluster ${i}`,
      })),
    }
    const Legend = await loadComponent()
    const w = mount(Legend)
    await nextTick()
    const filter = w.find('[data-testid="graph-theme-legend-filter"]')
    expect(filter.exists()).toBe(true)
    await filter.setValue('quantum')
    await nextTick()
    // Only the matching cluster row renders after filtering.
    expect(w.find('[data-testid="graph-theme-legend-row-thc:c3"]').exists()).toBe(true)
    expect(w.find('[data-testid="graph-theme-legend-row-thc:c0"]').exists()).toBe(false)
  })

  it('backwards-compat: clusters without super_theme_* land as single-child flat rows', async () => {
    const { artifacts } = await useStores()
    artifacts.themeClustersDoc = {
      clusters: [
        { graph_compound_parent_id: 'thc:only', canonical_label: 'Only', member_count: 3 },
      ],
    }
    const Legend = await loadComponent()
    const w = mount(Legend)
    await nextTick()
    // No super_theme_id → each cluster is its own group → renders flat.
    expect(w.find('[data-testid="graph-theme-legend-row-thc:only"]').exists()).toBe(true)
    expect(w.find('[data-testid="graph-theme-legend-super-sth:only"]').exists()).toBe(false)
  })
})
