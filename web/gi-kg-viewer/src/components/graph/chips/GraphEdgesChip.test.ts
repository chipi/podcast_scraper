// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useGraphFilterStore } from '../../../stores/graphFilters'
import type { GraphFilterState } from '../../../types/artifact'
import GraphEdgesChip from './GraphEdgesChip.vue'

const CHIP = '[data-testid="graph-chip-edges"]'
const POPOVER = '[data-testid="graph-popover-edges"]'

/** Build a filter state with the given edge-type → enabled map. */
function stateWithEdges(
  allowedEdgeTypes: Record<string, boolean>,
): GraphFilterState {
  return {
    allowedTypes: {},
    allowedEdgeTypes,
    hideUngroundedInsights: false,
    showGiLayer: true,
    showKgLayer: true,
    graphFeedFilterId: null,
  }
}

describe('GraphEdgesChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  /**
   * Seed the store state (the component reads ``gf.state``, normally
   * populated by the displayArtifact watcher) then mount the chip.
   */
  const mountChip = (allowedEdgeTypes: Record<string, boolean> | null) => {
    const gf = useGraphFilterStore()
    gf.state = allowedEdgeTypes == null ? null : stateWithEdges(allowedEdgeTypes)
    const w = mount(GraphEdgesChip, { attachTo: document.body })
    return { w, gf }
  }

  it('renders nothing when there are no edge types', () => {
    const { w } = mountChip(null)
    expect(w.find(CHIP).exists()).toBe(false)
  })

  it('renders nothing when allowedEdgeTypes is empty', () => {
    const { w } = mountChip({})
    expect(w.find(CHIP).exists()).toBe(false)
  })

  it('renders the inactive label when all edge types are enabled', () => {
    const { w } = mountChip({ mentions: true, about: true })
    expect(w.get(CHIP).text()).toContain('Edges ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
  })

  it('renders the active "K of N" label when some edges are disabled', () => {
    const { w } = mountChip({ mentions: true, about: false, cites: true })
    expect(w.get(CHIP).text()).toContain('Edges: 2 of 3 ▾')
  })

  it('lists edge keys sorted alphabetically with checkbox state', () => {
    const { w } = mountChip({ zeta: true, alpha: false })
    const labels = w.findAll(`${POPOVER} li span[title]`)
    expect(labels.map((l) => l.text())).toEqual(['alpha', 'zeta'])
    const boxes = w.findAll<HTMLInputElement>(`${POPOVER} input[type="checkbox"]`)
    // alpha (sorted first) is unchecked, zeta is checked.
    expect(boxes[0].element.checked).toBe(false)
    expect(boxes[1].element.checked).toBe(true)
  })

  it('keeps the popover hidden until the chip is clicked, and toggles aria-expanded', async () => {
    const { w } = mountChip({ mentions: true, about: true })
    expect(w.get(POPOVER).isVisible()).toBe(false)
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
    // Clicking again closes it.
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(false)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
  })

  it('toggling an edge checkbox updates the store + the chip label', async () => {
    const { w, gf } = mountChip({ mentions: true, about: true })
    await w.get(CHIP).trigger('click')
    const boxes = w.findAll(`${POPOVER} input[type="checkbox"]`)
    // Sorted keys: ['about', 'mentions']; toggle the first off.
    await boxes[0].trigger('change')
    expect(gf.state!.allowedEdgeTypes.about).toBe(false)
    expect(w.get(CHIP).text()).toContain('Edges: 1 of 2 ▾')
  })

  it('"none" disables every edge type, "all" re-enables them', async () => {
    const { w, gf } = mountChip({ mentions: true, about: true })
    await w.get(CHIP).trigger('click')
    const [allBtn, noneBtn] = w.get(POPOVER).findAll('button.underline')

    await noneBtn.trigger('click')
    expect(Object.values(gf.state!.allowedEdgeTypes)).toEqual([false, false])
    expect(w.get(CHIP).text()).toContain('Edges: 0 of 2 ▾')

    await allBtn.trigger('click')
    expect(Object.values(gf.state!.allowedEdgeTypes)).toEqual([true, true])
    expect(w.get(CHIP).text()).toContain('Edges ▾')
  })
})
