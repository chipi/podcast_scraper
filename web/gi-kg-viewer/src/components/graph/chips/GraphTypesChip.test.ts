// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useGraphFilterStore } from '../../../stores/graphFilters'
import type { GraphFilterState } from '../../../types/artifact'
import GraphTypesChip from './GraphTypesChip.vue'

const CHIP = '[data-testid="graph-chip-types"]'
const POPOVER = '[data-testid="graph-popover-types"]'
const RESET = '[data-testid="graph-types-reset"]'

/** Build a filter state with the given node-type → enabled map. */
function stateWithTypes(
  allowedTypes: Record<string, boolean>,
): GraphFilterState {
  return {
    allowedTypes,
    allowedEdgeTypes: {},
    hideUngroundedInsights: false,
    showGiLayer: true,
    showKgLayer: true,
    graphFeedFilterId: null,
  }
}

/**
 * Graph-spec default visibility: Quote + Speaker hidden, everything else
 * visible. A state matching this is *not* "active" (no deviation).
 */
const DEFAULT_TYPES = {
  Episode: true,
  Insight: true,
  Quote: false,
  Speaker: false,
}

describe('GraphTypesChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  const mountChip = (
    allowedTypes: Record<string, boolean> | null,
    typeHistogramCounts: Record<string, number> = {},
  ) => {
    const gf = useGraphFilterStore()
    gf.state = allowedTypes == null ? null : stateWithTypes(allowedTypes)
    const w = mount(GraphTypesChip, {
      props: { typeHistogramCounts },
      attachTo: document.body,
    })
    return { w, gf }
  }

  it('renders the inactive label when types match spec defaults', () => {
    const { w } = mountChip({ ...DEFAULT_TYPES })
    expect(w.get(CHIP).text()).toContain('Types ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
  })

  it('renders the active "K of N" label when types deviate from defaults', () => {
    // Quote on (default is off) => deviates. 3 of 4 enabled.
    const { w } = mountChip({
      Episode: true,
      Insight: true,
      Quote: true,
      Speaker: false,
    })
    expect(w.get(CHIP).text()).toContain('Types: 3 of 4 ▾')
  })

  it('renders legend labels and per-type histogram counts from props', () => {
    const { w } = mountChip(
      { Episode: true, Entity_person: true },
      { Episode: 7, Entity_person: 3 },
    )
    const text = w.get(POPOVER).text()
    expect(text).toContain('Entity (person)') // legend remap
    expect(text).toContain('(7)')
    expect(text).toContain('(3)')
  })

  it('falls back to 0 for types missing from the histogram props', () => {
    const { w } = mountChip({ Episode: true }, {})
    expect(w.get(POPOVER).text()).toContain('(0)')
  })

  it('keeps the popover hidden until the chip is clicked, and toggles aria-expanded', async () => {
    const { w } = mountChip({ ...DEFAULT_TYPES })
    expect(w.get(POPOVER).isVisible()).toBe(false)
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })

  it('lists type keys sorted alphabetically with checkbox state', () => {
    const { w } = mountChip({ Zeta: true, Alpha: false })
    const boxes = w.findAll<HTMLInputElement>(`${POPOVER} input[type="checkbox"]`)
    // Sorted: Alpha (off), Zeta (on).
    expect(boxes[0].element.checked).toBe(false)
    expect(boxes[1].element.checked).toBe(true)
  })

  it('toggling a type checkbox updates the store + activates the chip', async () => {
    const { w, gf } = mountChip({ ...DEFAULT_TYPES })
    await w.get(CHIP).trigger('click')
    // Sorted keys: Episode, Insight, Quote, Speaker. Toggle Quote (index 2) on.
    const boxes = w.findAll(`${POPOVER} input[type="checkbox"]`)
    await boxes[2].trigger('change')
    expect(gf.state!.allowedTypes.Quote).toBe(true)
    expect(w.get(CHIP).text()).toContain('Types: 3 of 4 ▾')
  })

  it('the reset button only appears when active, and restores defaults', async () => {
    const { w, gf } = mountChip({
      Episode: true,
      Insight: true,
      Quote: true, // deviates
      Speaker: false,
    })
    await w.get(CHIP).trigger('click')
    expect(w.find(RESET).exists()).toBe(true)

    await w.get(RESET).trigger('click')
    // applyGraphDefaultNodeTypeVisibility flips Quote back off.
    expect(gf.state!.allowedTypes.Quote).toBe(false)
    expect(gf.state!.allowedTypes.Speaker).toBe(false)
    expect(gf.state!.allowedTypes.Episode).toBe(true)
    expect(w.get(CHIP).text()).toContain('Types ▾')
    // Reset button disappears once back at defaults.
    expect(w.find(RESET).exists()).toBe(false)
  })

  it('reset button is hidden while the chip is at spec defaults', async () => {
    const { w } = mountChip({ ...DEFAULT_TYPES })
    await w.get(CHIP).trigger('click')
    expect(w.find(RESET).exists()).toBe(false)
  })

  it('"none" disables every type, "all" enables every type (both deviate)', async () => {
    const { w, gf } = mountChip({ ...DEFAULT_TYPES })
    await w.get(CHIP).trigger('click')
    const [allBtn, noneBtn] = w.get(POPOVER).findAll('button.underline')

    await noneBtn.trigger('click')
    expect(Object.values(gf.state!.allowedTypes).every((v) => v === false)).toBe(true)
    expect(w.get(CHIP).text()).toContain('Types: 0 of 4 ▾')

    await allBtn.trigger('click')
    expect(Object.values(gf.state!.allowedTypes).every((v) => v === true)).toBe(true)
    // All-on still deviates from spec (Quote/Speaker should be off by default).
    expect(w.get(CHIP).text()).toContain('Types: 4 of 4 ▾')
  })
})
