// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useGraphExplorerStore } from '../../../stores/graphExplorer'
import GraphDegreeChip from './GraphDegreeChip.vue'

const CHIP = '[data-testid="graph-chip-degree"]'
const POPOVER = '[data-testid="graph-popover-degree"]'

describe('GraphDegreeChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  const mountChip = (degreeHistogramCounts: Record<string, number> = {}) =>
    mount(GraphDegreeChip, { props: { degreeHistogramCounts }, attachTo: document.body })

  it('renders the inactive label by default', () => {
    const w = mountChip()
    expect(w.get(CHIP).text()).toContain('Degree ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
  })

  it('keeps the popover hidden until the chip is clicked', async () => {
    const w = mountChip()
    expect(w.get(POPOVER).isVisible()).toBe(false)
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
  })

  it('renders the per-bucket counts from props', () => {
    const w = mountChip({ '2-5': 5 })
    expect(w.text()).toContain('(5)')
    // Buckets with no count fall back to 0.
    expect(w.text()).toContain('(0)')
  })

  it('selecting a bucket updates the store + the chip label, and Clear resets it', async () => {
    const w = mountChip({ '1-2': 5 })
    const ge = useGraphExplorerStore()
    await w.get(CHIP).trigger('click') // open popover

    // Click the first bucket button inside the popover.
    const buckets = w.get(POPOVER).findAll('button[aria-pressed]')
    expect(buckets.length).toBeGreaterThan(0)
    const firstBucketLabel = buckets[0].text().split(' ')[0]
    await buckets[0].trigger('click')

    expect(ge.activeDegreeBucket).toBe(firstBucketLabel)
    expect(w.get(CHIP).text()).toContain(`Degree: ${firstBucketLabel}`)

    // The Clear button appears once active; clicking it resets.
    const clear = w.get(POPOVER).get('button[aria-label="Clear degree filter"]')
    await clear.trigger('click')
    expect(ge.activeDegreeBucket).toBeFalsy()
    expect(w.get(CHIP).text()).toContain('Degree ▾')
  })

  it('toggling the same bucket twice clears it (store round-trip)', async () => {
    const w = mountChip()
    const ge = useGraphExplorerStore()
    await w.get(CHIP).trigger('click')
    const buckets = w.get(POPOVER).findAll('button[aria-pressed]')
    await buckets[0].trigger('click')
    expect(ge.activeDegreeBucket).toBeTruthy()
    await buckets[0].trigger('click')
    expect(ge.activeDegreeBucket).toBeFalsy()
  })
})
