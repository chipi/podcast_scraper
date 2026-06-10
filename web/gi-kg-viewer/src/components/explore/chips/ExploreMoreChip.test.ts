// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useExploreStore } from '../../../stores/explore'
import ExploreMoreChip from './ExploreMoreChip.vue'

const CHIP = '[data-testid="explore-chip-more"]'

const mountChip = () => mount(ExploreMoreChip, { attachTo: document.body })

describe('ExploreMoreChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the inactive label when all filters are at defaults', () => {
    const w = mountChip()
    expect(w.get(CHIP).text()).toBe('More ▾')
  })

  it('exposes dialog a11y attributes and is a button', () => {
    const w = mountChip()
    const btn = w.get(CHIP)
    expect(btn.attributes('type')).toBe('button')
    expect(btn.attributes('aria-haspopup')).toBe('dialog')
    expect(btn.attributes('aria-label')).toBe('More explore filters')
  })

  it('uses the muted (inactive) class set by default', () => {
    const w = mountChip()
    expect(w.get(CHIP).classes()).toContain('text-muted')
    expect(w.get(CHIP).classes()).not.toContain('font-medium')
  })

  it('emits "open" when clicked', async () => {
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    expect(w.emitted('open')).toHaveLength(1)
  })

  it('counts groundedOnly as one active filter', async () => {
    const w = mountChip()
    const ex = useExploreStore()
    ex.filters.groundedOnly = true
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toBe('More: 1 ▾')
    expect(w.get(CHIP).classes()).toContain('font-medium')
    expect(w.get(CHIP).classes()).toContain('text-surface-foreground')
  })

  it('counts strict as one active filter', async () => {
    const w = mountChip()
    const ex = useExploreStore()
    ex.filters.strict = true
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toBe('More: 1 ▾')
  })

  it('counts a non-default limit as active but ignores the default 50', async () => {
    const w = mountChip()
    const ex = useExploreStore()
    // Default limit (50) should NOT register as active.
    ex.filters.limit = 50
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toBe('More ▾')
    // A different limit registers.
    ex.filters.limit = 25
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toBe('More: 1 ▾')
  })

  it('ignores a non-finite limit (NaN) as inactive', async () => {
    const w = mountChip()
    const ex = useExploreStore()
    ;(ex.filters as { limit: number }).limit = Number.NaN
    await w.vm.$nextTick()
    // Number.isFinite(NaN) === false → not counted.
    expect(w.get(CHIP).text()).toBe('More ▾')
  })

  it('counts a non-default sortBy as active', async () => {
    const w = mountChip()
    const ex = useExploreStore()
    ex.filters.sortBy = 'time'
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toBe('More: 1 ▾')
  })

  it('counts a non-blank minConfidence as active, ignoring whitespace-only', async () => {
    const w = mountChip()
    const ex = useExploreStore()
    // Whitespace-only trims to empty → inactive.
    ex.filters.minConfidence = '   '
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toBe('More ▾')
    // Real value → active.
    ex.filters.minConfidence = '0.8'
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toBe('More: 1 ▾')
  })

  it('sums multiple active filters into the count', async () => {
    const w = mountChip()
    const ex = useExploreStore()
    ex.filters.groundedOnly = true
    ex.filters.strict = true
    ex.filters.limit = 10
    ex.filters.sortBy = 'time'
    ex.filters.minConfidence = '0.5'
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toBe('More: 5 ▾')
  })
})
