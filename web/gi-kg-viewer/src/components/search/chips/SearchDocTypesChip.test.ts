// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSearchStore } from '../../../stores/search'
import SearchDocTypesChip from './SearchDocTypesChip.vue'

const CHIP = '[data-testid="search-chip-doctypes"]'
const POPOVER = '[data-testid="search-popover-doctypes"]'
const CLEAR = '[data-testid="search-popover-doctypes-clear"]'

const mountChip = () =>
  mount(SearchDocTypesChip, { attachTo: document.body })

describe('SearchDocTypesChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the inactive label by default', () => {
    const w = mountChip()
    expect(w.get(CHIP).text()).toContain('Doc types ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
    expect(w.get(CHIP).attributes('aria-haspopup')).toBe('dialog')
    expect(w.get(CHIP).attributes('aria-label')).toBe('Doc types filter')
  })

  it('keeps the popover hidden until the chip is clicked', async () => {
    const w = mountChip()
    expect(w.get(POPOVER).isVisible()).toBe(false)
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
    // Toggling again closes it.
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })

  it('renders all six type options', () => {
    const w = mountChip()
    const labels = w.get(POPOVER).findAll('li span').map((s) => s.text())
    expect(labels).toEqual([
      'Insights',
      'Quotes',
      'KG entities',
      'KG topics',
      'Summary bullets',
      'Transcript chunks',
    ])
  })

  it('checkbox reflects store selection via :checked', async () => {
    const search = useSearchStore()
    search.filters.types.push('quote')
    const w = mountChip()
    const boxes = w.get(POPOVER).findAll('input[type="checkbox"]')
    // Order matches TYPE_OPTIONS: insight, quote, ...
    expect((boxes[0].element as HTMLInputElement).checked).toBe(false)
    expect((boxes[1].element as HTMLInputElement).checked).toBe(true)
  })

  it('toggling a checkbox adds then removes the type from the store + updates label', async () => {
    const search = useSearchStore()
    const w = mountChip()
    const boxes = w.get(POPOVER).findAll('input[type="checkbox"]')

    await boxes[0].trigger('change') // insight on
    expect(search.filters.types).toContain('insight')
    expect(w.get(CHIP).text()).toContain('Doc types: 1 of 6')

    await boxes[2].trigger('change') // kg_entity on
    expect(search.filters.types).toContain('kg_entity')
    expect(w.get(CHIP).text()).toContain('Doc types: 2 of 6')

    await boxes[0].trigger('change') // insight off
    expect(search.filters.types).not.toContain('insight')
    expect(w.get(CHIP).text()).toContain('Doc types: 1 of 6')
  })

  it('hides the clear button until a type is active, and clear empties the store', async () => {
    const search = useSearchStore()
    const w = mountChip()
    // Inactive: clear button not rendered.
    expect(w.find(CLEAR).exists()).toBe(false)

    search.filters.types.push('insight', 'quote')
    await w.vm.$nextTick()
    expect(w.find(CLEAR).exists()).toBe(true)

    await w.get(CLEAR).trigger('click')
    expect(search.filters.types).toHaveLength(0)
    expect(w.get(CHIP).text()).toContain('Doc types ▾')
    expect(w.find(CLEAR).exists()).toBe(false)
  })
})
