// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSearchStore } from '../../../stores/search'
import SearchTopKChip from './SearchTopKChip.vue'

const CHIP = '[data-testid="search-chip-topk"]'
const POPOVER = '[data-testid="search-popover-topk"]'
const INPUT = '[data-testid="search-popover-topk-input"]'
const RESET = '[data-testid="search-popover-topk-reset"]'

const mountChip = () => mount(SearchTopKChip, { attachTo: document.body })

describe('SearchTopKChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the inactive label at the default top-k', () => {
    const w = mountChip()
    expect(w.get(CHIP).text()).toContain('Top‑k ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
    expect(w.get(CHIP).attributes('aria-label')).toBe('Top‑k results')
  })

  it('keeps the popover hidden until the chip is clicked', async () => {
    const w = mountChip()
    expect(w.get(POPOVER).isVisible()).toBe(false)
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
  })

  it('reflects the store value as active label when not default', async () => {
    const search = useSearchStore()
    search.filters.topK = 25
    const w = mountChip()
    expect(w.get(CHIP).text()).toContain('Top‑k: 25')
  })

  it('editing the number input writes back to the store + activates the label', async () => {
    const search = useSearchStore()
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    const input = w.get(INPUT)
    await input.setValue('42')
    expect(search.filters.topK).toBe(42)
    expect(w.get(CHIP).text()).toContain('Top‑k: 42')
  })

  it('hides the reset button at default and shows it once active', async () => {
    const search = useSearchStore()
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    expect(w.find(RESET).exists()).toBe(false)

    search.filters.topK = 7
    await w.vm.$nextTick()
    expect(w.find(RESET).exists()).toBe(true)
    expect(w.get(RESET).text()).toContain('Reset to 10')
  })

  it('reset restores the default value and closes the popover', async () => {
    const search = useSearchStore()
    search.filters.topK = 50
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)

    await w.get(RESET).trigger('click')
    expect(search.filters.topK).toBe(10)
    expect(w.get(CHIP).text()).toContain('Top‑k ▾')
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })

  it('Enter in the input closes the popover', async () => {
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    await w.get(INPUT).trigger('keydown.enter')
    expect(w.get(POPOVER).isVisible()).toBe(false)
  })
})
