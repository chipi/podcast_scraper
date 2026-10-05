// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSearchStore } from '../../../stores/search'
import SearchMinConfidenceChip from './SearchMinConfidenceChip.vue'

const CHIP = '[data-testid="search-chip-min-confidence"]'
const INPUT = '[data-testid="search-popover-min-confidence-input"]'

describe('SearchMinConfidenceChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('typing a value keeps the store a string, activates the chip and filters', async () => {
    // The input is type="number", and v-model on a number input hands the store a NUMBER. The chip
    // and the store's filter call .trim() on it: the page threw
    // "search.filters.minConfidence.trim is not a function" and the chip never filtered anything.
    const search = useSearchStore()
    const w = mount(SearchMinConfidenceChip, { attachTo: document.body })
    await w.get(INPUT).setValue('0.7')
    expect(typeof search.filters.minConfidence).toBe('string')
    expect(search.filters.minConfidence).toBe('0.7')
    expect(w.get(CHIP).text()).toContain('Min conf: 0.7')
  })

  it('clearing the input deactivates the chip', async () => {
    const search = useSearchStore()
    search.filters.minConfidence = '0.5'
    const w = mount(SearchMinConfidenceChip, { attachTo: document.body })
    await w.get(INPUT).setValue('')
    expect(search.filters.minConfidence).toBe('')
    expect(w.get(CHIP).text()).toContain('Min conf ▾')
  })
})
