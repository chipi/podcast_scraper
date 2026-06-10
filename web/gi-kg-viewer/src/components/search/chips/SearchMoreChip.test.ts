// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSearchStore } from '../../../stores/search'
import SearchMoreChip from './SearchMoreChip.vue'

const CHIP = '[data-testid="search-chip-more"]'

const mountChip = () => mount(SearchMoreChip, { attachTo: document.body })

describe('SearchMoreChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the inactive label by default', () => {
    const w = mountChip()
    expect(w.get(CHIP).text()).toContain('More ▾')
    expect(w.get(CHIP).attributes('aria-haspopup')).toBe('dialog')
    expect(w.get(CHIP).attributes('aria-label')).toBe('More search filters')
    expect(w.get(CHIP).attributes('type')).toBe('button')
  })

  it('emits "open" when clicked', async () => {
    const w = mountChip()
    await w.get(CHIP).trigger('click')
    expect(w.emitted('open')).toHaveLength(1)
  })

  it('groundedOnly marks the chip active with count 1', async () => {
    const search = useSearchStore()
    const w = mountChip()
    search.filters.groundedOnly = true
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toContain('More: 1')
  })

  it('counts feed/speaker/embeddingModel only when non-blank (trimmed)', async () => {
    const search = useSearchStore()
    const w = mountChip()
    // Whitespace-only stays inactive.
    search.filters.feed = '   '
    search.filters.speaker = '\t'
    search.filters.embeddingModel = ' '
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toContain('More ▾')

    search.filters.feed = 'daily'
    search.filters.speaker = 'alice'
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toContain('More: 2')
  })

  it('dedupeKgSurfaces=false (merge surfaces off) counts as active', async () => {
    const search = useSearchStore()
    const w = mountChip()
    search.filters.dedupeKgSurfaces = false
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toContain('More: 1')
  })

  it('sums every active dimension', async () => {
    const search = useSearchStore()
    const w = mountChip()
    search.filters.groundedOnly = true
    search.filters.feed = 'f'
    search.filters.speaker = 's'
    search.filters.embeddingModel = 'm'
    search.filters.dedupeKgSurfaces = false
    await w.vm.$nextTick()
    expect(w.get(CHIP).text()).toContain('More: 5')
  })
})
