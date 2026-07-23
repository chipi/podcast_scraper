// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSearchStore } from '../../stores/search'
import SearchFilterBar from './SearchFilterBar.vue'

const BAR = '[data-testid="search-filter-bar"]'

/** Stub each child so the bar renders standalone and surfaces its props. */
const DateChipStub = {
  name: 'DateChip',
  props: ['modelValue', 'label', 'chipTestid', 'popoverTestid'],
  emits: ['update:modelValue'],
  template: `<button data-testid="stub-date" :data-label="label"
    :data-model="modelValue"
    @click="$emit('update:modelValue', '2024-01-01')" />`,
}
const TopKStub = { name: 'SearchTopKChip', template: '<div data-testid="stub-topk" />' }
const DocTypesStub = {
  name: 'SearchDocTypesChip',
  template: '<div data-testid="stub-doctypes" />',
}
const MoreStub = {
  name: 'SearchMoreChip',
  emits: ['open'],
  template: '<button data-testid="stub-more" @click="$emit(\'open\')" />',
}
// Search v3 §S1 (Explore merge) — 4 chips folded in.
const TopicStub = {
  name: 'SearchTopicChip',
  template: '<div data-testid="stub-topic-contains" />',
}
const SpeakerStub = {
  name: 'SearchSpeakerChip',
  template: '<div data-testid="stub-speaker-contains" />',
}
const MinConfStub = {
  name: 'SearchMinConfidenceChip',
  template: '<div data-testid="stub-min-confidence" />',
}
const GroundedStub = {
  name: 'SearchGroundedChip',
  template: '<div data-testid="stub-grounded-only" />',
}

const STUBS = {
  DateChip: DateChipStub,
  SearchTopKChip: TopKStub,
  SearchDocTypesChip: DocTypesStub,
  SearchMoreChip: MoreStub,
  SearchTopicChip: TopicStub,
  SearchSpeakerChip: SpeakerStub,
  SearchMinConfidenceChip: MinConfStub,
  SearchGroundedChip: GroundedStub,
}

const mountBar = (props: { enabled: boolean; disabledTitle?: string }) =>
  mount(SearchFilterBar, {
    props,
    attachTo: document.body,
    global: { stubs: STUBS },
  })

describe('SearchFilterBar', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the bar region with all eight chips (4 baseline + 4 merged from Explore per S1)', () => {
    const w = mountBar({ enabled: true })
    expect(w.get(BAR).attributes('role')).toBe('region')
    expect(w.get(BAR).attributes('aria-label')).toBe('Search filters')
    const expected = [
      'date',
      'topk',
      'doctypes',
      'topic-contains',
      'speaker-contains',
      'min-confidence',
      'grounded-only',
      'more',
    ]
    for (const id of expected) {
      expect(w.find(`[data-testid="stub-${id}"]`).exists()).toBe(true)
    }
  })

  it('passes the "Since" label and testids to DateChip', () => {
    const w = mountBar({ enabled: true })
    const date = w.get('[data-testid="stub-date"]')
    expect(date.attributes('data-label')).toBe('Since')
  })

  it('binds the since model from the store (null → empty string)', () => {
    const w = mountBar({ enabled: true })
    // search.filters.since defaults to '' so the bound model is empty.
    expect(w.get('[data-testid="stub-date"]').attributes('data-model')).toBe('')
  })

  it('DateChip v-model writes back to search.filters.since', async () => {
    const search = useSearchStore()
    const w = mountBar({ enabled: true })
    await w.get('[data-testid="stub-date"]').trigger('click')
    expect(search.filters.since).toBe('2024-01-01')
  })

  it('reflects an existing store since value into the DateChip model', () => {
    const search = useSearchStore()
    search.filters.since = '2023-12-31'
    const w = mountBar({ enabled: true })
    expect(w.get('[data-testid="stub-date"]').attributes('data-model')).toBe(
      '2023-12-31',
    )
  })

  it('relays the SearchMoreChip "open" event as "open-more"', async () => {
    const w = mountBar({ enabled: true })
    await w.get('[data-testid="stub-more"]').trigger('click')
    expect(w.emitted('open-more')).toHaveLength(1)
  })

  it('when enabled, chip wrappers are not visually disabled and no title shown', () => {
    const w = mountBar({ enabled: true, disabledTitle: 'nope' })
    expect(w.get(BAR).attributes('title')).toBeUndefined()
    // No element carries the disabled wrapper classes when enabled.
    expect(w.find('.pointer-events-none').exists()).toBe(false)
    expect(w.find('.opacity-50').exists()).toBe(false)
  })

  it('when disabled, applies disabled classes and the tooltip title', () => {
    const w = mountBar({ enabled: false, disabledTitle: 'Run a search first' })
    expect(w.get(BAR).attributes('title')).toBe('Run a search first')
    // All nine chip wrappers gain the disabled classes (S1 added 4, S5 added
    // the Enriched chip → 9 total: Since, Top-k, Doc types, Topic, Speaker,
    // Min confidence, Grounded, Enriched, More).
    const disabled = w.findAll('.pointer-events-none')
    expect(disabled).toHaveLength(9)
    expect(
      disabled.every((d) => d.classes().includes('opacity-50')),
    ).toBe(true)
  })

  it('omits the title attribute when disabled but no disabledTitle given', () => {
    const w = mountBar({ enabled: false })
    expect(w.get(BAR).attributes('title')).toBeUndefined()
  })
})
