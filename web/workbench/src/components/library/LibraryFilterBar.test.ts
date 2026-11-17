// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { CorpusFeedItem } from '../../api/corpusLibraryApi'
import LibraryFilterBar from './LibraryFilterBar.vue'

const BAR = '[data-testid="library-filter-bar"]'
const RESET = '[data-testid="library-chip-reset"]'

const FEEDS: CorpusFeedItem[] = [
  { feed_id: 'feed-a', display_title: 'Feed A', episode_count: 3 },
]

/**
 * Stub the three child chips so the bar renders standalone. Each stub
 * surfaces its bound props as JSON and re-emits its v-model event so the
 * bar's event forwarding can be exercised.
 */
const FeedChipStub = {
  name: 'LibraryFeedChip',
  props: ['modelValue', 'feeds', 'corpusPath', 'loading', 'error'],
  emits: ['update:modelValue'],
  template:
    '<button data-testid="stub-feed" :data-props="JSON.stringify($props)" '
    + '@click="$emit(\'update:modelValue\', \'feed-z\')" />',
}

const DateChipStub = {
  name: 'DateChip',
  props: ['modelValue', 'chipTestid', 'popoverTestid'],
  emits: ['update:modelValue'],
  template:
    '<button data-testid="stub-date" :data-props="JSON.stringify($props)" '
    + '@click="$emit(\'update:modelValue\', \'2026-01-01\')" />',
}

const ClusteredChipStub = {
  name: 'LibraryClusteredChip',
  props: ['modelValue'],
  emits: ['update:modelValue'],
  template:
    '<button data-testid="stub-clustered" :data-props="JSON.stringify($props)" '
    + '@click="$emit(\'update:modelValue\', !modelValue)" />',
}

const STUBS = {
  LibraryFeedChip: FeedChipStub,
  DateChip: DateChipStub,
  LibraryClusteredChip: ClusteredChipStub,
}

type BarProps = {
  feedFilterId: string | null
  sinceYmd: string
  topicClusterOnly: boolean
  feeds: ReadonlyArray<CorpusFeedItem>
  corpusPath?: string | null
  feedsLoading?: boolean
  feedsError?: string | null
}

const baseProps: BarProps = {
  feedFilterId: null,
  sinceYmd: '',
  topicClusterOnly: false,
  feeds: FEEDS,
}

const mountBar = (props: Partial<BarProps> = {}) =>
  mount(LibraryFilterBar, {
    props: { ...baseProps, ...props },
    attachTo: document.body,
    global: { stubs: STUBS },
  })

const propsOf = (w: ReturnType<typeof mountBar>, testid: string) =>
  JSON.parse(w.get(`[data-testid="${testid}"]`).attributes('data-props')!)

describe('LibraryFilterBar', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('always renders the bar container and all three chips', () => {
    const w = mountBar()
    expect(w.get(BAR).exists()).toBe(true)
    expect(w.find('[data-testid="stub-feed"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-date"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-clustered"]').exists()).toBe(true)
  })

  it('passes feed-related props down to the Feed chip', () => {
    const w = mountBar({
      feedFilterId: 'feed-a',
      corpusPath: '/corpus',
      feedsLoading: true,
      feedsError: 'oops',
    })
    const p = propsOf(w, 'stub-feed')
    expect(p.modelValue).toBe('feed-a')
    expect(p.feeds).toEqual(FEEDS)
    expect(p.corpusPath).toBe('/corpus')
    expect(p.loading).toBe(true)
    expect(p.error).toBe('oops')
  })

  it('passes sinceYmd and testids down to the Date chip', () => {
    const w = mountBar({ sinceYmd: '2026-05-01' })
    const p = propsOf(w, 'stub-date')
    expect(p.modelValue).toBe('2026-05-01')
    expect(p.chipTestid).toBe('library-chip-date')
    expect(p.popoverTestid).toBe('library-popover-date')
  })

  it('passes topicClusterOnly down to the Clustered chip', () => {
    const w = mountBar({ topicClusterOnly: true })
    expect(propsOf(w, 'stub-clustered').modelValue).toBe(true)
  })

  it('hides the reset button when no filter dimension is active', () => {
    const w = mountBar()
    expect(w.find(RESET).exists()).toBe(false)
  })

  it('shows reset when a feed is selected', () => {
    const w = mountBar({ feedFilterId: 'feed-a' })
    expect(w.find(RESET).exists()).toBe(true)
  })

  it('shows reset when a since date is set', () => {
    const w = mountBar({ sinceYmd: '2026-01-01' })
    expect(w.find(RESET).exists()).toBe(true)
  })

  it('shows reset when topicClusterOnly is on', () => {
    const w = mountBar({ topicClusterOnly: true })
    expect(w.find(RESET).exists()).toBe(true)
  })

  it('treats a whitespace-only sinceYmd as inactive', () => {
    const w = mountBar({ sinceYmd: '   ' })
    expect(w.find(RESET).exists()).toBe(false)
  })

  it('exposes an accessible label and button type on reset', () => {
    const w = mountBar({ topicClusterOnly: true })
    const btn = w.get(RESET)
    expect(btn.attributes('aria-label')).toBe('Reset all library filters')
    expect(btn.attributes('type')).toBe('button')
    expect(btn.text()).toContain('× reset')
  })

  it('emits reset when the reset button is clicked', async () => {
    const w = mountBar({ topicClusterOnly: true })
    await w.get(RESET).trigger('click')
    expect(w.emitted('reset')).toHaveLength(1)
  })

  it('forwards the Feed chip v-model up as update:feedFilterId', async () => {
    const w = mountBar()
    await w.get('[data-testid="stub-feed"]').trigger('click')
    expect(w.emitted('update:feedFilterId')).toEqual([['feed-z']])
  })

  it('forwards the Date chip v-model up as update:sinceYmd', async () => {
    const w = mountBar()
    await w.get('[data-testid="stub-date"]').trigger('click')
    expect(w.emitted('update:sinceYmd')).toEqual([['2026-01-01']])
  })

  it('forwards the Clustered chip v-model up as update:topicClusterOnly', async () => {
    const w = mountBar({ topicClusterOnly: false })
    await w.get('[data-testid="stub-clustered"]').trigger('click')
    expect(w.emitted('update:topicClusterOnly')).toEqual([[true]])
  })
})
