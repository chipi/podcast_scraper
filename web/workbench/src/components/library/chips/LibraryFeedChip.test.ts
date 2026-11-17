// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { CorpusFeedItem } from '../../../api/corpusLibraryApi'
import LibraryFeedChip from './LibraryFeedChip.vue'

const CHIP = '[data-testid="library-chip-feed"]'
const POPOVER = '[data-testid="library-popover-feed"]'
const PANEL = '[data-testid="library-feed-filter-panel"]'
const ALL_BTN = '[data-testid="corpus-feed-filter-all"]'

function feed(overrides: Partial<CorpusFeedItem> = {}): CorpusFeedItem {
  return {
    feed_id: 'feed-a',
    display_title: 'Feed A',
    episode_count: 3,
    ...overrides,
  }
}

const FEEDS: CorpusFeedItem[] = [
  feed({ feed_id: 'feed-a', display_title: 'Feed A', episode_count: 3 }),
  feed({ feed_id: 'feed-b', display_title: 'Feed B', episode_count: 7 }),
]

describe('LibraryFeedChip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  const mountChip = (props: Partial<{
    modelValue: string | null
    feeds: ReadonlyArray<CorpusFeedItem>
    corpusPath: string | null
    loading: boolean
    error: string | null
  }> = {}) =>
    mount(LibraryFeedChip, {
      props: { modelValue: null, feeds: FEEDS, ...props },
      attachTo: document.body,
    })

  it('renders the inactive "All feeds" label by default', () => {
    const w = mountChip()
    const chip = w.get(CHIP)
    expect(chip.text()).toBe('Feed ▾')
    expect(chip.attributes('aria-expanded')).toBe('false')
    expect(chip.attributes('aria-haspopup')).toBe('dialog')
    expect(chip.attributes('aria-label')).toBe('Feed filter')
  })

  it('keeps the popover hidden until the chip is clicked, then toggles it', async () => {
    const w = mountChip()
    expect(w.get(POPOVER).isVisible()).toBe(false)

    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')

    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(false)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
  })

  it('shows the selected feed label when modelValue matches a feed', () => {
    const w = mountChip({ modelValue: 'feed-b' })
    const chip = w.get(CHIP)
    expect(chip.text()).toBe('Feed: Feed B ▾')
    expect(chip.classes()).toContain('font-medium')
  })

  it('falls back to feed_id in the label when the feed has no display title', () => {
    const feeds = [feed({ feed_id: 'feed-c', display_title: null, episode_count: 1 })]
    const w = mountChip({ modelValue: 'feed-c', feeds })
    expect(w.get(CHIP).text()).toBe('Feed: feed-c ▾')
  })

  it('keeps the default label when modelValue points at an unknown feed', () => {
    const w = mountChip({ modelValue: 'does-not-exist' })
    // selectedFeed resolves to null → chipLabel stays "Feed ▾",
    // but isActive is true (modelValue !== null) → active styling.
    expect(w.get(CHIP).text()).toBe('Feed ▾')
    expect(w.get(CHIP).classes()).toContain('font-medium')
  })

  it('forwards model + feed props into the nested CorpusFeedFilterPanel', () => {
    const w = mountChip({ modelValue: 'feed-a' })
    const panel = w.findComponent({ name: 'CorpusFeedFilterPanel' })
    expect(panel.exists()).toBe(true)
    expect(panel.props('modelValue')).toBe('feed-a')
    expect(panel.props('feeds')).toEqual(FEEDS)
  })

  it('re-emits update:modelValue when the panel selects "All feeds"', async () => {
    const w = mountChip({ modelValue: 'feed-a' })
    await w.get(CHIP).trigger('click')
    await w.get(POPOVER).get(ALL_BTN).trigger('click')
    expect(w.emitted('update:modelValue')).toEqual([[null]])
  })

  it('re-emits the chosen feed_id when a feed row is selected', async () => {
    const w = mountChip({ modelValue: null })
    await w.get(CHIP).trigger('click')
    // The list contains the "All feeds" button plus one button per feed,
    // all carrying aria-pressed. Drop the "All feeds" row to get feed rows.
    const all = w
      .get('[data-testid="library-feed-filter-list"]')
      .findAll('button[aria-pressed]')
      .filter((b) => b.attributes('data-testid') !== 'corpus-feed-filter-all')
    expect(all.length).toBe(FEEDS.length)
    await all[1].trigger('click')
    expect(w.emitted('update:modelValue')).toEqual([['feed-b']])
  })

  it('surfaces the loading state through to the panel', async () => {
    const w = mountChip({ loading: true })
    await w.get(CHIP).trigger('click')
    expect(w.get(PANEL).text()).toContain('Loading…')
  })

  it('surfaces an error string through to the panel', async () => {
    const w = mountChip({ error: 'boom' })
    await w.get(CHIP).trigger('click')
    expect(w.get(PANEL).text()).toContain('boom')
  })
})
