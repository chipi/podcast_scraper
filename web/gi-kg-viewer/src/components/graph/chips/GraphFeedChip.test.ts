// @vitest-environment happy-dom
import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchCorpusFeeds, type CorpusFeedItem } from '../../../api/corpusLibraryApi'
import { useGraphFilterStore } from '../../../stores/graphFilters'
import { useShellStore } from '../../../stores/shell'
import type { GraphFilterState } from '../../../types/artifact'
import GraphFeedChip from './GraphFeedChip.vue'

vi.mock('../../../api/corpusLibraryApi', () => ({
  fetchCorpusFeeds: vi.fn(),
}))

const CHIP = '[data-testid="graph-chip-feed"]'
const POPOVER = '[data-testid="graph-popover-feed"]'
const ALL = '[data-testid="corpus-feed-filter-all"]'

const FEEDS: CorpusFeedItem[] = [
  { feed_id: 'feed-a', display_title: 'Alpha Show', episode_count: 3 },
  { feed_id: 'feed-b', display_title: 'Beta Show', episode_count: 7 },
]

/** Minimal graph-filter state (normally populated by the displayArtifact watcher). */
function freshState(): GraphFilterState {
  return {
    allowedTypes: {},
    allowedEdgeTypes: {},
    hideUngroundedInsights: false,
    showGiLayer: true,
    showKgLayer: true,
    graphFeedFilterId: null,
  }
}

describe('GraphFeedChip', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(fetchCorpusFeeds).mockReset()
    vi.mocked(fetchCorpusFeeds).mockResolvedValue({ path: '/c', feeds: FEEDS })
  })

  /**
   * Seed a corpus path (so onMounted loadFeeds resolves) + graph-filter
   * state, then mount the chip and let onMounted's async loadFeeds settle.
   * PodcastCover is stubbed — it is irrelevant to chip behaviour and pulls
   * in binary-API plumbing.
   */
  const mountChip = async (opts: { corpusPath?: string } = {}) => {
    const shell = useShellStore()
    shell.corpusPath = opts.corpusPath ?? '/c'
    const gf = useGraphFilterStore()
    gf.state = freshState()
    const w = mount(GraphFeedChip, {
      attachTo: document.body,
      global: { stubs: { PodcastCover: true } },
    })
    await flushPromises()
    return { w, gf, shell }
  }

  it('renders the inactive label by default', async () => {
    const { w } = await mountChip()
    expect(w.get(CHIP).text()).toContain('Feed ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
  })

  it('loads feeds on mount using the corpus path', async () => {
    await mountChip({ corpusPath: '/some/corpus' })
    expect(fetchCorpusFeeds).toHaveBeenCalledWith('/some/corpus')
  })

  it('does not fetch feeds when the corpus path is empty', async () => {
    const { w } = await mountChip({ corpusPath: '   ' })
    expect(fetchCorpusFeeds).not.toHaveBeenCalled()
    // Chip still renders (it is always visible, unlike the Sources chip).
    expect(w.get(CHIP).text()).toContain('Feed ▾')
  })

  it('keeps the popover hidden until the chip is clicked, and toggles aria-expanded', async () => {
    const { w } = await mountChip()
    expect(w.get(POPOVER).isVisible()).toBe(false)
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(true)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
    // Clicking again closes it.
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).isVisible()).toBe(false)
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('false')
  })

  it('renders one row per loaded feed with its episode count', async () => {
    const { w } = await mountChip()
    await w.get(CHIP).trigger('click')
    expect(w.text()).toContain('Alpha Show')
    expect(w.text()).toContain('Beta Show')
    expect(w.text()).toContain('(3)')
    expect(w.text()).toContain('(7)')
  })

  it('selecting a feed updates the store + the chip label', async () => {
    const { w, gf } = await mountChip()
    await w.get(CHIP).trigger('click')

    // The feed rows expose aria-pressed buttons; the "All feeds" row is first.
    const rows = w.get(POPOVER).findAll('button[aria-pressed]')
    // rows[0] = All feeds, rows[1] = Alpha, rows[2] = Beta.
    await rows[2].trigger('click')

    expect(gf.state!.graphFeedFilterId).toBe('feed-b')
    expect(w.get(CHIP).text()).toContain('Feed: Beta Show ▾')
    expect(w.get(CHIP).attributes('aria-expanded')).toBe('true')
  })

  it('falls back to the feed_id in the label when there is no display title', async () => {
    vi.mocked(fetchCorpusFeeds).mockResolvedValue({
      path: '/c',
      feeds: [{ feed_id: 'feed-z', display_title: null, episode_count: 1 }],
    })
    const { w, gf } = await mountChip()
    await w.get(CHIP).trigger('click')
    const rows = w.get(POPOVER).findAll('button[aria-pressed]')
    await rows[1].trigger('click')
    expect(gf.state!.graphFeedFilterId).toBe('feed-z')
    expect(w.get(CHIP).text()).toContain('Feed: feed-z ▾')
  })

  it('"All feeds" clears the active feed filter and resets the label', async () => {
    const { w, gf } = await mountChip()
    await w.get(CHIP).trigger('click')
    const rows = w.get(POPOVER).findAll('button[aria-pressed]')
    await rows[1].trigger('click')
    expect(gf.state!.graphFeedFilterId).toBe('feed-a')

    // "All feeds" row emits null → clearFeedFilter().
    await w.get(POPOVER).get(ALL).trigger('click')
    expect(gf.state!.graphFeedFilterId).toBeNull()
    expect(w.get(CHIP).text()).toContain('Feed ▾')
  })

  it('reloads feeds when the corpus path changes', async () => {
    const { shell } = await mountChip({ corpusPath: '/c1' })
    expect(fetchCorpusFeeds).toHaveBeenLastCalledWith('/c1')
    shell.corpusPath = '/c2'
    await flushPromises()
    expect(fetchCorpusFeeds).toHaveBeenLastCalledWith('/c2')
  })

  it('surfaces a load error in the panel instead of feed rows', async () => {
    vi.mocked(fetchCorpusFeeds).mockRejectedValue(new Error('boom'))
    const { w } = await mountChip()
    await w.get(CHIP).trigger('click')
    expect(w.get(POPOVER).text()).toContain('boom')
    // No feed rows rendered on error.
    expect(w.get(POPOVER).find('button[aria-pressed="true"]').exists()).toBe(false)
  })
})
