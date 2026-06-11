// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Telemetry: the shell store fires posthog.capture on surface switches. Mock the
// SDK so the slide-toggle interactions don't reach the network.
vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import LeftPanel from './LeftPanel.vue'
import { useShellStore } from '../../stores/shell'
import type { SearchHit } from '../../api/searchApi'

// SearchPanel + ExplorePanel are heavy, API-driven children with their own
// tests. Stub both to passthroughs that re-emit the events LeftPanel forwards,
// so we can assert the parent's emit plumbing and expose contract.
const STUBS = {
  SearchPanel: {
    name: 'SearchPanel',
    emits: ['go-graph', 'open-library-episode', 'open-episode-summary'],
    // Exposes focusQuery so LeftPanel's defineExpose path has a real target.
    template: '<div data-stub="search-panel"></div>',
    methods: {
      focusQuery() {
        // Recorded via the spy injected below.
        ;(this as unknown as { $focusQuerySpy?: () => void }).$focusQuerySpy?.()
      },
    },
  },
  ExplorePanel: {
    name: 'ExplorePanel',
    emits: ['go-graph'],
    template:
      '<div data-stub="explore-panel"><button data-testid="explore-go-graph" @click="$emit(\'go-graph\')"></button></div>',
  },
}

function mountPanel() {
  return mount(LeftPanel, { attachTo: document.body, global: { stubs: STUBS } })
}

const SLIDE_HOST = '[data-testid="left-panel-slide-host"]'
const ENTER_EXPLORE = '[data-testid="left-panel-enter-explore"]'
const BACK_SEARCH = '[data-testid="left-panel-back-search"]'
const EXPLORE_FOOTER = '[data-testid="left-panel-explore-footer"]'

describe('LeftPanel', () => {
  beforeEach(() => setActivePinia(createPinia()))

  // ── Structure / default render ──────────────────────────────────────────────

  it('renders both Search and Explore surfaces (slide track) at mount', () => {
    const w = mountPanel()
    expect(w.find(SLIDE_HOST).exists()).toBe(true)
    // Both panes live in the DOM simultaneously (CSS slide, not v-if).
    expect(w.find('[data-stub="search-panel"]').exists()).toBe(true)
    expect(w.find('[data-stub="explore-panel"]').exists()).toBe(true)
    expect(w.find(EXPLORE_FOOTER).exists()).toBe(true)
  })

  it('defaults to the Search surface (no explore translate class)', () => {
    const w = mountPanel()
    const shell = useShellStore()
    expect(shell.leftPanelSurface).toBe('search')
    const track = w.find(`${SLIDE_HOST} > div`)
    expect(track.classes()).toContain('translate-x-0')
    expect(track.classes()).not.toContain('-translate-x-1/2')
  })

  it('reflects an explore surface set on the store via the translate class', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    shell.setLeftPanelSurface('explore')
    await w.vm.$nextTick()
    const track = w.find(`${SLIDE_HOST} > div`)
    expect(track.classes()).toContain('-translate-x-1/2')
    expect(track.classes()).not.toContain('translate-x-0')
  })

  // ── Interaction: surface switching → store ──────────────────────────────────

  it('Explore corpus button sets the store surface to explore', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    await w.get(ENTER_EXPLORE).trigger('click')
    expect(shell.leftPanelSurface).toBe('explore')
  })

  it('Back to Search button returns the store surface to search', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    shell.setLeftPanelSurface('explore')
    await w.vm.$nextTick()
    await w.get(BACK_SEARCH).trigger('click')
    expect(shell.leftPanelSurface).toBe('search')
  })

  it('exposes accessible labels on the two navigation buttons', () => {
    const w = mountPanel()
    expect(w.get(ENTER_EXPLORE).attributes('aria-label')).toBe('Open Explore corpus mode')
    expect(w.get(BACK_SEARCH).attributes('aria-label')).toBe('Back to semantic search')
  })

  // ── Event forwarding from SearchPanel ───────────────────────────────────────

  it('forwards SearchPanel go-graph as its own go-graph', async () => {
    const w = mountPanel()
    w.findComponent({ name: 'SearchPanel' }).vm.$emit('go-graph')
    await w.vm.$nextTick()
    expect(w.emitted('go-graph')).toHaveLength(1)
  })

  it('forwards SearchPanel open-library-episode with its payload', async () => {
    const w = mountPanel()
    const payload = { metadata_relative_path: 'meta/ep-7.json' }
    w.findComponent({ name: 'SearchPanel' }).vm.$emit('open-library-episode', payload)
    await w.vm.$nextTick()
    expect(w.emitted('open-library-episode')![0]).toEqual([payload])
  })

  it('forwards SearchPanel open-episode-summary with its SearchHit payload', async () => {
    const w = mountPanel()
    const hit = { id: 'hit-1' } as unknown as SearchHit
    w.findComponent({ name: 'SearchPanel' }).vm.$emit('open-episode-summary', hit)
    await w.vm.$nextTick()
    expect(w.emitted('open-episode-summary')![0]).toEqual([hit])
  })

  // ── Event forwarding from ExplorePanel ──────────────────────────────────────

  it('forwards ExplorePanel go-graph as its own go-graph', async () => {
    const w = mountPanel()
    await w.get('[data-testid="explore-go-graph"]').trigger('click')
    expect(w.emitted('go-graph')).toHaveLength(1)
  })

  // ── Exposed focusQuery contract ─────────────────────────────────────────────

  it('exposed focusQuery() switches the store back to search', async () => {
    const w = mountPanel()
    const shell = useShellStore()
    shell.setLeftPanelSurface('explore')
    await w.vm.$nextTick()
    ;(w.vm as unknown as { focusQuery: () => void }).focusQuery()
    await w.vm.$nextTick()
    expect(shell.leftPanelSurface).toBe('search')
  })

  it('exposed focusQuery() calls focusQuery on the SearchPanel ref (after ticks)', async () => {
    const w = mountPanel()
    const search = w.findComponent({ name: 'SearchPanel' })
    const spy = vi.fn()
    ;(search.vm as unknown as { $focusQuerySpy: () => void }).$focusQuerySpy = spy
    ;(w.vm as unknown as { focusQuery: () => void }).focusQuery()
    // defineExpose nests two nextTicks before invoking the child method.
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    expect(spy).toHaveBeenCalledTimes(1)
  })
})
