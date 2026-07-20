// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Telemetry: the shell store fires posthog.capture on surface switches. Mock the
// SDK so any left-panel interactions don't reach the network.
vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import LeftPanel from './LeftPanel.vue'
import { useShellStore } from '../../stores/shell'
import type { SearchHit } from '../../api/searchApi'

// SearchPanel is a heavy, API-driven child with its own tests. Stub it to a
// passthrough that re-emits the events LeftPanel forwards, so we can assert the
// parent's emit plumbing and expose contract without dragging in the search API.
const STUBS = {
  SearchPanel: {
    name: 'SearchPanel',
    emits: ['go-graph', 'open-library-episode', 'open-episode-summary'],
    // Exposes focusQuery so LeftPanel's defineExpose path has a real target.
    template: '<div data-stub="search-panel"></div>',
    methods: {
      focusQuery() {
        ;(this as unknown as { $focusQuerySpy?: () => void }).$focusQuerySpy?.()
      },
    },
  },
}

function mountPanel() {
  return mount(LeftPanel, { attachTo: document.body, global: { stubs: STUBS } })
}

// Search v3 §S1 (Explore merge): the slide host + Explore mode-switch test IDs
// (``left-panel-slide-host``, ``left-panel-enter-explore``,
// ``left-panel-back-search``, ``left-panel-explore-footer``) are RETIRED. LeftPanel
// renders SearchPanel directly.

describe('LeftPanel (Search v3 — merged surface)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders only the Search stub — Explore surface + slide host retired (S1)', () => {
    const w = mountPanel()
    expect(w.find('[data-stub="search-panel"]').exists()).toBe(true)
    // Explicit regression assertions: the retired testids MUST NOT reappear.
    expect(w.find('[data-testid="left-panel-slide-host"]').exists()).toBe(false)
    expect(w.find('[data-testid="left-panel-enter-explore"]').exists()).toBe(false)
    expect(w.find('[data-testid="left-panel-back-search"]').exists()).toBe(false)
    expect(w.find('[data-testid="left-panel-explore-footer"]').exists()).toBe(false)
  })

  it('leftPanelSurface stays "search" — only value the union carries after S1', () => {
    mountPanel()
    const shell = useShellStore()
    expect(shell.leftPanelSurface).toBe('search')
  })

  // ── Event forwarding from SearchPanel ─────────────────────────────────────

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

  // ── Exposed focusQuery contract ───────────────────────────────────────────

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
