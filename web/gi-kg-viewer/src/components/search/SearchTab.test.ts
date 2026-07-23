// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import SearchTab from './SearchTab.vue'

/**
 * SearchTab is the Search v3 §S2 Workspace shell (§S4-shell pivot: single
 * column, no inner sidebar). Composes an inner SearchPanel (the query
 * surface source of truth) that slices S4 (operator bar), S5 (enriched
 * hero) grow into. Retired: the ``open-episode-summary`` emit — after the
 * L/S buttons were removed, the row-click alone opens the Episode subject
 * panel via ``open-library-episode``.
 */
const STUBS = {
  SearchPanel: {
    name: 'SearchPanel',
    emits: ['go-graph', 'open-library-episode'],
    template: '<div data-stub="search-panel"></div>',
  },
}

function mountTab() {
  return mount(SearchTab, { attachTo: document.body, global: { stubs: STUBS } })
}

describe('SearchTab (Query Workspace shell)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the workspace region + inner SearchPanel stub', () => {
    const w = mountTab()
    const ws = w.find('[data-testid="search-workspace"]')
    expect(ws.exists()).toBe(true)
    expect(ws.attributes('role') ?? 'section').toBeTruthy()
    expect(ws.attributes('aria-label')).toBe('Query workspace')
    expect(w.find('[data-stub="search-panel"]').exists()).toBe(true)
  })

  it('does NOT render the inner Saved+Recent sidebar (moved to LeftPanel in §S4-shell pivot)', () => {
    const w = mountTab()
    // The workspace is a single-column layout since the pivot; Saved+Recent
    // live in the app-level LeftPanel (backed by USERPREFS-1).
    expect(w.find('[data-testid="workspace-sidebar"]').exists()).toBe(false)
    expect(w.find('[data-testid="workspace-sidebar-saved-empty"]').exists()).toBe(false)
    expect(w.find('[data-testid="workspace-sidebar-recent-empty"]').exists()).toBe(false)
  })

  it('forwards SearchPanel go-graph as its own go-graph', async () => {
    const w = mountTab()
    w.findComponent({ name: 'SearchPanel' }).vm.$emit('go-graph')
    await w.vm.$nextTick()
    expect(w.emitted('go-graph')).toHaveLength(1)
  })

  it('forwards SearchPanel open-library-episode with payload', async () => {
    const w = mountTab()
    const payload = { metadata_relative_path: 'meta/ep-9.json' }
    w.findComponent({ name: 'SearchPanel' }).vm.$emit('open-library-episode', payload)
    await w.vm.$nextTick()
    expect(w.emitted('open-library-episode')![0]).toEqual([payload])
  })

  it('does NOT expose an open-episode-summary emit (retired with L/S buttons)', () => {
    const w = mountTab()
    // Retired by the L/S-button removal followup — row-click on ResultCard
    // now maps to open-library-episode; there is no separate summary emit.
    expect(w.emitted('open-episode-summary')).toBeUndefined()
  })
})
