// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import SearchTab from './SearchTab.vue'
import type { SearchHit } from '../../api/searchApi'

/**
 * SearchTab is the Search v3 §S2 Workspace shell. It composes an inner
 * SearchPanel (the query surface source of truth) with the sidebar shell
 * that S4 (operator bar), S5 (enriched-answer hero), and S7 (saved queries)
 * grow into. This spec asserts the structural contract of the shell + the
 * event plumbing to the outer host (App.vue) — it does NOT reach into
 * SearchPanel's internals; those have their own tests.
 */
const STUBS = {
  SearchPanel: {
    name: 'SearchPanel',
    emits: ['go-graph', 'open-library-episode', 'open-episode-summary'],
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

  it('renders the sidebar with Saved + Recent honest empty states (S7 wires)', () => {
    const w = mountTab()
    const sidebar = w.find('[data-testid="workspace-sidebar"]')
    expect(sidebar.exists()).toBe(true)
    // Saved section: placeholder until S7.
    expect(w.find('[data-testid="workspace-sidebar-saved-empty"]').exists()).toBe(true)
    // Recent section: honest empty state when USERPREFS-1 key unset.
    expect(w.find('[data-testid="workspace-sidebar-recent-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="workspace-sidebar-recent-list"]').exists()).toBe(false)
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

  it('forwards SearchPanel open-episode-summary with its SearchHit payload', async () => {
    const w = mountTab()
    const hit = { doc_id: 'hit-1' } as unknown as SearchHit
    w.findComponent({ name: 'SearchPanel' }).vm.$emit('open-episode-summary', hit)
    await w.vm.$nextTick()
    expect(w.emitted('open-episode-summary')![0]).toEqual([hit])
  })
})
