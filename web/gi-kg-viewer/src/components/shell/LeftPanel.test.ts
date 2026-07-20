// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import LeftPanel from './LeftPanel.vue'
import { useUserPreferencesStore } from '../../stores/userPreferences'

/**
 * LeftPanel is the Search v3 §S4-shell pivot surface — Saved + Recent
 * queries backed by USERPREFS-1. The old compact SearchPanel launcher
 * retired; search now lives only in the main-window Search tab. This spec
 * covers the structural contract + the apply-query emission that the App
 * host routes to the workspace runSearch path.
 */
function mountPanel() {
  return mount(LeftPanel, { attachTo: document.body })
}

describe('LeftPanel (Saved + Recent queries)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders the Saved-queries aside with honest empty states when USERPREFS-1 is unset', () => {
    const w = mountPanel()
    const aside = w.find('[data-testid="left-panel-saved-queries"]')
    expect(aside.exists()).toBe(true)
    expect(aside.attributes('aria-label')).toBe('Saved and recent queries')
    expect(w.find('[data-testid="left-panel-saved-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="left-panel-recent-empty"]').exists()).toBe(true)
    expect(w.find('[data-testid="left-panel-saved-list"]').exists()).toBe(false)
    expect(w.find('[data-testid="left-panel-recent-list"]').exists()).toBe(false)
    w.unmount()
  })

  it('renders saved-queries list from USERPREFS-1 `search.savedQueries`', async () => {
    const prefs = useUserPreferencesStore()
    prefs.set('search.savedQueries', [
      { q: 'llm eval strategy', label: 'LLM eval', ts: 1 },
      { q: 'cell therapy landscape', ts: 2 },
    ])
    const w = mountPanel()
    expect(w.find('[data-testid="left-panel-saved-list"]').exists()).toBe(true)
    expect(w.find('[data-testid="left-panel-saved-empty"]').exists()).toBe(false)
    const buttons = w.findAll('[data-testid="left-panel-saved-list"] button')
    expect(buttons).toHaveLength(2)
    expect(buttons[0].text()).toContain('LLM eval')
    expect(buttons[1].text()).toContain('cell therapy landscape')
    w.unmount()
  })

  it('renders recent-queries list from USERPREFS-1 `search.recentQueries`', async () => {
    const prefs = useUserPreferencesStore()
    prefs.set('search.recentQueries', [{ q: 'graph latency', ts: 5 }])
    const w = mountPanel()
    expect(w.find('[data-testid="left-panel-recent-list"]').exists()).toBe(true)
    expect(w.find('[data-testid="left-panel-recent-empty"]').exists()).toBe(false)
    const buttons = w.findAll('[data-testid="left-panel-recent-list"] button')
    expect(buttons).toHaveLength(1)
    expect(buttons[0].text()).toContain('graph latency')
    w.unmount()
  })

  it('emits apply-query with the raw query text when a Saved row is clicked (labels are display-only)', async () => {
    const prefs = useUserPreferencesStore()
    prefs.set('search.savedQueries', [{ q: 'llm eval strategy', label: 'LLM eval' }])
    const w = mountPanel()
    await w.find('[data-testid="left-panel-saved-list"] button').trigger('click')
    expect(w.emitted('apply-query')).toHaveLength(1)
    expect(w.emitted('apply-query')![0]).toEqual(['llm eval strategy'])
    w.unmount()
  })

  it('emits apply-query when a Recent row is clicked', async () => {
    const prefs = useUserPreferencesStore()
    prefs.set('search.recentQueries', [{ q: 'graph latency', ts: 5 }])
    const w = mountPanel()
    await w.find('[data-testid="left-panel-recent-list"] button').trigger('click')
    expect(w.emitted('apply-query')![0]).toEqual(['graph latency'])
    w.unmount()
  })

  it('caps recent-queries display at 20 entries', () => {
    const prefs = useUserPreferencesStore()
    const many = Array.from({ length: 30 }, (_, i) => ({ q: `q-${i}`, ts: i }))
    prefs.set('search.recentQueries', many)
    const w = mountPanel()
    const buttons = w.findAll('[data-testid="left-panel-recent-list"] button')
    expect(buttons).toHaveLength(20)
    w.unmount()
  })
})
