// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import LibraryTab from './LibraryTab.vue'

/**
 * UXS-015 / RFC-104 — the Library tab mode toggle. Verifies Shows is the default,
 * the toggle switches modes, and the choice persists to localStorage. The heavy
 * child views are stubbed — this test owns the toggle + persistence contract only.
 */

const STORAGE_KEY = 'gikg.library.mode'

const storage = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => storage.set(k, v),
  removeItem: (k: string) => storage.delete(k),
})

function mountTab() {
  return mount(LibraryTab, {
    global: { stubs: { ShowsBrowse: true, LibraryView: true } },
  })
}

beforeEach(() => {
  setActivePinia(createPinia())
  storage.clear()
})

afterEach(() => {
  storage.clear()
})

describe('LibraryTab — mode toggle + persistence', () => {
  it('defaults to Shows mode', () => {
    const w = mountTab()
    expect(w.find('[data-testid="library-mode-shows"]').attributes('aria-pressed')).toBe('true')
    expect(w.html()).toContain('shows-browse-stub')
    expect(w.html()).not.toContain('library-view-stub')
  })

  it('switches to Episodes and persists the choice', async () => {
    const w = mountTab()
    await w.find('[data-testid="library-mode-episodes"]').trigger('click')

    expect(w.find('[data-testid="library-mode-episodes"]').attributes('aria-pressed')).toBe('true')
    expect(w.html()).toContain('library-view-stub')
    expect(localStorage.getItem(STORAGE_KEY)).toBe('episodes')
  })

  it('switches back to Shows', async () => {
    const w = mountTab()
    await w.find('[data-testid="library-mode-episodes"]').trigger('click')
    await w.find('[data-testid="library-mode-shows"]').trigger('click')

    expect(w.html()).toContain('shows-browse-stub')
    expect(localStorage.getItem(STORAGE_KEY)).toBe('shows')
  })

  it('restores the persisted mode on a fresh mount', () => {
    localStorage.setItem(STORAGE_KEY, 'episodes')
    const w = mountTab()
    expect(w.find('[data-testid="library-mode-episodes"]').attributes('aria-pressed')).toBe('true')
    expect(w.html()).toContain('library-view-stub')
  })

  it('forwards the switch-main-tab event from the episodes view', async () => {
    localStorage.setItem(STORAGE_KEY, 'episodes')
    const w = mountTab()
    // The stubbed LibraryView emits switch-main-tab; LibraryTab must re-emit it.
    w.findComponent({ name: 'LibraryView' }).vm.$emit('switch-main-tab', 'graph')
    await w.vm.$nextTick()
    expect(w.emitted('switch-main-tab')?.[0]).toEqual(['graph'])
  })
})
