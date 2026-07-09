// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

/**
 * UXS-015 / RFC-104 — the Library tab mode toggle. Verifies Episodes is the
 * default, the toggle switches modes, and the choice persists to localStorage.
 *
 * The child views are module-mocked (not just render-stubbed): the real
 * LibraryView.vue is ~900 lines with no own test, so a static stub would still
 * pull it into v8's "imported by a test" coverage scope and tank the headline
 * function %. vi.mock keeps the real module off the graph entirely.
 */
vi.mock('./LibraryView.vue', () => ({
  default: { name: 'LibraryView', template: '<div data-testid="library-view-stub" />' },
}))
vi.mock('./ShowsBrowse.vue', () => ({
  default: { name: 'ShowsBrowse', template: '<div data-testid="shows-browse-stub" />' },
}))

// Imported after the mocks so LibraryTab picks up the mocked children.
import LibraryTab from './LibraryTab.vue'

const STORAGE_KEY = 'gikg.library.mode'

const storage = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => storage.set(k, v),
  removeItem: (k: string) => storage.delete(k),
})

function mountTab() {
  return mount(LibraryTab)
}

beforeEach(() => {
  setActivePinia(createPinia())
  storage.clear()
})

afterEach(() => {
  storage.clear()
})

describe('LibraryTab — mode toggle + persistence', () => {
  it('defaults to Episodes mode (status quo; Shows is opt-in — PRD-044 OQ1)', () => {
    const w = mountTab()
    expect(w.find('[data-testid="library-mode-episodes"]').attributes('aria-pressed')).toBe('true')
    expect(w.html()).toContain('library-view-stub')
    expect(w.html()).not.toContain('shows-browse-stub')
  })

  it('switches to Shows and persists the choice', async () => {
    const w = mountTab()
    await w.find('[data-testid="library-mode-shows"]').trigger('click')

    expect(w.find('[data-testid="library-mode-shows"]').attributes('aria-pressed')).toBe('true')
    expect(w.html()).toContain('shows-browse-stub')
    expect(localStorage.getItem(STORAGE_KEY)).toBe('shows')
  })

  it('switches back to Episodes', async () => {
    const w = mountTab()
    await w.find('[data-testid="library-mode-shows"]').trigger('click')
    await w.find('[data-testid="library-mode-episodes"]').trigger('click')

    expect(w.html()).toContain('library-view-stub')
    expect(localStorage.getItem(STORAGE_KEY)).toBe('episodes')
  })

  it('restores the persisted Shows mode on a fresh mount', () => {
    localStorage.setItem(STORAGE_KEY, 'shows')
    const w = mountTab()
    expect(w.find('[data-testid="library-mode-shows"]').attributes('aria-pressed')).toBe('true')
    expect(w.html()).toContain('shows-browse-stub')
  })

  it('forwards the switch-main-tab event from the episodes view', async () => {
    const w = mountTab()
    // Episodes is the default → LibraryView is mounted. It emits switch-main-tab;
    // LibraryTab must re-emit it to App.vue.
    w.findComponent({ name: 'LibraryView' }).vm.$emit('switch-main-tab', 'graph')
    await w.vm.$nextTick()
    expect(w.emitted('switch-main-tab')?.[0]).toEqual(['graph'])
  })
})
