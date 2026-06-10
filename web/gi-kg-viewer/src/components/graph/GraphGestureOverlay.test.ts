// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import GraphGestureOverlay from './GraphGestureOverlay.vue'

const STORAGE_KEY = 'ps_graph_hints_seen'

// happy-dom in this repo ships a bare `localStorage` without Storage methods,
// so follow the established theme.test.ts pattern: a module-level Map-backed
// stub (applied once) cleared in beforeEach. Installing the stub inside a hook
// is flaky here, so it must live at module top level.
const storage = new Map<string, string>()

vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => storage.set(k, v),
  removeItem: (k: string) => storage.delete(k),
  clear: () => storage.clear(),
})
const OVERLAY = '[data-testid="graph-gesture-overlay"]'
const DISMISS = '[data-testid="graph-gesture-overlay-dismiss"]'

// The overlay's auto-show runs in onMounted (sets `visible=true`). That reactive
// change is applied *after* mount() returns, so flush one tick before asserting
// (the watcher-driven show path is already awaited via setProps elsewhere).
const mountOverlay = async (hasNodes: boolean) => {
  const w = mount(GraphGestureOverlay, { props: { hasNodes }, attachTo: document.body })
  await w.vm.$nextTick()
  return w
}

describe('GraphGestureOverlay', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    storage.clear()
  })

  it('does not auto-show when there are no nodes', async () => {
    const w = await mountOverlay(false)
    expect(w.find(OVERLAY).exists()).toBe(false)
  })

  it('auto-shows on mount when nodes exist and hints were never dismissed', async () => {
    const w = await mountOverlay(true)
    expect(w.find(OVERLAY).exists()).toBe(true)
  })

  it('does not auto-show when localStorage marks hints already seen', async () => {
    localStorage.setItem(STORAGE_KEY, '1')
    const w = await mountOverlay(true)
    expect(w.find(OVERLAY).exists()).toBe(false)
  })

  it('renders the dialog with the expected a11y attributes and title', async () => {
    const w = await mountOverlay(true)
    const dialog = w.get('[role="dialog"]')
    expect(dialog.attributes('aria-modal')).toBe('true')
    expect(dialog.attributes('aria-labelledby')).toBe('graph-gesture-dialog-title')
    expect(w.get('#graph-gesture-dialog-title').text()).toBe('Graph gestures')
  })

  it('lists all six gesture hint rows', async () => {
    const w = await mountOverlay(true)
    const rows = w.findAll('ul li')
    expect(rows.length).toBe(6)
    const text = w.get(OVERLAY).text()
    expect(text).toContain('Open node details')
    expect(text).toContain('Alt + B')
    expect(text).toContain('Expand 1-hop neighbourhood')
    expect(text).toContain('Box zoom / select')
  })

  it('renders the teal + blue ring legend', async () => {
    const w = await mountOverlay(true)
    const text = w.get(OVERLAY).text()
    expect(text).toContain('More episodes available (teal ring)')
    expect(text).toContain('Episodes loaded from here (blue ring)')
  })

  it('dismiss button hides the overlay, persists the flag, and emits dismissed', async () => {
    const w = await mountOverlay(true)
    expect(w.find(OVERLAY).exists()).toBe(true)

    await w.get(DISMISS).trigger('click')

    expect(w.find(OVERLAY).exists()).toBe(false)
    expect(localStorage.getItem(STORAGE_KEY)).toBe('1')
    expect(w.emitted('dismissed')).toHaveLength(1)
  })

  it('clicking the backdrop (self) dismisses the overlay', async () => {
    const w = await mountOverlay(true)
    await w.get(OVERLAY).trigger('click')
    expect(w.find(OVERLAY).exists()).toBe(false)
    expect(w.emitted('dismissed')).toHaveLength(1)
  })

  it('clicking inside the dialog does not dismiss the overlay', async () => {
    const w = await mountOverlay(true)
    await w.get('[role="dialog"]').trigger('click')
    expect(w.find(OVERLAY).exists()).toBe(true)
    expect(w.emitted('dismissed')).toBeUndefined()
  })

  it('hides and resets when hasNodes flips to false', async () => {
    const w = await mountOverlay(true)
    expect(w.find(OVERLAY).exists()).toBe(true)
    await w.setProps({ hasNodes: false })
    expect(w.find(OVERLAY).exists()).toBe(false)
  })

  it('auto-opens when hasNodes flips from false to true', async () => {
    const w = await mountOverlay(false)
    expect(w.find(OVERLAY).exists()).toBe(false)
    await w.setProps({ hasNodes: true })
    expect(w.find(OVERLAY).exists()).toBe(true)
  })

  it('does not auto-open on hasNodes transition when already dismissed', async () => {
    const w = await mountOverlay(false)
    localStorage.setItem(STORAGE_KEY, '1')
    await w.setProps({ hasNodes: true })
    expect(w.find(OVERLAY).exists()).toBe(false)
  })

  it('exposed reopen() shows the overlay even after it was dismissed', async () => {
    const w = await mountOverlay(true)
    await w.get(DISMISS).trigger('click')
    expect(w.find(OVERLAY).exists()).toBe(false)

    // reopen bypasses the storage-dismissed guard (manualOpen path).
    ;(w.vm as unknown as { reopen: () => void }).reopen()
    await w.vm.$nextTick()
    expect(w.find(OVERLAY).exists()).toBe(true)
  })

  it('exposed reopen() is a no-op when there are no nodes', async () => {
    const w = await mountOverlay(false)
    ;(w.vm as unknown as { reopen: () => void }).reopen()
    await w.vm.$nextTick()
    expect(w.find(OVERLAY).exists()).toBe(false)
  })

  it('Escape key dismisses the overlay when focus is outside the dialog', async () => {
    const w = await mountOverlay(true)
    expect(w.find(OVERLAY).exists()).toBe(true)
    // Let the visible watcher attach the Esc listener (nextTick inside watch).
    await w.vm.$nextTick()
    await w.vm.$nextTick()

    document.body.focus()
    const evt = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })
    window.dispatchEvent(evt)
    await w.vm.$nextTick()

    expect(w.find(OVERLAY).exists()).toBe(false)
    expect(w.emitted('dismissed')).toHaveLength(1)
  })
})
