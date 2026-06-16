// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useGraphExplorerStore } from '../../stores/graphExplorer'
import GraphBottomBar from './GraphBottomBar.vue'

// happy-dom's built-in Storage is unreliable here (getItem/setItem are not
// callable in this mount context), so back localStorage with a Map — same
// pattern as stores/theme.test.ts.
const storage = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => storage.set(k, v),
  removeItem: (k: string) => storage.delete(k),
})

const BAR = '[data-testid="graph-bottom-bar"]'
const LEFT = '[data-testid="graph-bottom-bar-left"]'
const CENTRE = '[data-testid="graph-bottom-bar-centre"]'
const RIGHT = '[data-testid="graph-bottom-bar-right"]'
const MINIMAP = '[data-testid="graph-minimap-toggle"]'
const RELAYOUT = '[data-testid="graph-relayout"]'
const LAYOUT_CYCLE = '[data-testid="graph-layout-cycle"]'
const FIT = '[data-testid="graph-zoom-fit"]'
const ZOOM_OUT = '[data-testid="graph-zoom-out"]'
const ZOOM_IN = '[data-testid="graph-zoom-in"]'
const ZOOM_RESET = '[data-testid="graph-zoom-reset"]'
const GESTURES = '[data-testid="graph-gesture-overlay-reopen"]'
const EXPORT_PNG = '[data-testid="graph-export-png"]'
const COLLAPSE = '[data-testid="graph-bottom-bar-toggle"]'
const EXPAND = '[data-testid="graph-bottom-bar-expand"]'

const BOTTOM_BAR_COLLAPSED_KEY = 'ps_graph_bottom_bar_collapsed'

// GraphStatusLine pulls in heavy graph/store machinery; stub it so we can assert
// wiring (props + re-emitted events) without mounting the real tree.
const GraphStatusLineStub = {
  name: 'GraphStatusLine',
  props: { variant: { type: String, default: '' }, embedded: { type: Boolean, default: false } },
  emits: ['request-reload', 'request-graph-full-reset'],
  template: '<div data-testid="graph-status-line-stub" />',
}

const baseProps = {
  zoomPercent: 100,
  searchHighlightCount: 0,
  preferredLayout: 'fcose' as const,
}

const mountBar = (props: Record<string, unknown> = {}) =>
  mount(GraphBottomBar, {
    props: { ...baseProps, ...props },
    global: { stubs: { GraphStatusLine: GraphStatusLineStub } },
    attachTo: document.body,
  })

describe('GraphBottomBar', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    storage.clear()
  })

  it('renders expanded by default with the core control groups', () => {
    const w = mountBar()
    expect(w.get(BAR).attributes('aria-expanded')).toBe('true')
    expect(w.find(LEFT).exists()).toBe(true)
    expect(w.find(RIGHT).exists()).toBe(true)
    expect(w.find(EXPAND).exists()).toBe(false)
  })

  it('shows the zoom percentage from props', () => {
    const w = mountBar({ zoomPercent: 175 })
    expect(w.get(RIGHT).text()).toContain('175%')
  })

  it('shows the preferred layout name in the cycle button', () => {
    const w = mountBar({ preferredLayout: 'grid' })
    expect(w.get(LAYOUT_CYCLE).text()).toContain('grid')
  })

  it('computes a layout-cycle title describing current + next layout', () => {
    const w = mountBar({ preferredLayout: 'fcose' })
    // cose -> breadthfirst is the first wrap in GRAPH_LAYOUT_CYCLE_ORDER.
    expect(w.get(LAYOUT_CYCLE).attributes('title')).toBe(
      'Current: fCoSE force-directed. Click to switch to Breadthfirst and re-layout.',
    )
  })

  it('wraps the layout cycle from the last entry back to the first', () => {
    const w = mountBar({ preferredLayout: 'timeline' })
    expect(w.get(LAYOUT_CYCLE).attributes('title')).toBe(
      'Current: Timeline (date axis). Click to switch to fCoSE force-directed and re-layout.',
    )
  })

  it('emits relayout / cycle-layout / fit / zoom / export / gestures on clicks', async () => {
    const w = mountBar()
    await w.get(RELAYOUT).trigger('click')
    await w.get(LAYOUT_CYCLE).trigger('click')
    await w.get(FIT).trigger('click')
    await w.get(ZOOM_OUT).trigger('click')
    await w.get(ZOOM_IN).trigger('click')
    await w.get(ZOOM_RESET).trigger('click')
    await w.get(GESTURES).trigger('click')
    await w.get(EXPORT_PNG).trigger('click')

    const e = w.emitted()
    expect(e.relayout).toHaveLength(1)
    expect(e['cycle-layout']).toHaveLength(1)
    expect(e.fit).toHaveLength(1)
    expect(e['zoom-out']).toHaveLength(1)
    expect(e['zoom-in']).toHaveLength(1)
    expect(e['zoom-reset']).toHaveLength(1)
    expect(e['reopen-gestures']).toHaveLength(1)
    expect(e['export-png']).toHaveLength(1)
  })

  it('toggles the minimap store flag via the minimap button', async () => {
    const w = mountBar()
    const ge = useGraphExplorerStore()
    expect(ge.minimapOpen).toBe(false)

    await w.get(MINIMAP).trigger('click')
    expect(ge.minimapOpen).toBe(true)
    // Active styling reflects the open state.
    expect(w.get(MINIMAP).classes()).toContain('text-primary')

    await w.get(MINIMAP).trigger('click')
    expect(ge.minimapOpen).toBe(false)
  })

  it('hides the Gestures button when showGesturesInBottomBar is false', () => {
    const w = mountBar({ showGesturesInBottomBar: false })
    expect(w.find(GESTURES).exists()).toBe(false)
    expect(w.get(RIGHT).attributes('aria-label')).toBe('Graph fit, zoom, and export')
  })

  it('uses the gestures-inclusive toolbar aria-label by default', () => {
    const w = mountBar()
    expect(w.get(RIGHT).attributes('aria-label')).toBe(
      'Graph fit, zoom, gestures, and export',
    )
  })

  it('omits the centre lens controls unless showLensControls is true', () => {
    const w = mountBar()
    expect(w.find(CENTRE).exists()).toBe(false)
  })

  it('renders the lens controls + re-emits status-line events when showLensControls', async () => {
    const w = mountBar({ showLensControls: true })
    const centre = w.get(CENTRE)
    const stub = centre.getComponent(GraphStatusLineStub)
    expect(stub.props('variant')).toBe('controls')
    expect(stub.props('embedded')).toBe(true)

    // request-reload from the status line surfaces as request-corpus-graph-sync.
    stub.vm.$emit('request-reload')
    await w.vm.$nextTick()
    expect(w.emitted('request-corpus-graph-sync')).toHaveLength(1)

    stub.vm.$emit('request-graph-full-reset')
    await w.vm.$nextTick()
    expect(w.emitted('request-graph-full-reset')).toHaveLength(1)
  })

  it('collapses to the expand affordance and back via the toggle buttons', async () => {
    const w = mountBar()
    await w.get(COLLAPSE).trigger('click')

    expect(w.get(BAR).attributes('aria-expanded')).toBe('false')
    expect(w.find(LEFT).exists()).toBe(false)
    const expand = w.get(EXPAND)
    expect(expand.isVisible()).toBe(true)
    expect(localStorage.getItem(BOTTOM_BAR_COLLAPSED_KEY)).toBe('1')

    await expand.trigger('click')
    expect(w.get(BAR).attributes('aria-expanded')).toBe('true')
    expect(w.find(LEFT).exists()).toBe(true)
    expect(localStorage.getItem(BOTTOM_BAR_COLLAPSED_KEY)).toBeNull()
  })

  it('reads the collapsed state from localStorage on mount', async () => {
    localStorage.setItem(BOTTOM_BAR_COLLAPSED_KEY, '1')
    const w = mountBar()
    await w.vm.$nextTick()
    expect(w.get(BAR).attributes('aria-expanded')).toBe('false')
    expect(w.find(EXPAND).exists()).toBe(true)
  })

  it('force-expands on mount when there are search highlights even if persisted collapsed', () => {
    localStorage.setItem(BOTTOM_BAR_COLLAPSED_KEY, '1')
    const w = mountBar({ searchHighlightCount: 3 })
    expect(w.get(BAR).attributes('aria-expanded')).toBe('true')
  })

  it('auto-expands when search highlights arrive while collapsed', async () => {
    const w = mountBar()
    await w.get(COLLAPSE).trigger('click')
    expect(w.get(BAR).attributes('aria-expanded')).toBe('false')

    await w.setProps({ searchHighlightCount: 2 })
    expect(w.get(BAR).attributes('aria-expanded')).toBe('true')
  })

  it('toggles collapsed via the Alt+B global keyboard shortcut', async () => {
    const w = mountBar()
    expect(w.get(BAR).attributes('aria-expanded')).toBe('true')

    window.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'b', altKey: true, bubbles: true }),
    )
    await w.vm.$nextTick()
    expect(w.get(BAR).attributes('aria-expanded')).toBe('false')

    window.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'B', altKey: true, bubbles: true }),
    )
    await w.vm.$nextTick()
    expect(w.get(BAR).attributes('aria-expanded')).toBe('true')
  })

  it('ignores Alt+B when typing in an editable field', async () => {
    const w = mountBar()
    const input = document.createElement('input')
    document.body.appendChild(input)

    const ev = new KeyboardEvent('keydown', { key: 'b', altKey: true, bubbles: true })
    Object.defineProperty(ev, 'target', { value: input })
    window.dispatchEvent(ev)
    await w.vm.$nextTick()

    expect(w.get(BAR).attributes('aria-expanded')).toBe('true')
    input.remove()
  })

  it('ignores keydown that is not Alt+B', async () => {
    const w = mountBar()
    window.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'b', altKey: false, bubbles: true }),
    )
    window.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'x', altKey: true, bubbles: true }),
    )
    await w.vm.$nextTick()
    expect(w.get(BAR).attributes('aria-expanded')).toBe('true')
  })

  it('removes its global keydown listener on unmount', async () => {
    const w = mountBar()
    w.unmount()
    // Dispatching after unmount must not throw and there is no bar to toggle.
    expect(() =>
      window.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'b', altKey: true, bubbles: true }),
      ),
    ).not.toThrow()
  })
})
