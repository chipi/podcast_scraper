// @vitest-environment happy-dom
import { mount, type VueWrapper } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import HoverRichTip from './HoverRichTip.vue'

// HoverRichTip wraps its default slot in an anchor <div> and Teleports a
// role=tooltip panel into document.body while open. The panel only mounts when
// `active && open`. Show is delayed (360ms) on pointerenter, immediate on
// focus; hide is delayed (140ms). We drive those timers with fake timers and
// query document.body for the teleported panel.
const SHOW_DELAY_MS = 360
const HIDE_DELAY_MS = 140

const mounted: VueWrapper[] = []
afterEach(() => {
  while (mounted.length) mounted.pop()!.unmount()
  vi.clearAllTimers()
  vi.useRealTimers()
})

beforeEach(() => {
  vi.useFakeTimers()
})

function mountTip(
  props: Record<string, unknown> = {},
  slots: Record<string, string> = {
    default: '<button data-testid="anchor-btn">anchor</button>',
    panel: '<p data-testid="tip-panel-body">Rich content</p>',
  },
) {
  const w = mount(HoverRichTip, {
    props,
    slots,
    attachTo: document.body,
  })
  mounted.push(w)
  return w
}

function panel(): HTMLElement | null {
  return document.body.querySelector('[role="tooltip"]')
}

async function flush(w: VueWrapper) {
  await w.vm.$nextTick()
}

describe('HoverRichTip', () => {
  it('renders the default slot inside the anchor and no panel initially', () => {
    const w = mountTip()
    expect(w.get('[data-testid="anchor-btn"]').text()).toBe('anchor')
    expect(panel()).toBeNull()
  })

  it('forwards $attrs onto the anchor element (inheritAttrs:false)', () => {
    const w = mountTip({ class: 'my-anchor', 'data-testid': 'anchor-root' })
    const anchor = w.get('[data-testid="anchor-root"]')
    expect(anchor.classes()).toContain('my-anchor')
  })

  it('does not set aria-describedby while closed', () => {
    const w = mountTip({ 'data-testid': 'anchor-root' })
    expect(w.get('[data-testid="anchor-root"]').attributes('aria-describedby')).toBeUndefined()
  })

  it('opens after the show delay on pointerenter and renders the panel slot', async () => {
    const w = mountTip()
    await w.trigger('pointerenter')
    // Before the delay elapses, still closed.
    expect(panel()).toBeNull()
    vi.advanceTimersByTime(SHOW_DELAY_MS)
    await flush(w)
    const p = panel()
    expect(p).not.toBeNull()
    expect(p!.getAttribute('role')).toBe('tooltip')
    expect(p!.textContent).toContain('Rich content')
  })

  it('sets aria-describedby to the panel id when open', async () => {
    const w = mountTip({ 'data-testid': 'anchor-root' })
    await w.get('[data-testid="anchor-root"]').trigger('pointerenter')
    vi.advanceTimersByTime(SHOW_DELAY_MS)
    await flush(w)
    const described = w.get('[data-testid="anchor-root"]').attributes('aria-describedby')
    expect(described).toBeTruthy()
    expect(panel()!.id).toBe(described)
  })

  it('opens immediately on focusin (no show delay)', async () => {
    const w = mountTip()
    await w.get('[data-testid="anchor-btn"]').trigger('focusin')
    // No timer advance needed.
    await flush(w)
    expect(panel()).not.toBeNull()
  })

  it('ignores a focusin whose target is not within the anchor subtree', async () => {
    const w = mountTip({ 'data-testid': 'anchor-root' })
    const anchor = w.get('[data-testid="anchor-root"]').element as HTMLElement
    // Dispatch focusin directly on the anchor element but with a target that the
    // capture handler reads as outside its subtree: an element detached from the
    // anchor. anchorRef.contains(target) is false → openNow is skipped.
    const stray = document.createElement('span')
    document.body.appendChild(stray)
    const ev = new FocusEvent('focusin', { bubbles: true })
    Object.defineProperty(ev, 'target', { value: stray, configurable: true })
    anchor.dispatchEvent(ev)
    await flush(w)
    stray.remove()
    expect(panel()).toBeNull()
  })

  it('closes after the hide delay on pointerleave', async () => {
    const w = mountTip()
    await w.trigger('pointerenter')
    vi.advanceTimersByTime(SHOW_DELAY_MS)
    await flush(w)
    expect(panel()).not.toBeNull()

    await w.trigger('pointerleave')
    // Still open until hide delay elapses.
    expect(panel()).not.toBeNull()
    vi.advanceTimersByTime(HIDE_DELAY_MS)
    await flush(w)
    expect(panel()).toBeNull()
  })

  it('keeps the panel open when the pointer moves onto the panel before hide fires', async () => {
    const w = mountTip()
    await w.trigger('pointerenter')
    vi.advanceTimersByTime(SHOW_DELAY_MS)
    await flush(w)

    await w.trigger('pointerleave') // schedules hide
    // Pointer enters panel → clears the hide timer.
    panel()!.dispatchEvent(new PointerEvent('pointerenter', { bubbles: true }))
    vi.advanceTimersByTime(HIDE_DELAY_MS)
    await flush(w)
    expect(panel()).not.toBeNull()
  })

  it('closes when the pointer leaves the panel', async () => {
    const w = mountTip()
    await w.trigger('pointerenter')
    vi.advanceTimersByTime(SHOW_DELAY_MS)
    await flush(w)
    const p = panel()!
    p.dispatchEvent(new PointerEvent('pointerleave', { bubbles: true }))
    vi.advanceTimersByTime(HIDE_DELAY_MS)
    await flush(w)
    expect(panel()).toBeNull()
  })

  it('cancels a pending show when the pointer leaves before the show delay', async () => {
    const w = mountTip()
    await w.trigger('pointerenter')
    // Leave before show delay elapses → schedules hide, but open was never set.
    await w.trigger('pointerleave')
    vi.advanceTimersByTime(SHOW_DELAY_MS + HIDE_DELAY_MS)
    await flush(w)
    expect(panel()).toBeNull()
  })

  it('closes on Escape keydown while open', async () => {
    const w = mountTip()
    await w.get('[data-testid="anchor-btn"]').trigger('focusin')
    await flush(w)
    expect(panel()).not.toBeNull()

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    await flush(w)
    expect(panel()).toBeNull()
  })

  it('ignores non-Escape keydown while open', async () => {
    const w = mountTip()
    await w.get('[data-testid="anchor-btn"]').trigger('focusin')
    await flush(w)
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }))
    await flush(w)
    expect(panel()).not.toBeNull()
  })

  it('closes on focusout when focus moves outside the panel', async () => {
    const w = mountTip()
    await w.get('[data-testid="anchor-btn"]').trigger('focusin')
    await flush(w)
    expect(panel()).not.toBeNull()
    // relatedTarget is null → not inside panel → close scheduled.
    await w.get('[data-testid="anchor-btn"]').trigger('focusout')
    vi.advanceTimersByTime(HIDE_DELAY_MS)
    await flush(w)
    expect(panel()).toBeNull()
  })

  it('keeps open on focusout when focus moves into the panel', async () => {
    const w = mountTip()
    await w.get('[data-testid="anchor-btn"]').trigger('focusin')
    await flush(w)
    const p = panel()!
    // focusout whose relatedTarget is inside the panel → no close.
    w.get('[data-testid="anchor-btn"]').element.dispatchEvent(
      new FocusEvent('focusout', { bubbles: true, relatedTarget: p }),
    )
    vi.advanceTimersByTime(HIDE_DELAY_MS)
    await flush(w)
    expect(panel()).not.toBeNull()
  })

  it('does not open when active=false (pass-through wrapper only)', async () => {
    const w = mountTip({ active: false })
    await w.trigger('pointerenter')
    vi.advanceTimersByTime(SHOW_DELAY_MS)
    await flush(w)
    expect(panel()).toBeNull()
    // focus path is also gated.
    await w.get('[data-testid="anchor-btn"]').trigger('focusin')
    await flush(w)
    expect(panel()).toBeNull()
  })

  it('force-closes an open panel when active flips to false', async () => {
    const w = mountTip({ active: true })
    await w.get('[data-testid="anchor-btn"]').trigger('focusin')
    await flush(w)
    expect(panel()).not.toBeNull()
    await w.setProps({ active: false })
    await flush(w)
    expect(panel()).toBeNull()
  })

  it('clamps panel max-width to prefWidth (min floor 200)', async () => {
    const w = mountTip({ prefWidth: 100 })
    await w.get('[data-testid="anchor-btn"]').trigger('focusin')
    await flush(w)
    // prefWidth 100 is below the 200 floor → maxWidth resolves to 200px
    // (assuming a viewport wider than 200 + padding, which happy-dom provides).
    expect(panel()!.style.maxWidth).toBe('200px')
  })

  it('honours a larger prefWidth for the panel max-width', async () => {
    const w = mountTip({ prefWidth: 280 })
    await w.get('[data-testid="anchor-btn"]').trigger('focusin')
    await flush(w)
    expect(panel()!.style.maxWidth).toBe('280px')
  })

  it('tears the teleported panel down on unmount', async () => {
    const w = mountTip()
    await w.get('[data-testid="anchor-btn"]').trigger('focusin')
    await flush(w)
    expect(panel()).not.toBeNull()
    w.unmount()
    expect(panel()).toBeNull()
  })
})
