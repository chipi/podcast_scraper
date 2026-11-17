// @vitest-environment happy-dom
import { mount, type VueWrapper } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'

import HelpTip from './HelpTip.vue'

// Teleport mounts the popover into the shared document.body, which persists
// across tests. Track every wrapper and unmount it after each test so a stale
// open panel from a prior test can't be found by a later popover() query.
const mounted: VueWrapper[] = []
afterEach(() => {
  while (mounted.length) mounted.pop()!.unmount()
})

// HelpTip is a trigger button + a Teleported popover (role=tooltip) holding the
// default slot. The popover only mounts while open. Teleport defaults to body,
// so we query document.body for the popover content. We pass attachTo so the
// trigger lives in the real document for pointer-outside-close behaviour.

function mountTip(props: Record<string, unknown> = {}, slot = '<p>Helpful content</p>') {
  const w = mount(HelpTip, {
    props,
    slots: { default: slot },
    attachTo: document.body,
  })
  mounted.push(w)
  return w
}

function popover(): HTMLElement | null {
  return document.body.querySelector('[role="tooltip"]')
}

describe('HelpTip', () => {
  it('renders the default trigger button with "?" label and aria defaults', () => {
    const w = mountTip()
    const btn = w.get('button')
    expect(btn.text()).toBe('?')
    expect(btn.attributes('aria-label')).toBe('Help')
    expect(btn.attributes('aria-expanded')).toBe('false')
  })

  it('uses custom buttonText and buttonAriaLabel when provided', () => {
    const w = mountTip({ buttonText: 'E', buttonAriaLabel: 'Episode ids' })
    const btn = w.get('button')
    expect(btn.text()).toBe('E')
    expect(btn.attributes('aria-label')).toBe('Episode ids')
  })

  it('keeps the popover unmounted until the trigger is clicked', () => {
    mountTip()
    expect(popover()).toBeNull()
  })

  it('opens the popover (rendering the slot) on click and reflects aria-expanded', async () => {
    const w = mountTip({}, '<p data-testid="tip-body">Slotted help</p>')
    await w.get('button').trigger('click')
    expect(w.get('button').attributes('aria-expanded')).toBe('true')
    const panel = popover()
    expect(panel).not.toBeNull()
    expect(panel!.getAttribute('role')).toBe('tooltip')
    expect(panel!.textContent).toContain('Slotted help')
    expect(panel!.querySelector('[data-testid="tip-body"]')).not.toBeNull()
  })

  it('toggles the popover closed on a second click', async () => {
    const w = mountTip()
    await w.get('button').trigger('click')
    expect(popover()).not.toBeNull()
    await w.get('button').trigger('click')
    expect(popover()).toBeNull()
    expect(w.get('button').attributes('aria-expanded')).toBe('false')
  })

  it('closes the popover on an outside pointerdown', async () => {
    const w = mountTip()
    await w.get('button').trigger('click')
    expect(popover()).not.toBeNull()

    // Pointerdown outside trigger + panel closes it (capture-phase listener).
    document.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
    await w.vm.$nextTick()
    expect(popover()).toBeNull()
  })

  it('keeps the popover open when pointerdown lands inside the trigger', async () => {
    const w = mountTip()
    const btn = w.get('button').element as HTMLButtonElement
    await w.get('button').trigger('click')

    const ev = new PointerEvent('pointerdown', { bubbles: true })
    btn.dispatchEvent(ev)
    await w.vm.$nextTick()
    expect(popover()).not.toBeNull()
  })

  it('keeps the popover open when pointerdown lands inside the panel', async () => {
    const w = mountTip()
    await w.get('button').trigger('click')
    const panel = popover()!
    panel.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
    await w.vm.$nextTick()
    expect(popover()).not.toBeNull()
  })

  it('applies a custom buttonClass instead of the default round style', () => {
    const w = mountTip({ buttonClass: 'my-custom-trigger' })
    const btn = w.get('button')
    expect(btn.classes()).toContain('my-custom-trigger')
    // Default style not applied.
    expect(btn.classes()).not.toContain('rounded-full')
  })

  it('removes the document/window listeners on unmount (no popover left behind)', async () => {
    const w = mountTip()
    await w.get('button').trigger('click')
    expect(popover()).not.toBeNull()
    w.unmount()
    // Teleported panel is torn down with the component.
    expect(popover()).toBeNull()
  })
})
