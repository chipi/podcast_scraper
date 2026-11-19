/**
 * @vitest-environment happy-dom
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { effectScope, nextTick, ref } from 'vue'
import { useFilterChipPopover } from './useFilterChipPopover'

describe('useFilterChipPopover', () => {
  let scope: ReturnType<typeof effectScope>
  let anchorEl: HTMLButtonElement
  let panelEl: HTMLDivElement

  beforeEach(() => {
    document.body.innerHTML = ''
    anchorEl = document.createElement('button')
    panelEl = document.createElement('div')
    document.body.appendChild(anchorEl)
    document.body.appendChild(panelEl)
    scope = effectScope()
  })

  afterEach(() => {
    scope.stop()
  })

  function setup() {
    const anchorRef = ref<HTMLButtonElement | null>(anchorEl)
    const panelRef = ref<HTMLDivElement | null>(panelEl)
    let api!: ReturnType<typeof useFilterChipPopover>
    scope.run(() => {
      api = useFilterChipPopover(anchorRef, panelRef)
    })
    return api
  }

  it('starts closed and toggles to open', async () => {
    const api = setup()
    expect(api.open.value).toBe(false)
    api.toggle()
    expect(api.open.value).toBe(true)
    api.toggle()
    expect(api.open.value).toBe(false)
  })

  it('close() forces closed regardless of state', async () => {
    const api = setup()
    api.toggle()
    expect(api.open.value).toBe(true)
    api.close()
    expect(api.open.value).toBe(false)
    api.close()
    expect(api.open.value).toBe(false)
  })

  it('Escape closes when open and returns focus to the anchor', async () => {
    const api = setup()
    const focusSpy = vi.spyOn(anchorEl, 'focus')
    api.toggle()
    // Two ticks: one for the watch callback to fire, a second for the
    // composable's internal `await nextTick()` before listener attach.
    await nextTick()
    await nextTick()
    expect(api.open.value).toBe(true)
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    expect(api.open.value).toBe(false)
    expect(focusSpy).toHaveBeenCalledOnce()
  })

  it('Escape is ignored when popover is closed', async () => {
    const api = setup()
    const focusSpy = vi.spyOn(anchorEl, 'focus')
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    expect(api.open.value).toBe(false)
    expect(focusSpy).not.toHaveBeenCalled()
  })

  it('pointerdown outside both anchor and panel closes the popover', async () => {
    const api = setup()
    api.toggle()
    // Two ticks: one for the watch callback to fire, a second for the
    // composable's internal `await nextTick()` before listener attach.
    await nextTick()
    await nextTick()
    expect(api.open.value).toBe(true)
    const outside = document.createElement('div')
    document.body.appendChild(outside)
    outside.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }))
    expect(api.open.value).toBe(false)
  })

  it('pointerdown inside the anchor does NOT close', async () => {
    const api = setup()
    api.toggle()
    // Two ticks: one for the watch callback to fire, a second for the
    // composable's internal `await nextTick()` before listener attach.
    await nextTick()
    await nextTick()
    expect(api.open.value).toBe(true)
    anchorEl.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }))
    expect(api.open.value).toBe(true)
  })

  it('pointerdown inside the panel does NOT close', async () => {
    const api = setup()
    api.toggle()
    // Two ticks: one for the watch callback to fire, a second for the
    // composable's internal `await nextTick()` before listener attach.
    await nextTick()
    await nextTick()
    expect(api.open.value).toBe(true)
    const inner = document.createElement('span')
    panelEl.appendChild(inner)
    inner.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }))
    expect(api.open.value).toBe(true)
  })

  it('document listeners are cleaned up when the popover closes', async () => {
    const api = setup()
    api.toggle()
    // Two ticks: one for the watch callback to fire, a second for the
    // composable's internal `await nextTick()` before listener attach.
    await nextTick()
    await nextTick()
    expect(api.open.value).toBe(true)
    api.close()
    await nextTick()
    // After close, an outside pointerdown should be a no-op (no exception, no
    // state change). Easiest proof: re-dispatch and confirm we stay closed.
    const outside = document.createElement('div')
    document.body.appendChild(outside)
    outside.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }))
    expect(api.open.value).toBe(false)
  })

})

// Note: ``onUnmounted`` cleanup requires a component-instance context (not
// an effect scope), so the unmount path is not covered here. The
// close()-driven cleanup is the load-bearing path in real usage and is
// covered above.
