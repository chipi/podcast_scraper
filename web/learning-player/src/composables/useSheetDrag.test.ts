import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'
import { SHEET_DISMISS_PX, shouldDismissSheet, useSheetDrag } from './useSheetDrag'

describe('shouldDismissSheet (operator 2026-10-10: pull the handle down to close)', () => {
  it('a long pull closes, however slow', () => {
    expect(shouldDismissSheet(SHEET_DISMISS_PX, 2000)).toBe(true)
  })
  it('a quick flick closes from a short pull', () => {
    expect(shouldDismissSheet(40, 50)).toBe(true)
  })
  it('a short, slow pull springs back; so does a twitch', () => {
    expect(shouldDismissSheet(40, 1000)).toBe(false)
    expect(shouldDismissSheet(10, 5)).toBe(false)
  })
})

describe('useSheetDrag', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  function setup() {
    const el = document.createElement('div')
    const onDismiss = vi.fn()
    const drag = useSheetDrag(ref(el), onDismiss)
    const ev = (clientY: number) => ({ clientY, pointerType: 'touch', button: 0, currentTarget: null }) as unknown as PointerEvent
    return { el, onDismiss, drag, ev }
  }

  it('the sheet follows the finger down, never up', () => {
    const { el, drag, ev } = setup()
    drag.onPointerdown(ev(100))
    drag.onPointermove(ev(160))
    expect(el.style.transform).toBe('translateY(60px)')
    drag.onPointermove(ev(40))
    expect(el.style.transform).toBe('')
  })

  it('past the threshold it closes, and the sheet is put back for the next open', () => {
    const { el, onDismiss, drag, ev } = setup()
    drag.onPointerdown(ev(100))
    drag.onPointermove(ev(100 + SHEET_DISMISS_PX + 10))
    drag.onPointerup()
    expect(onDismiss).not.toHaveBeenCalled()
    vi.advanceTimersByTime(300)
    expect(onDismiss).toHaveBeenCalledTimes(1)
    expect(el.style.transform).toBe('')
  })

  it('short of it the sheet springs back and stays open; a cancelled pull too', () => {
    const { el, onDismiss, drag, ev } = setup()
    drag.onPointerdown(ev(100))
    drag.onPointermove(ev(130))
    drag.onPointerup()
    vi.advanceTimersByTime(300)
    expect(onDismiss).not.toHaveBeenCalled()
    expect(el.style.transform).toBe('')
    drag.onPointerdown(ev(100))
    drag.onPointermove(ev(400))
    drag.onPointercancel()
    vi.advanceTimersByTime(300)
    expect(onDismiss).not.toHaveBeenCalled()
    expect(el.style.transform).toBe('')
  })
})
