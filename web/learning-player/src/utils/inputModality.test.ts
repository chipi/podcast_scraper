// @vitest-environment happy-dom
import { describe, expect, it, vi } from "vitest"

import { returnFocus } from "./inputModality"

describe("returnFocus", () => {
  it("after a tap, hands focus back WITHOUT the keyboard ring", () => {
    const el = document.createElement("button")
    const focus = vi.spyOn(el, "focus")
    window.dispatchEvent(new Event("keydown"))
    window.dispatchEvent(new Event("pointerdown"))
    returnFocus(el)
    expect(focus).toHaveBeenCalledWith({ preventScroll: true, focusVisible: false })
  })

  it("after a key (Escape), the ring shows where focus went", () => {
    const el = document.createElement("button")
    const focus = vi.spyOn(el, "focus")
    window.dispatchEvent(new Event("pointerdown"))
    window.dispatchEvent(new Event("keydown"))
    returnFocus(el)
    expect(focus).toHaveBeenCalledWith({ preventScroll: true, focusVisible: true })
  })
})
