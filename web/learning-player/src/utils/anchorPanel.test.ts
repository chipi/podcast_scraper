import { describe, expect, it } from "vitest"
import { anchorPanel, type Rect } from "./anchorPanel"

const VIEWPORT = { width: 390, height: 844 } // the operator's iPhone
const PANEL = { width: 224, height: 160 }
const rect = (over: Partial<Rect> = {}): Rect => ({ top: 100, bottom: 130, left: 40, right: 80, ...over })

describe("anchorPanel", () => {
  it("keeps a right-anchored panel fully on screen when the trigger is near the LEFT edge", () => {
    // The reported bug: a 224px panel anchored to a trigger 40–80px from the left ran off the left
    // edge. It must clamp to the margin instead of producing a negative left.
    const { left } = anchorPanel(rect({ left: 40, right: 80 }), PANEL, VIEWPORT, { align: "end" })
    expect(left).toBeGreaterThanOrEqual(8)
    expect(left + PANEL.width).toBeLessThanOrEqual(VIEWPORT.width - 8)
  })

  it("keeps the panel on screen when the trigger is near the RIGHT edge", () => {
    const { left } = anchorPanel(rect({ left: 330, right: 370 }), PANEL, VIEWPORT, { align: "end" })
    expect(left).toBeGreaterThanOrEqual(8)
    expect(left + PANEL.width).toBeLessThanOrEqual(VIEWPORT.width - 8)
  })

  it("aligns the panel's right edge to the trigger's right edge when there is room (align:end)", () => {
    const { left } = anchorPanel(rect({ left: 260, right: 300 }), PANEL, VIEWPORT, { align: "end" })
    expect(left).toBe(300 - PANEL.width)
  })

  it("aligns the panel's left edge to the trigger's left edge for align:start", () => {
    const { left } = anchorPanel(rect({ left: 100, right: 140 }), PANEL, VIEWPORT, { align: "start" })
    expect(left).toBe(100)
  })

  it("hangs below the trigger by default", () => {
    const { top } = anchorPanel(rect({ top: 100, bottom: 130 }), PANEL, VIEWPORT)
    expect(top).toBe(130 + 4)
  })

  it("flips above when there is no room below but room above", () => {
    const { top } = anchorPanel(rect({ top: 800, bottom: 830 }), PANEL, VIEWPORT)
    expect(top).toBe(800 - 4 - PANEL.height)
  })

  it("pins to the top margin when a tall panel fits neither below nor above", () => {
    const tall = { width: 224, height: 800 }
    const { top } = anchorPanel(rect({ top: 400, bottom: 430 }), tall, VIEWPORT)
    expect(top).toBe(8)
  })
})
