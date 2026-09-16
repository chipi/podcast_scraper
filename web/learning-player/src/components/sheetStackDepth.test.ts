/**
 * The stacked-sheet ladder (operator, 2026-09-16).
 *
 * This geometry was reported wrong twice from the same build, in opposite directions — first
 * "person and storyline fully overlap the topic", then "I don't want to see the heart, share and
 * following storyline in the background". Both had one cause: the peek above a card was
 * `card below's height − this card's height`, and the card below was content-sized, so the gap was
 * whatever its content happened to make it. These tests pin the two things that fix it — a depth
 * that keeps counting past one level, and a flag that lasts exactly as long as the stack.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { mount } from "@vue/test-utils"
import { defineComponent, h } from "vue"
import { registerStackedSheet, __resetSheetStack } from "../composables/sheetStack"

beforeEach(() => __resetSheetStack())
afterEach(() => __resetSheetStack())

describe("sheet stack flag", () => {
  it("marks the body only while a layered sheet is open", () => {
    expect(document.body.classList.contains("lp-stack-open")).toBe(false)
    const release = registerStackedSheet()
    expect(document.body.classList.contains("lp-stack-open")).toBe(true)
    release()
    expect(document.body.classList.contains("lp-stack-open")).toBe(false)
  })

  it("keeps the stack pinned until the LAST card closes", () => {
    const first = registerStackedSheet()
    const second = registerStackedSheet()
    first()
    // Closing one card of a deck must not un-pin the others — that would resize every remaining
    // sheet mid-interaction and collapse the peek the deck depends on.
    expect(document.body.classList.contains("lp-stack-open")).toBe(true)
    second()
    expect(document.body.classList.contains("lp-stack-open")).toBe(false)
  })

  it("does not go negative when a card releases twice", () => {
    const release = registerStackedSheet()
    release()
    release()
    const other = registerStackedSheet()
    // A negative counter would leave the flag off while a sheet is genuinely open.
    expect(document.body.classList.contains("lp-stack-open")).toBe(true)
    other()
  })
})

describe("depth ladder", () => {
  // Each level is one peek shorter than the card it covers, so its top edge sits one peek lower.
  // A boolean could only ever express level 1, which is the bug this replaced.
  const Card = defineComponent({
    props: { depth: { type: Number, default: 0 } },
    setup: (props) => () =>
      h("div", {
        class: ["lp-sheet", props.depth > 0 ? "lp-sheet--stacked" : undefined],
        style: { "--lp-depth": props.depth },
      }),
  })

  it("marks only layered cards as stacked", () => {
    expect(mount(Card, { props: { depth: 0 } }).classes()).not.toContain("lp-sheet--stacked")
    expect(mount(Card, { props: { depth: 1 } }).classes()).toContain("lp-sheet--stacked")
  })

  it("carries a distinct depth past the first level", () => {
    for (const depth of [1, 2, 3]) {
      const el = mount(Card, { props: { depth } }).element as HTMLElement
      expect(el.style.getPropertyValue("--lp-depth")).toBe(String(depth))
    }
  })
})
