import { mount } from "@vue/test-utils"
import { afterEach, describe, expect, it } from "vitest"

import ToolbarMenu from "./ToolbarMenu.vue"

const OPTIONS = [
  { value: "az", label: "A–Z" },
  { value: "episodes", label: "Most" },
]
const mountMenu = (props = {}) =>
  mount(ToolbarMenu, {
    props: { options: OPTIONS, menuLabel: "Sort", modelValue: "az", testid: "sort", ...props },
    attachTo: document.body,
  })

afterEach(() => {
  document.body.innerHTML = ""
})

describe("ToolbarMenu", () => {
  it("keeps the options menu closed until the trigger is tapped", async () => {
    const w = mountMenu()
    expect(w.get('[data-testid="sort"]').attributes("aria-expanded")).toBe("false")
    expect(w.find('[data-testid="sort-opt-az"]').exists()).toBe(false)
    await w.get('[data-testid="sort"]').trigger("click")
    expect(w.get('[data-testid="sort"]').attributes("aria-expanded")).toBe("true")
    expect(w.get('[data-testid="sort-opt-az"]').exists()).toBe(true)
    expect(w.get('[data-testid="sort-opt-episodes"]').exists()).toBe(true)
  })

  it("marks the active option and emits + closes on pick", async () => {
    const w = mountMenu()
    await w.get('[data-testid="sort"]').trigger("click")
    // Active option carries aria-checked.
    expect(w.get('[data-testid="sort-opt-az"]').attributes("aria-checked")).toBe("true")
    expect(w.get('[data-testid="sort-opt-episodes"]').attributes("aria-checked")).toBe("false")
    await w.get('[data-testid="sort-opt-episodes"]').trigger("click")
    expect(w.emitted("update:modelValue")?.at(-1)).toEqual(["episodes"])
    // Menu closes after a pick.
    expect(w.find('[data-testid="sort-opt-az"]').exists()).toBe(false)
  })

  it("pill variant shows the CURRENT option's label (not the widest)", () => {
    const w = mountMenu({ variant: "pill", modelValue: "episodes" })
    expect(w.get('[data-testid="sort"]').text()).toContain("Most")
    expect(w.get('[data-testid="sort"]').text()).not.toContain("A–Z")
  })

  it("closes on Escape", async () => {
    const w = mountMenu()
    await w.get('[data-testid="sort"]').trigger("click")
    expect(w.find('[data-testid="sort-opt-az"]').exists()).toBe(true)
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    await w.vm.$nextTick()
    expect(w.find('[data-testid="sort-opt-az"]').exists()).toBe(false)
  })
})
