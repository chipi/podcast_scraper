import { afterEach, describe, expect, it, vi } from "vitest"
import { holdScroll, keepInPlace, restoreScroll, waitUntilScrollable } from "./scrollRestore"

function scroller(scrollHeight: number, clientHeight = 400): HTMLElement {
  const el = document.createElement("div")
  Object.defineProperty(el, "scrollHeight", { value: scrollHeight, configurable: true, writable: true })
  Object.defineProperty(el, "clientHeight", { value: clientHeight, configurable: true })
  return el
}

afterEach(() => vi.restoreAllMocks())

describe("scrollRestore", () => {
  it("restores at once when the content is already tall enough", async () => {
    const el = scroller(3000)
    await restoreScroll(el, 900)
    expect(el.scrollTop).toBe(900)
  })

  it("waits for content that arrives after mount, instead of clamping to the loading state", async () => {
    // A page re-mounted by Back is a loading line until its fetch lands. Applying the offset then
    // is what sent Back to the top.
    const el = scroller(500)
    setTimeout(() => Object.defineProperty(el, "scrollHeight", { value: 3000 }), 50)
    let settled = false
    const done = waitUntilScrollable(el, 900).then(() => (settled = true))
    await new Promise((r) => setTimeout(r, 10))
    expect(settled).toBe(false)
    await done
    await restoreScroll(el, 900)
    expect(el.scrollTop).toBe(900)
  })

  it("gives up after the timeout so a page that never grows cannot hang Back", async () => {
    const el = scroller(500)
    const started = Date.now()
    await waitUntilScrollable(el, 900, 60)
    expect(Date.now() - started).toBeGreaterThanOrEqual(55)
  })

  it("follows the section it holds when content arriving above pushes it down", async () => {
    const el = scroller(3000)
    let section = 900 // where the held section sits inside the scroller
    el.scrollTop = 900
    holdScroll(el, () => section, 1000)
    await new Promise((r) => setTimeout(r, 40))
    section = 1400 // a rail above loaded; the reader would be left 500px above the section
    await new Promise((r) => setTimeout(r, 60))
    expect(el.scrollTop).toBe(1400)
  })

  it("lets go when something scrolls AWAY on purpose — focus, scrollIntoView — with no input event", async () => {
    const el = scroller(3000)
    el.scrollTop = 900
    holdScroll(el, () => 900, 1000)
    await new Promise((r) => setTimeout(r, 40))
    el.scrollTop = 1600 // height unchanged: not content arriving, a deliberate scroll
    await new Promise((r) => setTimeout(r, 60))
    expect(el.scrollTop).toBe(1600)
  })

  it("lets go the moment the reader scrolls — it never fights them", async () => {
    const el = scroller(3000)
    el.scrollTop = 900
    holdScroll(el, () => 900, 1000)
    window.dispatchEvent(new Event("wheel"))
    el.scrollTop = 1400 // the reader's own scroll
    await new Promise((r) => setTimeout(r, 60))
    expect(el.scrollTop).toBe(1400)
  })

  it("is abandoned when the reader scrolls before the content is ready — a late restore never yanks", async () => {
    const el = scroller(500) // still loading: cannot hold 900 yet
    const done = restoreScroll(el, 900, 400)
    el.scrollTop = 300 // the reader scrolls (programmatically here; a wheel would do the same)
    window.dispatchEvent(new Event("wheel"))
    Object.defineProperty(el, "scrollHeight", { value: 3000 }) // ...and then the content arrives
    await done
    expect(el.scrollTop).toBe(300)
  })

  it("keeps a sheet's opener where it sat when content loads above it while the sheet was open", async () => {
    const box = scroller(3000)
    box.style.overflowY = "auto"
    Object.defineProperty(box, "scrollHeight", { value: 3000, configurable: true })
    document.body.appendChild(box)
    const opener = document.createElement("button")
    box.appendChild(opener)
    let above = 629 // the opener's offset inside the box; rails above it are still loading
    vi.spyOn(box, "getBoundingClientRect").mockReturnValue({ top: 0 } as DOMRect)
    vi.spyOn(opener, "getBoundingClientRect").mockImplementation(() => ({ top: above - box.scrollTop }) as DOMRect)
    keepInPlace(opener, 629) // seen at 629 when the sheet opened
    above = 902 // the rest of the card filled in above it
    await new Promise((r) => setTimeout(r, 60))
    expect(opener.getBoundingClientRect().top).toBe(629)
    box.remove()
  })

  it("does nothing for the top of the page", async () => {
    const scrollTo = vi.spyOn(window, "scrollTo").mockImplementation(() => {})
    await restoreScroll(null, 0)
    expect(scrollTo).not.toHaveBeenCalled()
  })
})
