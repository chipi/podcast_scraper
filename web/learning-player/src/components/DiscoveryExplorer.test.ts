import { flushPromises, mount } from "@vue/test-utils"
import { createPinia, setActivePinia } from "pinia"
import { afterEach, describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"
import { createMemoryHistory, createRouter } from "vue-router"
import * as api from "../services/api"
import en from "../i18n/locales/en.json"
import DiscoveryExplorer from "./DiscoveryExplorer.vue"
import type { TrendingEntity } from "../services/types"

/**
 * The Discover "all ›" control (operator 2026-09-19).
 *
 * It was a `RouterLink` to `{ name: 'browse', query: { trends: kind } }`. The only surface that
 * renders this header IS Browse — "Discover" in the tab bar is the `browse` route — so it pointed
 * at the page you were already on, carrying the kind you were already reading. Vue Router
 * navigated, the query changed, and nothing moved: "I click all, and nothing really happens".
 *
 * These pin the three things that make the replacement not-that: it expands IN PLACE, it appears
 * only when rows are genuinely hidden, and it collapses when you switch kind.
 */
const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: "/", name: "home", component: { template: "<div/>" } },
    { path: "/browse", name: "browse", component: { template: "<div/>" } },
  ],
})

function trending(n: number, prefix = "topic"): TrendingEntity[] {
  return Array.from({ length: n }, (_, i) => ({
    entity_id: `${prefix}:${i}`,
    kind: prefix,
    label: `${prefix} ${i}`,
    velocity: 2 + i / 100,
    volume: 10,
    heating_up: true,
    total: 10,
    series: [1, 2, 3],
  }))
}

function mountExplorer() {
  setActivePinia(createPinia())
  return mount(DiscoveryExplorer, {
    props: { collapsed: 10, seeAll: true, title: "Trends" },
    global: { plugins: [i18n, router, createPinia()] },
  })
}

const rows = (w: ReturnType<typeof mountExplorer>) => w.findAll('[data-testid="discovery-row"]')
const allBtn = (w: ReturnType<typeof mountExplorer>) => w.find('[data-testid="discovery-see-all"]')

afterEach(() => vi.restoreAllMocks())

describe("DiscoveryExplorer — the Trends expand control", () => {
  it("expands the list IN PLACE rather than navigating anywhere", async () => {
    vi.spyOn(api, "getTrending").mockResolvedValue(trending(18))
    vi.spyOn(api, "getStorylines").mockResolvedValue([])
    const push = vi.spyOn(router, "push")
    const w = mountExplorer()
    await flushPromises()

    expect(rows(w)).toHaveLength(10)
    await allBtn(w).trigger("click")
    await flushPromises()

    expect(rows(w)).toHaveLength(18)
    // The whole point: it is not a link. A navigation here is the bug coming back.
    expect(push).not.toHaveBeenCalled()
  })

  it("is a button, and reports its expanded state to assistive tech", async () => {
    vi.spyOn(api, "getTrending").mockResolvedValue(trending(18))
    vi.spyOn(api, "getStorylines").mockResolvedValue([])
    const w = mountExplorer()
    await flushPromises()

    const btn = allBtn(w)
    expect(btn.element.tagName).toBe("BUTTON")
    expect(btn.attributes("aria-expanded")).toBe("false")
    // And it names the region it controls — otherwise it announces a state with no way to reach it.
    const controls = btn.attributes("aria-controls")
    expect(controls).toBeTruthy()
    expect(w.find(`#${controls}`).exists()).toBe(true)

    await btn.trigger("click")
    expect(allBtn(w).attributes("aria-expanded")).toBe("true")
  })

  it("does not render when the list already fits — a control that does nothing IS the bug", async () => {
    vi.spyOn(api, "getTrending").mockResolvedValue(trending(6))
    vi.spyOn(api, "getStorylines").mockResolvedValue([])
    const w = mountExplorer()
    await flushPromises()

    expect(rows(w)).toHaveLength(6)
    expect(allBtn(w).exists()).toBe(false)
  })

  it("collapses again when you switch kind", async () => {
    vi.spyOn(api, "getTrending").mockImplementation(async (kind: string) =>
      trending(18, kind === "person" ? "person" : "topic"),
    )
    vi.spyOn(api, "getStorylines").mockResolvedValue([])
    const w = mountExplorer()
    await flushPromises()

    await allBtn(w).trigger("click")
    await flushPromises()
    expect(rows(w)).toHaveLength(18)

    await w.find('[data-testid="discovery-tab-person"]').trigger("click")
    await flushPromises()
    // Landing on a 30-row list you never asked to open would be its own surprise, and the label
    // would be describing a state you did not choose.
    expect(rows(w)).toHaveLength(10)
    expect(allBtn(w).attributes("aria-expanded")).toBe("false")
  })
})
