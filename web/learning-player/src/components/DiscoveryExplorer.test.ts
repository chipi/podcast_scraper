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
 * Trends on Explore: one tabbed list over four kinds, its first page and then "Show more (N)", paged
 * from the server (operator 2026-10-11). The old header "all ›" revealed rows already loaded; it is
 * gone, and nothing in the header expands or navigates.
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

function page(rows: TrendingEntity[], total = rows.length): api.TrendingPage {
  return { items: rows, total }
}

function mountExplorer() {
  setActivePinia(createPinia())
  return mount(DiscoveryExplorer, {
    props: { collapsed: 3, title: "Trends" },
    global: { plugins: [i18n, router, createPinia()] },
  })
}

const rows = (w: ReturnType<typeof mountExplorer>) => w.findAll('[data-testid="discovery-row"]')

afterEach(() => vi.restoreAllMocks())

describe("DiscoveryExplorer — a taste, paged from the server", () => {
  it("shows the first page, has no header expand control, and pages on with Show more", async () => {
    const spy = vi
      .spyOn(api, "getTrendingPage")
      .mockResolvedValueOnce(page(trending(3), 18))
      .mockResolvedValueOnce(page(trending(5).map((r, i) => ({ ...r, entity_id: `topic:m${i}` })), 18))
    vi.spyOn(api, "getStorylines").mockResolvedValue([])
    const push = vi.spyOn(router, "push")
    const w = mountExplorer()
    await flushPromises()

    expect(rows(w)).toHaveLength(3)
    expect(w.find('[data-testid="discovery-see-all"]').exists()).toBe(false)
    await w.get('[data-testid="show-all-toggle"]').trigger("click")
    await flushPromises()
    expect(rows(w)).toHaveLength(8)
    expect(spy).toHaveBeenLastCalledWith("topic", expect.objectContaining({ offset: 3, limit: 5 }))
    expect(push).not.toHaveBeenCalled()
  })

  it("a kind switch starts that kind at its first page", async () => {
    const spy = vi
      .spyOn(api, "getTrendingPage")
      .mockImplementation(async (kind: string) => page(trending(3, kind), 18))
    vi.spyOn(api, "getStorylines").mockResolvedValue([])
    const w = mountExplorer()
    await flushPromises()
    await w.find('[data-testid="discovery-tab-person"]').trigger("click")
    await flushPromises()
    expect(spy).toHaveBeenLastCalledWith("person", expect.objectContaining({ offset: 0, limit: 3 }))
    expect(rows(w)).toHaveLength(3)
  })
})

describe("DiscoveryExplorer — the icon toggles say what they are set to (operator 2026-10-08)", () => {
  it("keeps the icons on the pills' row and states sort (and scope, signed in) in a caption", async () => {
    vi.spyOn(api, "getTrendingPage").mockResolvedValue(page(trending(3)))
    vi.spyOn(api, "getStorylines").mockResolvedValue([])
    const w = mountExplorer()
    await flushPromises()
    const hint = () => w.get('[data-testid="discovery-state-hint"]').text()
    // Signed out: only the sort applies.
    expect(hint()).toBe("rising fastest")
    await w.get('[data-testid="discovery-sort"]').trigger("click")
    expect(hint()).toBe("most talked about")
    // The toggle is an icon button again, not a labelled two-option switch.
    expect(w.find('[data-testid="discovery-sort-trending"]').exists()).toBe(false)
  })
})
