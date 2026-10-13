import { flushPromises, mount } from "@vue/test-utils"
import { keptAlive } from "../test/keptAlive"
import { createPinia, setActivePinia } from "pinia"
import { afterEach, describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"
import { createMemoryHistory, createRouter } from "vue-router"
import * as api from "../services/api"
import en from "../i18n/locales/en.json"
import DiscoveryList from "./DiscoveryList.vue"
import * as analytics from "../services/analytics"
import type { Storyline, TrendingEntity } from "../services/types"

/**
 * Storyline rows and what happens when one cannot be opened (operator 2026-09-19).
 *
 * A storyline has no endpoint of its own — it is read as its most-central member topic's card — so
 * a row without an anchor has nothing to open. The original bug was that such rows LOOKED
 * openable and did nothing: the client fell back to the `thc:` id, which resolves no topic.
 *
 * The fix has now been wrong twice, which is why this file exists:
 *   1. `?? entity_id` — the dead tap.
 *   2. `disabled` — dropped the row out of the tab order, so keyboard users lost it entirely.
 *   3. `aria-disabled` alone — the row still lit up on hover, promising a target it did not have.
 */
const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: "/", name: "home", component: { template: "<div/>" } },
    { path: "/browse", name: "browse", component: { template: "<div/>" } },
    { path: "/explore", name: "explore", component: { template: "<div/>" } },
  ],
})

function row(id: string, anchor: string | null): TrendingEntity {
  return {
    entity_id: id,
    kind: "storyline",
    label: `Story ${id}`,
    velocity: 2,
    volume: 10,
    heating_up: true,
    total: 10,
    series: [1, 2, 3],
    anchor_topic_id: anchor,
  }
}

function page(rows: TrendingEntity[], total = rows.length): api.TrendingPage {
  return { items: rows, total }
}

function mountList(rows: TrendingEntity[], stories: Storyline[] = []) {
  setActivePinia(createPinia())
  vi.spyOn(api, "getTrendingPage").mockResolvedValue(page(rows))
  vi.spyOn(api, "getStorylines").mockResolvedValue(stories)
  return mount(DiscoveryList, {
    props: { kind: "storyline" as const, sort: "rising" as const },
    global: { plugins: [i18n, router, createPinia()] },
  })
}

afterEach(() => vi.restoreAllMocks())

describe("DiscoveryList — a storyline row that cannot be opened", () => {
  it("opens with the ANCHOR topic id, never the thc: id", async () => {
    const w = mountList([row("thc:ai-safety", "topic:ai")])
    await flushPromises()

    await w.find('[data-testid="discovery-row"] button').trigger("click")
    // Handing a `thc:` id to a consumer that resolves a TOPIC is what made the tap dead.
    // `rank` is the 1-based position tapped (#2267): only this list knows it.
    expect(w.emitted("open")?.[0]).toEqual([{ kind: "storyline", id: "topic:ai", rank: 1 }])
  })

  it("emits nothing at all when there is no anchor", async () => {
    const w = mountList([row("thc:orphan", null)])
    await flushPromises()

    await w.find('[data-testid="discovery-row"] button').trigger("click")
    expect(w.emitted("open")).toBeUndefined()
  })

  it("looks inert BEFORE it is tapped, not after", async () => {
    // The whole complaint was "I click and nothing happens". A row that dims only on tap has not
    // fixed that; it has to read as non-interactive on sight.
    const w = mountList([row("thc:orphan", null), row("thc:fine", "topic:ok")])
    await flushPromises()

    const [inert, fine] = w.findAll('[data-testid="discovery-row"]')
    expect(inert.find("button").classes()).toContain("opacity-60")
    expect(inert.find("button").classes()).toContain("cursor-default")
    // And the ROW does not light up under the cursor, which would promise a target.
    expect(inert.classes()).not.toContain("hover:bg-overlay")
    expect(fine.classes()).toContain("hover:bg-overlay")
  })

  it("stays reachable by keyboard and says why it does nothing", async () => {
    // `disabled` would drop it out of the tab order — the same silence, moved to keyboard users.
    const w = mountList([row("thc:orphan", null)])
    await flushPromises()

    const btn = w.find('[data-testid="discovery-row"] button')
    expect(btn.attributes("disabled")).toBeUndefined()
    expect(btn.attributes("aria-disabled")).toBe("true")
    expect(btn.attributes("aria-label")).toContain("nothing to open")
  })

  it("does not announce not-disabled-ness on every working row", async () => {
    const w = mountList([row("thc:fine", "topic:ok")])
    await flushPromises()

    const btn = w.find('[data-testid="discovery-row"] button')
    expect(btn.attributes("aria-disabled")).toBeUndefined()
  })

  it("a non-storyline kind opens on its own id and is never inert", async () => {
    setActivePinia(createPinia())
    vi.spyOn(api, "getTrendingPage").mockResolvedValue(
      page([{ ...row("topic:ai", null), kind: "topic", label: "AI" }])
    )
    vi.spyOn(api, "getStorylines").mockResolvedValue([])
    const w = mount(DiscoveryList, {
      props: { kind: "topic" as const, sort: "rising" as const },
      global: { plugins: [i18n, router, createPinia()] },
    })
    await flushPromises()

    // `anchor_topic_id` is null for every non-storyline kind by design — it must not make those
    // rows inert, which is the obvious way this guard could over-reach.
    const btn = w.find('[data-testid="discovery-row"] button')
    expect(btn.attributes("aria-disabled")).toBeUndefined()
    await btn.trigger("click")
    expect(w.emitted("open")?.[0]).toEqual([{ kind: "topic", id: "topic:ai", rank: 1 }])
  })
})

describe("DiscoveryList — an empty Mine says why and offers everyone (operator 2026-10-07)", () => {
  afterEach(() => vi.restoreAllMocks())

  function mountScoped(rows: TrendingEntity[], scope: "mine" | "corpus") {
    vi.spyOn(api, "getTrendingPage").mockResolvedValue(page(rows))
    vi.spyOn(api, "getStorylines").mockResolvedValue([])
    return mount(DiscoveryList, {
      props: { kind: "topic", sort: "rising", scope },
      global: { plugins: [i18n, router] },
    })
  }

  it("explains an empty Mine and asks to show everyone's trends", async () => {
    const w = mountScoped([], "mine")
    await flushPromises()
    expect(w.get('[data-testid="discovery-mine-empty"]').text()).toContain("as you listen, save and follow")
    await w.get('[data-testid="discovery-show-everyone"]').trigger("click")
    expect(w.emitted("show-everyone")).toHaveLength(1)
  })

  it("says nothing of the sort for an empty everyone list, or a Mine with rows", async () => {
    const empty = mountScoped([], "corpus")
    await flushPromises()
    expect(empty.find('[data-testid="discovery-mine-empty"]').exists()).toBe(false)
    const full = mountScoped([row("topic:a", null)], "mine")
    await flushPromises()
    expect(full.find('[data-testid="discovery-mine-empty"]').exists()).toBe(false)
  })
})

describe("DiscoveryList — a return to Discover re-reads YOUR trends (2026-10-09)", () => {
  async function cycle(scope: "mine" | "corpus", late: boolean) {
    setActivePinia(createPinia())
    const spy = vi.spyOn(api, "getTrendingPage").mockResolvedValue(page([]))
    vi.spyOn(api, "getStorylines").mockResolvedValue([])
    const tab = await keptAlive(DiscoveryList, {
      plugins: [i18n, router, createPinia()],
      props: { kind: "topic", sort: "rising", scope },
      late,
    })
    const first = spy.mock.calls.length
    await tab.leave()
    await tab.back()
    return spy.mock.calls.length - first
  }

  // `late`: the list mounts after Discover is already showing (it renders further down the page),
  // so it gets no activation for its mount — the case the first version of this fix missed.
  it.each([false, true])("mine: re-asks on return (late=%s)", async (late) => {
    expect(await cycle("mine", late)).toBeGreaterThan(0)
  })

  it.each([false, true])("everyone: does not — nothing you did changes it (late=%s)", async (late) => {
    expect(await cycle("corpus", late)).toBe(0)
  })
})

describe("DiscoveryList — paged on the server (operator 2026-10-11)", () => {
  afterEach(() => vi.restoreAllMocks())

  const topic = (i: number): TrendingEntity => ({ ...row(`topic:t${i}`, null), kind: "topic", label: `T${i}` })

  it("asks for the first page in the chosen order, then the next page on Show more, then back", async () => {
    setActivePinia(createPinia())
    const trackSpy = vi.spyOn(analytics, "track")
    const spy = vi
      .spyOn(api, "getTrendingPage")
      .mockResolvedValueOnce(page([topic(1), topic(2), topic(3)], 9))
      .mockResolvedValueOnce(page([topic(4), topic(5), topic(6), topic(7), topic(8)], 9))
    const w = mount(DiscoveryList, {
      props: { kind: "topic" as const, sort: "trending" as const, collapsed: 3 },
      global: { plugins: [i18n, router, createPinia()] },
    })
    await flushPromises()
    expect(spy).toHaveBeenLastCalledWith("topic", { scope: "corpus", window: "3m", sort: "trending", offset: 0, limit: 3 })
    expect(w.findAll('[data-testid="discovery-row"]')).toHaveLength(3)
    const more = w.get('[data-testid="show-all-toggle"]')
    expect(more.text()).toBe("Show more (6)")

    await more.trigger("click")
    await flushPromises()
    expect(spy).toHaveBeenLastCalledWith("topic", { scope: "corpus", window: "3m", sort: "trending", offset: 3, limit: 5 })
    expect(trackSpy).toHaveBeenCalledWith("trends_show_more", { kind: "topic", sort: "trending", shown: "2-5" })
    expect(w.findAll('[data-testid="discovery-row"]')).toHaveLength(8)
    expect(w.get('[data-testid="show-all-toggle"]').text()).toBe("Show more (1)")
  })

  it("a late Mine answer never pairs Everyone's rows with Mine's count", async () => {
    // Seen in e2e: switching Mine off asked twice; Mine's 0-row answer landed second and left
    // Everyone's rows on screen with total 0 — no "Show more".
    setActivePinia(createPinia())
    let releaseMine: (p: api.TrendingPage) => void = () => {}
    vi.spyOn(api, "getTrendingPage").mockImplementation((_kind, q) =>
      q.scope === "mine"
        ? new Promise((r) => (releaseMine = r))
        : Promise.resolve(page([topic(1), topic(2), topic(3)], 9))
    )
    const w = mount(DiscoveryList, {
      props: { kind: "topic" as const, sort: "rising" as const, scope: "mine" as const, collapsed: 3 },
      global: { plugins: [i18n, router, createPinia()] },
    })
    await w.setProps({ scope: "corpus" })
    await flushPromises()
    releaseMine(page([], 0))
    await flushPromises()
    expect(w.findAll('[data-testid="discovery-row"]')).toHaveLength(3)
    expect(w.get('[data-testid="show-all-toggle"]').text()).toBe("Show more (6)")
  })

  it("re-asks the server when the sort changes, rather than re-sorting what it holds", async () => {
    setActivePinia(createPinia())
    const spy = vi.spyOn(api, "getTrendingPage").mockResolvedValue(page([topic(1)], 1))
    const w = mount(DiscoveryList, {
      props: { kind: "topic" as const, sort: "rising" as const, collapsed: 3 },
      global: { plugins: [i18n, router, createPinia()] },
    })
    await flushPromises()
    await w.setProps({ sort: "trending" })
    await flushPromises()
    expect(spy).toHaveBeenLastCalledWith("topic", expect.objectContaining({ sort: "trending", offset: 0 }))
    expect(w.find('[data-testid="show-all-toggle"]').exists()).toBe(false) // everything fits
  })
})

describe("DiscoveryList — a theme row says how many topics it holds (2026-10-12)", () => {
  afterEach(() => vi.restoreAllMocks())

  it("reads the size off the trending row, so a narrow theme has one too", async () => {
    setActivePinia(createPinia())
    const theme: TrendingEntity = { ...row("tc:code-switching", null), kind: "theme", label: "code switching", size: 4 }
    vi.spyOn(api, "getTrendingPage").mockResolvedValue(page([theme]))
    const clusters = vi.spyOn(api, "getTopClusters")
    const w = mount(DiscoveryList, {
      props: { kind: "theme" as const, sort: "rising" as const },
      global: { plugins: [i18n, router, createPinia()] },
    })
    await flushPromises()
    const label = w.find('[data-testid="discovery-row"] button').attributes("aria-label")
    expect(label).toContain("code switching (4)")
    expect(clusters, "the 50 widest themes are not where a row's size lives").not.toHaveBeenCalled()
  })
})
