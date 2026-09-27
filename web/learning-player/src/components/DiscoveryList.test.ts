import { flushPromises, mount } from "@vue/test-utils"
import { createPinia, setActivePinia } from "pinia"
import { afterEach, describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"
import { createMemoryHistory, createRouter } from "vue-router"
import * as api from "../services/api"
import en from "../i18n/locales/en.json"
import DiscoveryList from "./DiscoveryList.vue"
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

function mountList(rows: TrendingEntity[], stories: Storyline[] = []) {
  setActivePinia(createPinia())
  vi.spyOn(api, "getTrending").mockResolvedValue(rows)
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
    expect(w.emitted("open")?.[0]).toEqual([{ kind: "storyline", id: "topic:ai" }])
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
    vi.spyOn(api, "getTrending").mockResolvedValue([
      { ...row("topic:ai", null), kind: "topic", label: "AI" },
    ])
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
    expect(w.emitted("open")?.[0]).toEqual([{ kind: "topic", id: "topic:ai" }])
  })
})
