import { mount } from "@vue/test-utils"
import { flushPromises } from "@vue/test-utils"
import { describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"
import { createRouter, createMemoryHistory } from "vue-router"
import EntityEpisodeList from "./EntityEpisodeList.vue"
import en from "../i18n/locales/en.json"
import type { EpisodeSummary } from "../services/types"

/**
 * The shared "discussed in N episodes" list (operator 2026-09-19).
 *
 * It replaced four uncapped `<ul><li v-for>` blocks across topic / storyline / person / org, so a
 * regression here is a regression on every entity surface at once — and it shipped with no test
 * at all, which is what this file fixes.
 *
 * The cap is the reason the component exists: a topic with sixty episodes emitted sixty rows and
 * pushed the conversation arc, the perspectives block and the notes somewhere no reader reaches.
 */
const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: "/", name: "home", component: { template: "<div/>" } },
    { path: "/episode/:slug", name: "player", component: { template: "<div/>" } },
    { path: "/podcast/:feedId", name: "podcast", component: { template: "<div/>" } },
  ],
})

function ep(i: number): EpisodeSummary {
  return {
    slug: `e${i}`,
    title: `Episode ${i}`,
    feed_id: "f1",
    podcast_title: "A Show",
    publish_date: "2026-01-01",
    duration_seconds: 600,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: null,
    status: "ready",
    summary_preview: null,
    summary_text: null,
    summary_bullets: [],
    topics: [],
    has_transcript: true,
    has_summary: false,
    has_gi: false,
    has_kg: false,
    has_bridge: false,
  }
}

function mountList(count: number) {
  return mount(EntityEpisodeList, {
    props: { episodes: Array.from({ length: count }, (_, i) => ep(i)) },
    global: { plugins: [i18n, router] },
  })
}

const rows = (w: ReturnType<typeof mountList>) => w.findAll('[data-testid="episode-row"]')
const more = (w: ReturnType<typeof mountList>) => w.find('[data-testid="entity-episodes-more"]')

describe("EntityEpisodeList", () => {
  it("shows five rows and offers the rest, rather than rendering the whole list", async () => {
    const w = mountList(25)
    expect(rows(w)).toHaveLength(5)
    expect(more(w).exists()).toBe(true)
  })

  it("reveals five more per press, then stops offering", async () => {
    const w = mountList(13)

    await more(w).trigger("click")
    expect(rows(w)).toHaveLength(10)

    await more(w).trigger("click")
    expect(rows(w)).toHaveLength(13)
    // Everything is out — the control must go, not sit there doing nothing. That is the exact
    // shape of the bug this whole round started from.
    expect(more(w).exists()).toBe(false)
  })

  it("the label counts what the next press will reveal, not what remains", async () => {
    // Recommended shipped saying "Show 8 more" and then revealing four — a control describing
    // someone else's behaviour. The page size, not the remainder, is the honest number.
    const w = mountList(25)
    expect(more(w).text()).toContain("5")

    // The DISCRIMINATING case: 8 items, so 3 are hidden — fewer than a page. Above, "what
    // remains" (20) and "what this press reveals" (5) differ only in that one number, and the
    // regression being guarded ("Show 8 more" revealing four) lives exactly in the gap between
    // them. Read the LABEL, and read it while the control still exists.
    const w2 = mountList(8)
    expect(more(w2).text(), "the label should promise the 3 it can actually reveal").toContain("3")

    await more(w2).trigger("click")
    expect(rows(w2)).toHaveLength(8)
    expect(more(w2).exists(), "nothing left to reveal").toBe(false)
  })

  it("offers nothing when the list already fits", () => {
    const w = mountList(5)
    expect(rows(w)).toHaveLength(5)
    expect(more(w).exists()).toBe(false)
  })

  it("renders nothing rather than throwing on an empty list", () => {
    const w = mountList(0)
    expect(rows(w)).toHaveLength(0)
    expect(more(w).exists()).toBe(false)
  })

  it("re-collapses when it is handed a DIFFERENT entity's episodes", async () => {
    // These surfaces drill in place: opening a sibling topic from a chip hands the SAME component
    // instance a different entity's list. Without the reset you land on the new topic already
    // ten rows deep, having never asked to expand it.
    const w = mountList(25)
    await more(w).trigger("click")
    expect(rows(w)).toHaveLength(10)

    await w.setProps({ episodes: Array.from({ length: 25 }, (_, i) => ep(100 + i)) })
    expect(rows(w)).toHaveLength(5)
    expect(rows(w)[0].text()).toContain("Episode 100")
  })

  it("says nothing on mount, then announces only what a press revealed", async () => {
    // Two failure modes, and the first fix traded one for the other. Without any live region a
    // screen-reader user presses "show more" and hears nothing — indistinguishable from a dead
    // button. But wrapping the LIST in the live region announces all five initial episodes at
    // mount, before the user has done anything, which is worse than the silence it replaced.
    const w = mountList(25)
    const status = w.find('[aria-live="polite"]')
    expect(status.exists()).toBe(true)
    expect(status.classes()).toContain("sr-only")
    expect(status.text(), "it spoke before the user touched anything").toBe("")

    // The list itself must NOT be the live region.
    expect(w.find("ul").attributes("aria-live")).toBeUndefined()

    await more(w).trigger("click")
    expect(status.text()).toContain("5")
  })

  it("stops announcing the previous entity when it is handed a new one", async () => {
    const w = mountList(25)
    await more(w).trigger("click")
    expect(w.find('[aria-live="polite"]').text()).not.toBe("")

    await w.setProps({ episodes: Array.from({ length: 25 }, (_, i) => ep(100 + i)) })
    // A stale "5 more episodes shown" hanging over a list that just collapsed is a lie.
    expect(w.find('[aria-live="polite"]').text()).toBe("")
  })
})

describe("EntityEpisodeList paged on the server (2026-10-08)", () => {
  it("shows the first page, offers the TOTAL, and fetches the next page on Show more", async () => {
    // A storyline returned 99 episodes (1.29 MB on prod) to show five; now it returns five.
    const loadMore = vi.fn(async (offset: number, limit: number) =>
      Array.from({ length: Math.min(limit, 12 - offset) }, (_, i) => ep(offset + i))
    )
    const w = mount(EntityEpisodeList, {
      props: { episodes: Array.from({ length: 5 }, (_, i) => ep(i)), total: 12, loadMore },
      global: { plugins: [i18n, router] },
    })
    expect(rows(w)).toHaveLength(5)
    expect(w.get('[data-testid="entity-episodes-more"]').text()).toContain("5")
    await w.get('[data-testid="entity-episodes-more"]').trigger("click")
    await flushPromises()
    expect(loadMore).toHaveBeenCalledWith(5, 5)
    expect(rows(w)).toHaveLength(10)
    await w.get('[data-testid="entity-episodes-more"]').trigger("click")
    await flushPromises()
    expect(loadMore).toHaveBeenLastCalledWith(10, 5)
    expect(rows(w)).toHaveLength(12)
    expect(w.find('[data-testid="entity-episodes-more"]').exists()).toBe(false)
  })

  it("without loadMore it pages what it was given, as before", async () => {
    const w = mountList(7)
    expect(rows(w)).toHaveLength(5)
    await w.get('[data-testid="entity-episodes-more"]').trigger("click")
    expect(rows(w)).toHaveLength(7)
  })
})

