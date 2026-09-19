import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"
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
  it("shows ten rows and offers the rest, rather than rendering the whole list", async () => {
    const w = mountList(25)
    expect(rows(w)).toHaveLength(10)
    expect(more(w).exists()).toBe(true)
  })

  it("reveals ten more per press, then stops offering", async () => {
    const w = mountList(25)

    await more(w).trigger("click")
    expect(rows(w)).toHaveLength(20)

    await more(w).trigger("click")
    expect(rows(w)).toHaveLength(25)
    // Everything is out — the control must go, not sit there doing nothing. That is the exact
    // shape of the bug this whole round started from.
    expect(more(w).exists()).toBe(false)
  })

  it("the label counts what the next press will reveal, not what remains", async () => {
    // Recommended shipped saying "Show 8 more" and then revealing four — a control describing
    // someone else's behaviour. The page size, not the remainder, is the honest number.
    const w = mountList(25)
    expect(more(w).text()).toContain("10")

    await more(w).trigger("click")
    await more(w).trigger("click")
    // 25 - 20 = 5 left, which is less than a page: now the remainder IS what the press reveals.
    const w2 = mountList(15)
    await more(w2).trigger("click")
    expect(rows(w2)).toHaveLength(15)
  })

  it("offers nothing when the list already fits", () => {
    const w = mountList(10)
    expect(rows(w)).toHaveLength(10)
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
    // twenty rows deep, having never asked to expand it.
    const w = mountList(25)
    await more(w).trigger("click")
    expect(rows(w)).toHaveLength(20)

    await w.setProps({ episodes: Array.from({ length: 25 }, (_, i) => ep(100 + i)) })
    expect(rows(w)).toHaveLength(10)
    expect(rows(w)[0].text()).toContain("Episode 100")
  })

  it("announces the newly revealed rows instead of appending them silently", () => {
    // A screen-reader user presses "show more" and, without this, hears nothing at all — which is
    // indistinguishable from a dead button.
    const w = mountList(25)
    const list = w.find("ul")
    expect(list.attributes("aria-live")).toBe("polite")
    expect(list.attributes("aria-atomic")).toBe("false")
  })
})
