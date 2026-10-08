import { mount } from "@vue/test-utils"
import { createPinia, setActivePinia } from "pinia"
import { beforeEach, describe, expect, it, vi } from "vitest"

// `DownloadButton` is `v-if="native"`, so off-native it renders nothing and every assertion about
// where it SITS would pass vacuously. Mocked true for this file so the row composition is testable
// at all — the browser-tier e2e cannot cover it for exactly the same reason.
vi.mock("../services/native", async (orig) => ({
  ...(await orig<typeof import("../services/native")>()),
  isNative: () => true,
}))
import { createI18n } from "vue-i18n"
import { createRouter, createMemoryHistory } from "vue-router"
import en from "../i18n/locales/en.json"
import type { EpisodeSummary } from "../services/types"
import EpisodeCard from "./EpisodeCard.vue"
import { flushPromises } from "@vue/test-utils"
import * as api from "../services/api"
import { resetCorpusLanguagesForTests } from "../composables/useCorpusLanguages"

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })

/** aria-expanded controls that belong to the CARD itself — excludes the ⋯ overflow trigger and the
 *  add-to-collection trigger (now inside that ⋯), whose aria-expanded is correct popup semantics,
 *  not a summary/insights expander. */
function cardOwnExpanders(w: ReturnType<typeof mountCard>) {
  return w
    .findAll("[aria-expanded]")
    .filter((el) => !["add-to-collection", "overflow-trigger"].includes(el.attributes("data-testid") ?? ""))
}

beforeEach(() => {
  // Fresh pinia per test; auth defaults to signed-out → no queue button.
  setActivePinia(createPinia())
})
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: "/", name: "catalog", component: { template: "<div/>" } },
    { path: "/podcast/:feedId", name: "podcast", component: { template: "<div/>" } },
    { path: "/episode/:slug", name: "player", component: { template: "<div/>" } },
  ],
})

function makeEpisode(over: Partial<EpisodeSummary> = {}): EpisodeSummary {
  return {
    slug: "show-abc123",
    title: "A Great Episode",
    feed_id: "show",
    podcast_title: "The Show",
    publish_date: "2024-03-10",
    duration_seconds: 2880,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: null,
    status: "ready",
    summary_preview: "A crisp recap.",
    summary_text: null,
    description: "The show's own words about this episode.",
    summary_bullets: ["Sleep clears metabolic waste.", "Deep sleep consolidates memory."],
    topics: ["memory", "sleep"],
    has_transcript: true,
    has_summary: true,
    has_gi: true,
    has_kg: true,
    has_bridge: false,
    ...over,
  }
}

function mountCard(ep: EpisodeSummary) {
  return mount(EpisodeCard, { props: { episode: ep }, global: { plugins: [i18n, router] } })
}

describe("EpisodeCard — language badge container (V2-C.1)", () => {
  // The facts row under the artwork used to render whenever the episode HAD a language, even when
  // the badge inside it was hidden (single-language corpus) — an empty row with nothing in it.
  const facts = (w: ReturnType<typeof mountCard>) =>
    w.findAll("div").filter((d) => d.classes().includes("gap-1.5") && d.classes().includes("text-muted"))
  const bare = { publish_date: null, duration_seconds: null, language: "en" }

  it("renders no facts row when the badge would be its only content and is hidden", async () => {
    resetCorpusLanguagesForTests()
    vi.spyOn(api, "getPodcasts").mockResolvedValue([{ feed_id: "a", language: "en" }] as never)
    const w = mountCard(makeEpisode(bare))
    await flushPromises()
    expect(w.find('[data-testid="language-badge"]').exists()).toBe(false)
    expect(facts(w)).toHaveLength(0)
  })

  it("renders the row with the badge in a multilingual corpus", async () => {
    resetCorpusLanguagesForTests()
    vi.spyOn(api, "getPodcasts").mockResolvedValue([
      { feed_id: "a", language: "en" },
      { feed_id: "b", language: "es" },
    ] as never)
    const w = mountCard(makeEpisode(bare))
    await flushPromises()
    expect(facts(w)).toHaveLength(1)
    expect(facts(w)[0]!.find('[data-testid="language-badge"]').attributes("data-lang")).toBe("en")
  })
})

describe("EpisodeCard", () => {
  it("renders title, podcast, the publisher's description and duration", () => {
    const w = mountCard(makeEpisode())
    expect(w.text()).toContain("A Great Episode")
    expect(w.text()).toContain("The Show")
    expect(w.text()).toContain("The show's own words about this episode.")
    expect(w.text()).toContain("48 min")
  })

  it("links to the player and to the podcast view", () => {
    const w = mountCard(makeEpisode())
    const hrefs = w.findAll("a").map((a) => a.attributes("href"))
    expect(hrefs).toContain("/episode/show-abc123")
    expect(hrefs).toContain("/podcast/show")
  })

  it("compact still renders the shared action row, capped to the artwork width", () => {
    // Regression: a `v-if="!compact"` once dropped ALL actions from the queue's "recently played"
    // cards. Compact must still carry EpisodeActions, capped to the 80px artwork width (`w-20`) so
    // the ⋯ folds under favourite+queue rather than widening the column past the artwork.
    const w = mount(EpisodeCard, {
      props: { episode: makeEpisode(), compact: true },
      global: { plugins: [i18n, router] },
    })
    const actions = w.find('[data-testid="episode-actions"]')
    expect(actions.exists()).toBe(true)
    expect(actions.classes()).toContain("w-20")
    expect(actions.classes()).not.toContain("w-32")
  })

  it("degrades cleanly when enrichment is absent", () => {
    const w = mountCard(
      makeEpisode({
        summary_preview: null,
        summary_bullets: [],
        topics: [],
        has_gi: false,
        duration_seconds: null,
        podcast_title: null,
      })
    )
    expect(w.text()).toContain("A Great Episode")
    // No insights affordance without grounded summary bullets.
    expect(w.find('[role="dialog"]').exists()).toBe(false)
    expect(w.text()).not.toContain("min")
    // No podcast link when the title is absent.
    expect(w.findAll("a").map((a) => a.attributes("href"))).not.toContain("/podcast/show")
  })

  it("shows pending status when not ready", () => {
    const w = mountCard(makeEpisode({ status: "pending" }))
    expect(w.text()).toContain("Pending")
  })

  it("prefers local artwork_url over the remote image URLs", () => {
    const w = mountCard(
      makeEpisode({
        artwork_url: "/api/app/artwork?ref=x&size=thumb",
        episode_image_url: "https://remote/ep.jpg",
        feed_image_url: "https://remote/feed.jpg",
      })
    )
    expect(w.find("img").attributes("src")).toBe("/api/app/artwork?ref=x&size=thumb")
  })

  it("falls back to the remote image URL when no local artwork", () => {
    const w = mountCard(
      makeEpisode({ artwork_url: null, feed_image_url: "https://remote/feed.jpg" })
    )
    expect(w.find("img").attributes("src")).toBe("https://remote/feed.jpg")
  })

  // --- insights disclosure (#1583) ---
  //
  // These replace the tests that pinned the whole-card hover overlay and the sparkle popover. Both
  // mechanisms were deleted; see the component docblock for why. The assertions below encode the
  // properties that made them wrong, so a reintroduction fails here.

  it("has no expand toggle left behind", () => {
    // The count moved to the artwork column and became a label. A leftover `aria-expanded` control
    // that expands something already visible is worse than none. The add-to-collection menu trigger
    // legitimately carries aria-expanded (it opens a popup), so it is excluded — this asserts the
    // CARD's own summary/insights expander is gone.
    //
    // NO summary of either kind: the read-more toggle also carries `aria-expanded` and is legitimate
    // when there IS prose to reveal, so a card with prose cannot answer this question. jsdom has no
    // layout, so the clamp holds its safe "might be clipped" default and any card with a summary
    // offers the toggle — correctly.
    const w = mountCard(makeEpisode({ description: null } as never))
    expect(cardOwnExpanders(w)).toHaveLength(0)
  })

  it("renders the full description clamped, expandable via Read more (BE.2)", () => {
    // Compact by default (the artwork column sets the window), full on an explicit tap — so
    // unbounded prose never slices a fixed-height box.
    const w = mountCard(
      makeEpisode({ description: "A very long unbounded publisher blurb.".repeat(20) })
    )
    const toggle = w.get('[data-testid="card-read-more"]')
    expect(toggle.text()).toBe("Read more")
    expect(w.text()).toContain("A very long unbounded publisher blurb.") // present, clamped by CSS
  })

  it("shows the publisher's description, never our summary (operator 2026-10-05)", () => {
    // A list card says what the episode IS in the show's words, as the show row does; our summary
    // lives in the episode notes. With a description present, neither summary field appears.
    const w = mountCard(
      makeEpisode({ summary_text: "Our full summary prose.", summary_preview: "Our lede." })
    )
    expect(w.text()).toContain("The show's own words about this episode.")
    expect(w.text()).not.toContain("Our full summary prose.")
    expect(w.text()).not.toContain("Our lede.")
  })

  it("falls back to NOTHING, not our summary, when an episode has no description", () => {
    // No fallback: the card's prose slot is the publisher's, so an episode without a description
    // shows no prose — and therefore no Read more — rather than quietly showing ours instead.
    const w = mountCard(
      makeEpisode({
        description: null,
        summary_text: "Our full summary prose.",
        summary_preview: "Our lede.",
      } as never)
    )
    expect(w.text()).not.toContain("Our full summary prose.")
    expect(w.text()).not.toContain("Our lede.")
    expect(w.find('[data-testid="card-read-more"]').exists()).toBe(false)
  })

  it("has no hover-triggered reveal anywhere on the card", () => {
    // group-hover is not a gesture on touch, and with no hover intent it strobed every card as the
    // pointer passed down a list.
    expect(mountCard(makeEpisode()).html()).not.toContain("group-hover:opacity")
  })

  it("omits the insights affordance when there are no grounded bullets", () => {
    // Summaries nulled for the same reason as above: the read-more toggle is a legitimate
    // `aria-expanded` holder, so it has to be out of the picture for this to be about insights.
    const w = mountCard(
      makeEpisode({
        summary_bullets: [],
        has_gi: false,
        description: null,
      } as never)
    )
    expect(cardOwnExpanders(w)).toHaveLength(0)
  })
})

describe("the two columns are rebalanced (#2004 items 4/7)", () => {
  it("gives the show name the FULL column width on its own line, icons on a separate row", () => {
    // The action icons sit on their own right-aligned row; the show name is a full-width
    // `block truncate` BELOW them, so a long name — "Complex Systems with Patrick McKenzie
    // (patio11)" — uses the whole column and only ellipsizes when genuinely long, instead of being
    // crushed against the icons. Two earlier layouts failed: floated icons (the nowrap name ran
    // UNDER them) and a shared flex row (the name was squeezed to ~7 chars). The name must NOT be
    // flex-1 (that put it back on the icons' row).
    const w = mountCard(
      makeEpisode({ podcast_title: "Complex Systems with Patrick McKenzie (patio11)" })
    )
    const name = w.findAll("a").find((a) => a.text().includes("Complex Systems"))!
    expect(name.classes()).toContain("truncate")
    expect(name.classes()).toContain("block")
    expect(name.classes()).not.toContain("flex-1")
  })

  it("puts date and duration under the artwork", () => {
    const w = mountCard(makeEpisode())
    const left = w.get("article > div.lp-media-aside")
    expect(left.text()).toMatch(/\d/) // date / duration live here now
  })

  it("keeps the action row directly under the artwork, not footed to the column's bottom", () => {
    // `mt-auto` pushed the row to the bottom of a stretched column, so whenever the summary was the
    // taller side the controls floated below a gap, detached from the artwork they act on
    // (operator 2026-09-17).
    const w = mountCard(makeEpisode())
    const actions = w.get("article > div.lp-media-aside").find('[data-testid="episode-actions"]')
    expect(actions.exists()).toBe(true)
    expect(actions.classes()).not.toContain("mt-auto")
  })

  it("renders the artwork bigger than the old 80px", () => {
    const w = mountCard(makeEpisode({ artwork_url: "https://example.test/a.jpg" } as never))
    expect(w.get("img").classes()).toEqual(expect.arrayContaining(["h-32", "w-32"]))
  })

  it("keeps the column shape when an episode has no artwork", () => {
    // Without a placeholder the left column has no fixed-width child and collapses, squeezing the
    // date and insight count beside a zero-width gap.
    const w = mountCard(makeEpisode())
    const left = w.get("article > div.lp-media-aside")
    expect(left.find("img").exists()).toBe(false)
    expect(left.get('div[aria-hidden="true"]').classes()).toEqual(
      expect.arrayContaining(["h-32", "w-32"])
    )
  })
})

describe("the card carries no summary bullets (#2004 follow-up)", () => {
  it("renders NO bullet list", async () => {
    // Marko: "remove summary bullets". The bullets made one row fill the screen — Browse became a
    // scroll rather than a scan. The summary lives on the episode detail surface.
    const w = mountCard(makeEpisode())
    expect(w.find('[data-testid="card-bullets"]').exists()).toBe(false)
    expect(w.text()).not.toContain("Deep sleep consolidates memory.")
  })

  it("carries no key-points badge at ANY viewport", () => {
    // Removed outright (operator 2026-09-17). It had been hidden below `sm` and left visible from
    // `sm` up, so it survived on every desktop browser — the feature was gone from the product but
    // still on the card for anyone not on a phone. `hidden sm:inline-flex` is not a deletion.
    const w = mountCard(makeEpisode({ summary_bullets: ["a", "b", "c"], has_gi: true }))
    expect(w.find('[data-testid="card-key-point-count"]').exists()).toBe(false)
    expect(w.text()).not.toContain("key point")
  })
})


// --- the queue's two lists (operator 2026-09-23) ------------------------------------------------
//
// Both are EpisodeCard. Up next wants download visible (it is about to play, possibly off-signal);
// Recently played wants the queue toggle out of the way (it is about resuming, not re-queueing) and
// the 80px column only holds two targets anyway.

describe("action-row composition per surface", () => {
  /** The aria-labels of the buttons actually IN the visible row (the ⋯ panel is teleported). */
  function rowLabels(props: Record<string, unknown>): (string | undefined)[] {
    const w = mount(EpisodeCard, {
      props: { episode: makeEpisode(), ...props },
      global: { plugins: [i18n, router] },
    })
    return w
      .find('[data-testid="episode-actions"]')
      .findAll("button")
      .map((b) => b.attributes("aria-label"))
  }

  it("keeps download in the ⋯ by default", () => {
    // Four inline controls is 176px of 44px targets and wraps under the artwork-width column, which
    // is why the ⋯ exists (operator 2026-09-13). Default stays favourite · queue · ⋯.
    expect(rowLabels({}).some((l) => /download/i.test(l ?? ""))).toBe(false)
  })

  it("promotes download INTO the row for Up next, and out of the menu", () => {
    // "Is this on the device?" is the question Up next answers, usually right before losing signal,
    // and it was two taps behind a menu (operator 2026-09-23).
    const w = mount(EpisodeCard, {
      props: { episode: makeEpisode(), showDownload: true },
      global: { plugins: [i18n, router] },
    })
    const row = w.find('[data-testid="episode-actions"]')
    expect(row.find('[data-testid="download-button"]').exists()).toBe(true)
    // Exactly once in the whole card: promoted, not duplicated into the ⋯ as well.
    expect(w.findAll('[data-testid="download-button"]').length).toBe(1)
  })

  it("keeps queue in the row by default", () => {
    expect(rowLabels({}).some((l) => /queue/i.test(l ?? ""))).toBe(true)
  })

  it("demotes the queue toggle out of the row for Recently played", () => {
    // Demoted, NOT deleted — it moves into the ⋯, because it is still the only way to queue
    // something you just heard. The panel is teleported and renders on open, so what is asserted
    // here is its ABSENCE from the row; the menu half is covered end-to-end.
    const labels = rowLabels({ hideQueue: true })
    expect(labels.some((l) => /queue/i.test(l ?? ""))).toBe(false)
    expect(labels.some((l) => /more actions/i.test(l ?? ""))).toBe(true)
  })

  it("leaves the row two-wide when the queue toggle is demoted", () => {
    // The point of demoting it: an 80px compact column holds two 32px targets, so three wrapped the
    // ⋯ onto its own line under the artwork.
    expect(rowLabels({ hideQueue: true }).length).toBe(2)
  })
})

describe("Read more on a compact card", () => {
  it("is offered — compact is what Recently played renders", async () => {
    // It was suppressed on compact outright, so that list showed prose cut off with no way to reach
    // the rest (operator 2026-09-23). `summaryClipped` defaults true before layout, which is the
    // state a jsdom mount is in, so the toggle must be present here.
    const w = mount(EpisodeCard, {
      props: { episode: makeEpisode({ description: "A long description. ".repeat(40) }), compact: true },
      global: { plugins: [i18n, router] },
    })
    expect(w.find('[data-testid="card-read-more"]').exists()).toBe(true)
  })
})

