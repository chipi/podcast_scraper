import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import appSrc from "../App.vue?raw"
import addToCollectionSrc from "../components/AddToCollectionButton.vue?raw"
import downloadSrc from "../components/DownloadButton.vue?raw"
import episodeCardSrc from "../components/EpisodeCard.vue?raw"
import favoriteSrc from "../components/FavoriteButton.vue?raw"
import queueButtonSrc from "../components/QueueButton.vue?raw"
import transcriptSrc from "../components/TranscriptList.vue?raw"
import savedColorControlSrc from "../components/SavedColorControl.vue?raw"
import savedFilterBarSrc from "../components/SavedFilterBar.vue?raw"
import queueViewSrc from "../views/QueueView.vue?raw"

// `../style.css?raw` imports as an EMPTY string — vitest stubs CSS modules, so the guard below
// asserted nothing at all and passed. Read it off disk instead.
const styleSrc = readFileSync(
  path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "style.css"),
  "utf8"
)

/**
 * Guardrail (#1588, #1592) — two properties of the shell that are easy to regress silently and
 * expensive when they do.
 *
 * Static source checks rather than mounted assertions, because both are about CSS that only takes
 * effect under a media query jsdom does not evaluate. A mounted test would pass either way, which
 * is precisely how the transcript capture button stayed invisible on phones for so long.
 */

/** Every component that hides an affordance behind hover must also show it where hover is absent. */
const HOVER_HIDDEN = /opacity-0[^"]*group-hover:opacity-100/

describe("affordances survive on touch", () => {
  it("the transcript capture control is visible without hover (#1592)", () => {
    // Phones are the primary platform (the e2e suite's default project is a Pixel 7) and have no
    // hover. `opacity-0 + group-hover` alone leaves the control transparent but tappable —
    // undiscoverable rather than obviously missing, which is worse than absent. This is the entry
    // point to capture → highlights → notes → resurfacing, i.e. the whole learning loop.
    const buttons = transcriptSrc.split("<button").filter((b) => HOVER_HIDDEN.test(b))
    expect(buttons.length, "expected a hover-quiet capture button to exist").toBeGreaterThan(0)
    for (const b of buttons) {
      expect(
        b,
        "A hover-hidden control must also carry [@media(hover:none)]:opacity-100, or it is " +
          "invisible on the primary platform."
      ).toContain("[@media(hover:none)]:opacity-100")
    }
  })

  it("search is reachable from the primary nav (#1588)", () => {
    // Corpus-wide semantic search with jump-to-moment is the differentiator neither Spotify nor
    // Apple Podcasts offers. It previously had one entry point — the Home search box — so from the
    // catalogue, player, library or a show page there was no way to reach it at all.
    const nav = appSrc.slice(appSrc.indexOf("<nav"), appSrc.indexOf("</nav>"))
    expect(nav).toContain("name: 'search'")
  })

  it("search stays public, like browse — reads are open", () => {
    // If the search link were inside the `auth.isAuthenticated` block, signed-out visitors would
    // lose the one capability most likely to convert them.
    const nav = appSrc.slice(appSrc.indexOf("<nav"), appSrc.indexOf("</nav>"))
    const gated = nav.slice(nav.indexOf("auth.isAuthenticated"))
    expect(gated).not.toContain("name: 'search'")
  })

  /**
   * ## 44px targets (#1594)
   *
   * Source checks, for the same reason as the hover rule above: jsdom performs no layout, so
   * `offsetWidth` is 0 for everything and a mounted assertion would pass whatever the size. The
   * real geometry is measured in `e2e/touch-targets.spec.ts` against a Pixel 7; these checks are
   * the cheap regression net that runs on every commit.
   */
  const CARD_ACTIONS: Array<[string, string]> = [
    ["FavoriteButton", favoriteSrc],
    ["QueueButton", queueButtonSrc],
    ["DownloadButton", downloadSrc],
    ["AddToCollectionButton", addToCollectionSrc],
  ]

  it.each(CARD_ACTIONS)("%s carries the 44px hit area", (_name, src) => {
    // The ring is 32px because four 44px circles would be 176px of a 375px phone. `lp-tap` grows
    // the FINGER target without growing the ink; without it the control is a 32px target, which
    // is the sub-minimum size #1594 was opened about.
    expect(src).toContain("lp-tap")
  })

  it("the queue reorder arrows carry it too", () => {
    // These were 28px — the smallest targets in the app, and the ones most likely to be used in
    // motion, since reordering a queue is something you do while walking.
    const arrows = queueViewSrc
      .split("<button")
      .filter((b) => b.includes("queue.up") || b.includes("queue.down"))
    expect(arrows, "expected the up/down reorder buttons").toHaveLength(2)
    for (const a of arrows) expect(a).toContain("lp-tap")
  })

  it("the action row gap keeps neighbouring targets from overlapping", () => {
    // 32px ring + 12px gap = 44px pitch exactly. Shrink the gap and the invisible 44px boxes
    // overlap; the overlap band belongs to whichever button paints last, so a deliberate tap on
    // Download can fire Add-to-collection. Nothing about that is visible on screen, which is why
    // it needs a test rather than an eye.
    //
    // `gap-[12px]`, not `gap-3`: the scale is in rem and this app's root is not 16px, so `gap-3`
    // measured 11.4px on a Pixel 7 and the targets overlapped by 0.6px. Asserting the literal
    // px value is the point — a future `gap-3` here would re-introduce exactly that.
    // Anchor on the action-cluster classes, not a `<div class="` prefix: the cluster now floats
    // top-right (`float-right ml-3 flex shrink-0 items-center gap-[12px]`), so the class string no
    // longer STARTS at that div — but the 44px-pitch invariant (gap-[12px]) is what matters.
    const row = episodeCardSrc.slice(episodeCardSrc.indexOf("flex shrink-0 items-center"))
    expect(row.slice(0, 120)).toContain("gap-[12px]")
  })

  it("lp-tap sets its own positioning context", () => {
    // If `position` were left to the six call sites, one forgetting it would anchor the 44px box
    // to some ancestor: the target silently lands somewhere else on the page and looks fine.
    const rule = styleSrc.slice(styleSrc.indexOf(".lp-tap {"), styleSrc.indexOf(".lp-tap::after"))
    expect(rule).toContain("position: relative")
    const after = styleSrc.slice(styleSrc.indexOf(".lp-tap::after"))
    expect(after).toContain("width: 44px")
    expect(after).toContain("height: 44px")
  })

  it("the saved-item colour swatches are 44px buttons, not 16px dots", () => {
    // A 24px pitch cannot hold 44px targets at all, so these could not be fixed with `lp-tap` —
    // the button itself had to grow and the dot moved inside it. Assert the button is h-11 AND that
    // the coloured dot is a child, so "make the dot 44px" (five fat circles) also fails.
    //
    // The swatches moved out of HighlightsView in the Saved rework (RFC-121 ph. 3): the SET picker
    // is the shared `SavedColorControl`, the colour FILTER is the lifted `SavedFilterBar`. Both must
    // still be 44px.
    const swatches = [
      ...savedColorControlSrc.split("<button").filter((b) => b.includes("highlights.setColor")),
      ...savedFilterBarSrc.split("<button").filter((b) => b.includes("savedFilterColorOnly")),
    ]
    expect(swatches, "expected the set-picker and the filter swatch rows").toHaveLength(2)
    for (const b of swatches) {
      expect(b).toContain("h-11 w-11")
      expect(b, "the dot must be an inner span, not the button itself").toMatch(
        /<span[^>]*c\.swatch|<span[\s\S]*?c\.swatch/
      )
    }
  })
})
