import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import appSrc from "../App.vue?raw"
import addToCollectionSrc from "../components/AddToCollectionButton.vue?raw"
import downloadSrc from "../components/DownloadButton.vue?raw"
import episodeActionsSrc from "../components/EpisodeActions.vue?raw"
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
    // Matched on the EXACT i18n keys, closing quote included. A bare `includes("queue.up")` also
    // matched `queue.upNext` — the page's section heading, added 2026-09-23 — and counted the text
    // before the first `<button` as a third arrow. The guard was right about the app and wrong
    // about the string, which is the failure mode a substring matcher invites.
    // `slice(1)`: split()'s first chunk is whatever precedes the first `<button`, never a button.
    const arrows = queueViewSrc
      .split("<button")
      .slice(1)
      .filter((b) => b.includes("'queue.up'") || b.includes("'queue.down'"))
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
    // The row now lives in the ONE shared EpisodeActions component (every card/tile/rail renders it,
    // nobody rolls their own), so the invariant is asserted there. `flex-wrap` also makes gap-[12px]
    // the vertical pitch when the row folds inside a width-constrained caller (the list card's
    // left column).
    const row = episodeActionsSrc.slice(episodeActionsSrc.indexOf("flex flex-wrap items-center"))
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

  it("the Saved colour strip FITS a phone, so its last swatch is not hidden behind a scroll", () => {
    /*
     * The strip is `overflow-x-auto` with the scrollbar suppressed, so overspending the width does
     * not look like a bug — it looks like five colours. Measured off the operator's 393pt device on
     * 2026-09-27: the content box is 324.6pt, and the row was asking for 284pt of swatches (6 x 44
     * + 5 x gap-1) plus 8 + 1 + 8 + 44 on the right = 345pt. Violet was clipped exactly in half.
     *
     * The budget below is what makes the row honest. It is asserted from the source rather than
     * from a layout, because jsdom does not lay anything out and the failure is purely dimensional
     * — the same reason every other check in this file is static.
     */
    const PHONE_CONTENT_PT = 324.6 // measured; 393pt device less its ~34pt margins
    const TARGET = 44 // h-11 / w-11, the touch floor the check above pins
    const SWATCHES = 6 // "any colour" + HIGHLIGHT_COLORS
    const SEPARATOR = 1 // the w-px hairline

    // The row and the right-hand cluster must both be at gap-1 (4pt), and the swatch group at no
    // gap at all. A `gap-2` anywhere here is 4pt the strip does not have.
    // The LAST such div before the swatches, not the first — the first is the search+sort row
    // above, which has its own width budget and would let a `gap-2` here pass unnoticed.
    const anyAt = savedFilterBarSrc.indexOf("savedFilterColorAny")
    const row = savedFilterBarSrc.slice(
      savedFilterBarSrc.lastIndexOf('<div class="flex items-center gap-', anyAt),
      anyAt,
    )
    expect(row, "the colour row went back to gap-2 — that is 4pt the strip cannot spare").toContain(
      '<div class="flex items-center gap-1">',
    )
    expect(
      row,
      "the swatch group must carry NO gap: a 16pt dot inside a 44pt target is already spaced, and " +
        "the gap is what pushed the sixth colour off screen",
    ).not.toMatch(/class="flex min-w-0 items-center gap-\d/)
    expect(
      savedFilterBarSrc,
      "the muted/clear cluster went back to gap-2",
    ).toContain('class="ml-auto flex shrink-0 items-center gap-1"')

    // And the arithmetic those classes buy, stated so a seventh colour fails HERE rather than on a
    // device: strip + gap + hairline + gap + bell must clear the phone's content box.
    const needed = SWATCHES * TARGET + 4 + SEPARATOR + 4 + TARGET
    expect(
      needed,
      `the Saved filter row needs ${needed}pt but a phone gives ${PHONE_CONTENT_PT}pt — something ` +
        "was added to the row. Adding a colour or a control here means re-deriving this budget, " +
        "not letting the strip scroll: the scrollbar is hidden, so the overflow is invisible",
    ).toBeLessThanOrEqual(PHONE_CONTENT_PT)
  })
})

/**
 * Guardrail (#1588, operator 2026-09-20) — search must stay reachable on a phone.
 *
 * #1588 existed because search had ONE entry point and was unreachable from the catalogue, player,
 * library or a show page. Folding search into Discovery removed its bottom-nav tab, so the masthead
 * magnifier is now the only always-available search control. If it slips back inside the
 * `hidden … sm:flex` span it vanishes on phones and #1588 is live again — silently, because the
 * desktop layout would still look correct.
 *
 * A static source check for the same reason as the rest of this file: the breakage is a media
 * query, and jsdom does not evaluate one, so a mounted test would pass either way.
 */
describe("search survives the loss of its tab (#1588)", () => {
  it("the masthead search link sits OUTSIDE the desktop-only icon span", () => {
    const searchAt = appSrc.indexOf('data-testid="masthead-search"')
    expect(searchAt, "masthead search link not found in App.vue").toBeGreaterThan(-1)

    const desktopOnlyAt = appSrc.indexOf('class="hidden items-center gap-1.5 sm:flex"')
    expect(desktopOnlyAt, "desktop-only icon span not found in App.vue").toBeGreaterThan(-1)

    // Before the span opens => not inside it => visible at every width.
    expect(
      searchAt,
      "the masthead search icon must not be inside the `hidden … sm:flex` span: it is the only " +
        "always-available search control now that Search is not a bottom-nav tab",
    ).toBeLessThan(desktopOnlyAt)
  })

  it("the bottom nav no longer carries a Search tab", () => {
    // Paired with the check above: if BOTH the tab and the always-visible icon disappeared, a
    // phone would have no nav-level search at all.
    const nav = readFileSync(
      path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "components", "BottomNav.vue"),
      "utf8",
    )
    const tabs = nav.slice(nav.indexOf("const TABS"), nav.indexOf("] as const"))
    expect(tabs).not.toContain("'search'")
    expect(tabs).toContain("'browse'")
  })
})

/**
 * Guardrail (operator 2026-09-23, re-homed 2026-09-27) — the queue must stay reachable without
 * playing something first.
 *
 * `/queue` once had no nav entry: the only ways in were the player's queue button and Home's resume
 * hero, and that hero renders ONLY while an episode is in progress. Finish everything you were
 * listening to and the queue you had been filling became unreachable unless you first started an
 * episode you did not want to play — which is also the state you are in offline, wanting exactly
 * the thing you queued.
 *
 * Both of those entrances were removed on 2026-09-27 as duplication, leaving the masthead control
 * as the ONLY one. That makes these checks load-bearing in a way they were not before: with the
 * fallbacks gone, this link slipping inside the `hidden … sm:flex` span takes the queue off phones
 * entirely, and the desktop layout would still look correct.
 *
 * Static source checks for the same reason as the rest of this file — the breakage is a media
 * query, and jsdom does not evaluate one.
 */
describe("the queue is reachable at every width", () => {
  it("the masthead queue link sits OUTSIDE the desktop-only icon span", () => {
    const queueAt = appSrc.indexOf('data-testid="masthead-queue"')
    expect(queueAt, "masthead queue link not found in App.vue").toBeGreaterThan(-1)

    const desktopOnlyAt = appSrc.indexOf('class="hidden items-center gap-1.5 sm:flex"')
    expect(desktopOnlyAt, "desktop-only icon span not found in App.vue").toBeGreaterThan(-1)
    const spanClosesAt = appSrc.indexOf("</span>", desktopOnlyAt)
    expect(spanClosesAt, "desktop-only span never closes").toBeGreaterThan(desktopOnlyAt)

    // After the span closes => not inside it => visible at every width.
    expect(
      queueAt,
      "the masthead queue icon must not be inside the `hidden … sm:flex` span: since Home's " +
        "resume hero and the player's panel button were removed, it is the only way into /queue",
    ).toBeGreaterThan(spanClosesAt)
  })

  it("the go-to-queue glyph carries no add-modifier", () => {
    // It drew the list WITH a trailing wedge, which read as the same "add to queue" mark
    // `QueueButton` draws with a plus — a destination dressed as an action (operator 2026-09-27).
    // The modifier is the whole difference, so the absence of one is the thing to pin.
    const start = appSrc.indexOf('data-testid="masthead-queue"')
    const glyph = appSrc.slice(start, appSrc.indexOf("</svg>", start))
    const paths = [...glyph.matchAll(/<path d="([^"]+)"/g)].map((m) => m[1])
    expect(paths.length, "expected the masthead queue glyph's paths").toBeGreaterThan(0)
    for (const d of paths) {
      expect(
        d,
        `"${d}" is not a plain horizontal rule — a queue DESTINATION must not draw a plus, a ` +
          "tick or a wedge, or it reads as add-to-queue",
      ).toMatch(/^M\d+ \d+h\d+$/)
    }
  })

  it("QueueButton keeps a modifier, so the two are never the same mark", () => {
    // The other half of the pair: if QueueButton ever lost its plus/tick, the check above would
    // still pass while both controls drew the identical bare list.
    expect(queueButtonSrc, "the add state needs its plus").toContain('d="M21 12h-6"')
    expect(queueButtonSrc, "the queued state needs its tick").toContain('d="M15 16l2 2 4-4"')
  })
})

