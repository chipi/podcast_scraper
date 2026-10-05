import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import appSrc from "../App.vue?raw"
import addToCollectionSrc from "../components/AddToCollectionButton.vue?raw"
import downloadSrc from "../components/DownloadButton.vue?raw"
import episodeActionsSrc from "../components/EpisodeActions.vue?raw"
import favoriteSrc from "../components/FavoriteButton.vue?raw"
import navIconLinkSrc from "../components/NavIconLink.vue?raw"
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
    //
    // Split on `<Highlight`, not `<button`: the control became the shared `HighlightToggle` on
    // 2026-09-27. Splitting on a tag the file no longer contains yields ONE chunk — the whole
    // source — which still matched the regex and still contained the required class, somewhere.
    // The check went on passing while testing nothing in particular, which is the failure mode
    // this whole file exists to catch in the app.
    const controls = transcriptSrc.split("<HighlightToggle").filter((b) => HOVER_HIDDEN.test(b))
    expect(
      controls.length,
      "expected a hover-quiet capture control to exist — if it was renamed again, re-anchor this " +
        "split rather than letting it match the whole file",
    ).toBeGreaterThan(0)
    for (const b of controls) {
      expect(
        b,
        "A hover-hidden control must also carry [@media(hover:none)]:opacity-100, or it is " +
          "invisible on the primary platform."
      ).toContain("[@media(hover:none)]:opacity-100")
    }
  })

  it("a hover TOOLTIP is gated to devices that hover, so a tap cannot leave it stuck", () => {
    /*
     * The mirror image of the check above, and the operator found it on device (2026-09-27): "why
     * the queue label under button stays when I press it?"
     *
     * iOS applies `:hover` on tap and leaves it applied until you tap elsewhere. There is no hover
     * to end, so a `group-hover` tooltip lights on tap, SURVIVES the navigation, and sits under the
     * icon on the page it just opened — captioning a control the user is no longer looking at.
     *
     * A control hidden behind hover needs `[@media(hover:none)]:opacity-100` so touch can reach it.
     * A tooltip needs the opposite: `[@media(hover:hover)]:` so touch never triggers it. Both rules
     * live here because they are one question — what does hover mean on a device without one — and
     * getting them backwards is easy.
     */
    const tooltip = navIconLinkSrc.slice(
      navIconLinkSrc.indexOf('role="tooltip"') - 900,
      navIconLinkSrc.indexOf('role="tooltip"'),
    )
    expect(tooltip, "the tooltip span moved — re-anchor this check").toContain("opacity-0")
    expect(
      tooltip,
      "the nav tooltip reveals on a bare `group-hover`, so tapping the icon leaves the label " +
        "stuck under it on the next page. Gate it with [@media(hover:hover)]:group-hover:…",
    ).not.toMatch(/(?<!\]:)group-hover:opacity-100/)
    expect(tooltip).toContain("[@media(hover:hover)]:group-hover:opacity-100")
    // Keyboard focus is a real state that ends, so it must still reveal.
    expect(tooltip, "keyboard users lost the label").toContain("group-focus-visible:opacity-100")
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
 * Guardrail (#1588) — search must stay reachable on a phone, from any screen.
 *
 * #1588 existed because search had ONE entry point and was unreachable from the catalogue, player,
 * library or a show page. What guarantees it on a phone has changed twice: first a bottom-nav tab,
 * then (2026-09-20) the masthead magnifier at every width, and since 2026-09-30 the Discover TAB and
 * its search box — the magnifier left the phone header because "Close Listening" ran under it. The
 * Discover tab is on every phone screen, so its box is one tap from anywhere. These checks pin that
 * pair; if either goes, a phone has no always-reachable search and #1588 is live again — silently,
 * because desktop keeps its magnifier and still looks right.
 *
 * A static source check for the same reason as the rest of this file: the breakage is a media
 * query, and jsdom does not evaluate one, so a mounted test would pass either way.
 */
describe("search survives the loss of its tab (#1588)", () => {
  it("phones reach search through the Discover tab's own search box", () => {
    const dir = path.dirname(fileURLToPath(import.meta.url))
    const nav = readFileSync(path.join(dir, "..", "components", "BottomNav.vue"), "utf8")
    const tabs = nav.slice(nav.indexOf("const TABS"), nav.indexOf("] as const"))
    expect(tabs, "the Discover (browse) tab is what makes search reachable on a phone").toContain(
      "'browse'",
    )
    // The box lives in the block Home and Discover share (AskAndTrends, 2026-10-05); Discover
    // renders it under its own `browse` ids.
    const browse = readFileSync(path.join(dir, "..", "views", "BrowseView.vue"), "utf8")
    expect(browse, "Discovery must carry the shared search + trends block").toMatch(
      /<AskAndTrends[^>]*prefix="browse"/,
    )
    const block = readFileSync(path.join(dir, "..", "components", "AskAndTrends.vue"), "utf8")
    expect(block, "the shared block must carry Discover's search input").toContain("'browse-search-input'")
  })

  it("the masthead magnifier is desktop-only (no room for it in a phone header)", () => {
    const searchAt = appSrc.indexOf('data-testid="masthead-search"')
    expect(searchAt, "masthead search link not found in App.vue").toBeGreaterThan(-1)
    const desktopOnlyAt = appSrc.indexOf('class="hidden items-center gap-1.5 sm:flex"')
    expect(desktopOnlyAt, "desktop-only icon span not found in App.vue").toBeGreaterThan(-1)
    expect(searchAt, "the magnifier must sit inside the desktop-only span").toBeGreaterThan(
      desktopOnlyAt,
    )
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
      /*
       * HORIZONTAL COMMANDS ONLY — `M` and `h`, nothing else.
       *
       * The invariant is "no modifier", not "no decimals". The first version of this required
       * `^M\d+ \d+h\d+$`, which encoded the glyph that happened to be there rather than the rule,
       * and it rejected the list BULLETS added on 2026-09-27 (`M4 6h.01`) — a correct change failing
       * a guard that had over-specified. Stated properly, every modifier this is meant to exclude
       * needs a command it now forbids: a plus needs `v` for its upright, a tick needs `l`, a wedge
       * needs `l`/`z`. A bullet is a zero-length `h`, which is still just a horizontal mark.
       */
      expect(
        d,
        `"${d}" uses a non-horizontal path command — a queue DESTINATION must not draw a plus ` +
          "(needs v), a tick or a wedge (need l/z), or it reads as an action on this episode",
      ).toMatch(/^M[\d.]+ [\d.]+(h-?[\d.]+)+$/)
    }
  })

  it("QueueButton keeps a modifier, so the two are never the same mark", () => {
    // The other half of the pair: if QueueButton ever lost its plus/tick, the check above would
    // still pass while both controls drew the identical bare list.
    expect(queueButtonSrc, "the add state needs its plus").toContain('d="M21 12h-6"')
    expect(queueButtonSrc, "the queued state needs its tick").toContain('d="M15 16l2 2 4-4"')
  })
})

