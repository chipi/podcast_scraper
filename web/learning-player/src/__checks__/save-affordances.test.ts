import { describe, expect, it } from "vitest"
import appSrc from "../App.vue?raw"
import addToCollectionSrc from "../components/AddToCollectionButton.vue?raw"
import bookmarkIconSrc from "../components/BookmarkIcon.vue?raw"
import captureMomentSrc from "../components/CaptureMoment.vue?raw"
import favoriteSrc from "../components/FavoriteButton.vue?raw"
import highlightToggleSrc from "../components/HighlightToggle.vue?raw"
import queueButtonSrc from "../components/QueueButton.vue?raw"
import knowledgePanelSrc from "../components/KnowledgePanel.vue?raw"
import transcriptSrc from "../components/TranscriptList.vue?raw"

/**
 * One glyph per concept (UXS-014, operator 2026-09-27).
 *
 *   heart    = favourite         a WHOLE object          favorites store
 *   bookmark = highlight         a FRAGMENT              capture store
 *   folder+  = file to a board   anything                collections store
 *
 * The app had broken this in both directions at once, and neither break was visible from inside the
 * code — each component was individually reasonable:
 *
 *   - the insight save drew a HEART and announced "Save to favorites" while writing a capture,
 *     even though `types.ts` says `insight` is not a favourite kind (#1593)
 *   - `AddToCollectionButton` drew the BOOKMARK, which is `CaptureMoment`'s mark. The operator read
 *     the player's capture control as a stray collections button and asked to delete it — on a
 *     phone that is the only way to mark a moment, so a glyph collision came one instruction away
 *     from removing a feature.
 *
 * These are source checks because the failure is a resemblance between two files, which no mounted
 * test of either one can see.
 */

/**
 * Strip comments before matching.
 *
 * Every one of these files EXPLAINS the rule, quoting the glyph it no longer draws and the variant
 * that no longer exists — so a naive `toContain` fails on the documentation of the fix. The same
 * move `sheet-geometry.test.ts` makes, and for the same reason.
 */
function code(src: string): string {
  return src
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "")
}

/** The bookmark outline both capture controls draw, modulo the fill/stroke wrapper. */
const BOOKMARK = "v17l-7-4-7 4V4"

describe("one glyph per concept", () => {
  it("the bookmark belongs to CAPTURE — both controls that make a highlight draw it", () => {
    expect(code(bookmarkIconSrc), "BookmarkIcon stopped drawing a bookmark").toContain(BOOKMARK)
    expect(code(captureMomentSrc), "mark-a-moment stopped drawing the bookmark").toContain(BOOKMARK)
    // The transcript line and the insight go through the shared toggle rather than re-drawing it.
    expect(code(highlightToggleSrc)).toContain("BookmarkIcon")
    expect(code(transcriptSrc), "the transcript line must use the shared toggle").toContain(
      "HighlightToggle",
    )
    expect(code(knowledgePanelSrc), "the insight save must use the shared toggle").toContain(
      "HighlightToggle",
    )
  })

  it("add-to-collection draws a BOARD — not a bookmark, and not a plus", () => {
    const src = code(addToCollectionSrc)

    // It drew the bookmark until 2026-09-27, which is what made the player's capture button look
    // like a stray copy of the header's collect button.
    expect(
      src,
      "the collections glyph is colliding with the capture bookmark again — give it its own mark",
    ).not.toContain(BOOKMARK)
    expect(src, "...and not the outlined bookmark it used to draw either").not.toContain(
      "M6 3v18l6-4 6 4V3z",
    )

    // Four cells, positively asserted — "not a bookmark" alone would pass on a blank icon, or on a
    // folder, which was tried and rejected as the filesystem's metaphor for something the product
    // calls a Board (RFC-119: pinboards).
    const cells = [...src.matchAll(/<rect [^>]*width="7" height="7"/g)]
    expect(cells.length, "the board grid lost cells").toBe(4)

    /*
     * No plus. The pill variant already says "+ Collection" in words and the icon variant carries
     * `collections.addTo` as its accessible name, so a `+` is a third stroke paying for nothing —
     * and at 16px, the size that ships in a card row, it is exactly what made the rejected folder
     * the busiest mark in the set. Same call made against the folded-corner-plus-plus glyph on
     * 2026-09-13; this keeps it from being re-added a third time.
     */
    const iconBlock = src.slice(src.indexOf("v-else viewBox"), src.indexOf("</svg>"))
    expect(iconBlock, "a plus crept back into the collections glyph").not.toMatch(
      /d="M\d+ \d+v\d+"|d="M\d+ \d+h\d+"/,
    )
  })

  it("the heart is FAVOURITES only — it cannot be pointed at another store", () => {
    /*
     * `FavoriteButton` used to accept a `controlled` variant letting a parent own the active state
     * and the toggle. Its only caller was the insight save, writing to the capture store. The
     * variant is gone, and its absence is the guard: as long as one exists, the heart can be
     * reattached to a non-favourite store, which is precisely how the insight heart happened.
     */
    expect(
      code(favoriteSrc),
      "a `controlled` variant is back — the heart can now be pointed at a non-favourites store",
    ).not.toContain("controlled")
    expect(code(favoriteSrc), "the heart must read its own state from the favorites store").toContain(
      "favorites.has(",
    )
  })

  it("no two control glyphs draw the same geometry", () => {
    /*
     * The collision this exists for, and it is MINE (operator 2026-09-27, second pass).
     *
     * Told to stop the queue destination looking like an add-action, I removed its trailing wedge
     * and landed on a bare three-line list — byte-identical to the Summary opener over the player
     * artwork (`M4 6h16M4 12h16M4 18h10`). Two controls, one drawing, on screens the reader sees
     * together. I replaced one collision with another inside the same session, and the checks above
     * did not notice because they police heart / bookmark / board by name and nothing else.
     *
     * A by-name check can only catch collisions someone already thought of. This compares the
     * DRAWINGS: normalise every `d` attribute in the app's control glyphs and assert no two
     * distinct controls share a set. It catches the next one without anyone predicting it.
     *
     * Scoped to the glyphs that sit in shared chrome and action rows, because that is where
     * "which button is this?" is a real question. Decorative and one-off illustration paths are out.
     */
    /*
     * Keyed by CONCEPT, not by file. Sharing a drawing within one concept is the POINT — the whole
     * rule is that a highlight looks like a highlight wherever it is made, so `BookmarkIcon` and
     * `CaptureMoment` drawing the same bookmark is correct and this check must not flag it. What is
     * forbidden is two DIFFERENT concepts converging on one drawing, which is what the operator
     * kept running into: a capture control that looked like add-to-collection, then a queue
     * destination that looked like Summary.
     */
    const SURFACES: Array<[string, string]> = [
      ["go-to-queue", appSrc],
      ["queue-this-episode", queueButtonSrc],
      ["add-to-collection", addToCollectionSrc],
      ["highlight", captureMomentSrc],
      ["highlight", bookmarkIconSrc],
    ]

    /** Every `d="…"` in one file's icon SVGs, normalised and sorted into a comparable signature. */
    const glyphsOf = (src: string): string[] =>
      [...code(src).matchAll(/<svg[\s\S]*?<\/svg>/g)]
        .map((m) =>
          [...m[0].matchAll(/\sd="([^"]+)"/g)]
            .map((d) => d[1].replace(/\s+/g, " ").trim())
            .sort()
            .join(" | "),
        )
        .filter((sig) => sig.length > 0)

    const seen = new Map<string, string>()
    for (const [name, src] of SURFACES) {
      for (const sig of glyphsOf(src)) {
        const prev = seen.get(sig)
        expect(
          prev === undefined || prev === name,
          `${name} draws the same glyph as ${prev} — two controls, one drawing. Give one of them ` +
            `its own mark; see the "one glyph per concept" rule in UXS-014.\n  geometry: ${sig}`,
        ).toBe(true)
        seen.set(sig, name)
      }
    }
    expect(seen.size, "the glyph scan found nothing — the SVG regex has drifted").toBeGreaterThan(4)
  })

  it("the masthead queue does not draw the Summary glyph", () => {
    // The specific collision, pinned by value as well as by the general rule above — a regression
    // here is worth a message that names both controls rather than a generic duplicate report.
    const SUMMARY = "M4 6h16M4 12h16M4 18h10"
    const queue = code(appSrc).slice(
      code(appSrc).indexOf('data-testid="masthead-queue"'),
      code(appSrc).indexOf("</svg>", code(appSrc).indexOf('data-testid="masthead-queue"')),
    )
    const paths = [...queue.matchAll(/\sd="([^"]+)"/g)].map((m) => m[1]).join("")
    expect(
      paths.replace(/\s/g, ""),
      "the masthead queue is drawing the player's Summary glyph again",
    ).not.toBe(SUMMARY.replace(/\s/g, ""))
    expect(paths, "the queue glyph lost its list bullets").toContain("h.01")
  })

  it("no heart renders on a fragment", () => {
    // Asserted on the two surfaces that save fragments. Presence of the bookmark is covered above;
    // this is the other half — adding a heart BESIDE it would satisfy that check and not this one.
    for (const [name, src] of [
      ["KnowledgePanel", code(knowledgePanelSrc)],
      ["TranscriptList", code(transcriptSrc)],
    ] as const) {
      expect(src, `${name} renders a FavoriteButton — a fragment is not favouritable (#1593)`).not.toContain(
        "<FavoriteButton",
      )
      expect(src, `${name} hand-rolls a heart glyph`).not.toMatch(/[♥♡]/)
    }
  })
})
