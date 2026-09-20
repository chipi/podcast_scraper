import { describe, expect, it } from "vitest"
import { exportFilename, filenameStem } from "./exportFilename"

/**
 * Exports were named from the episode SLUG — `{feed_slug}-{sha256hex}` — so saving your notes
 * produced `long-horizon-notes-9f2c4a1b8e…-notes.md`, reported as "some crazy name" (operator
 * 2026-09-19). A slug is an addressing detail; a saved file is something a person finds by reading
 * it months later.
 */
describe("exportFilename", () => {
  it("turns a title into something a person would recognise", () => {
    expect(exportFilename("How Sleep Works", "md", "notes")).toBe("how-sleep-works.md")
  })

  it("never lets path syntax into a filename", () => {
    // This string reaches a filesystem, a share sheet, and whatever app receives it. A slash is
    // the difference between a file and a write somewhere else.
    const bad = 'Q&A: sleep/wake — "cycles" <or> \\rhythms?|* 🎧'
    const out = filenameStem(bad, "notes")
    // An ALLOWLIST: anything that is not [a-z0-9] becomes a separator. A blacklist missed the
    // em-dash and the emoji, because it only knows the characters someone thought of.
    expect(out).toMatch(/^[a-z0-9-]+$/)
    expect(out).toBe("q-a-sleep-wake-cycles-or-rhythms")
  })

  it("folds diacritics instead of dropping the letter", () => {
    // "beyonc.md" would be a worse name than the slug it replaced.
    expect(filenameStem("Beyoncé at Coachella", "notes")).toBe("beyonce-at-coachella")
  })

  it("bounds the length — a 200-character title is real and is not a filename", () => {
    const stem = filenameStem("word ".repeat(80), "notes")
    expect(stem.length).toBeLessThanOrEqual(60)
    // And it does not end mid-separator, which would read as a typo.
    expect(stem).not.toMatch(/-$/)
  })

  it("falls back rather than producing a bare extension", () => {
    // Emoji-only and CJK titles strip to nothing under an ASCII-ish rule. ".md" is not a filename.
    for (const empty of ["", "   ", "🎧🎧🎧", "—", null, undefined]) {
      expect(exportFilename(empty, "md", "episode-notes")).toBe("episode-notes.md")
    }
  })

  it("collapses runs rather than emitting doubled separators", () => {
    expect(filenameStem("Sleep   ---   Wake", "notes")).toBe("sleep-wake")
    expect(filenameStem("Ep. 12 — “AI” & you", "notes")).toBe("ep-12-ai-you")
  })
})
