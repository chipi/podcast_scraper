/**
 * Icons that carry meaning must be DRAWN, not typed (UXS-014).
 *
 * Four separate bugs this session were the same bug: a codepoint used as an icon rendered as a tofu
 * box on iOS, because the glyph is absent from the platform UI font.
 *
 *   ✕  U+2715  every sheet/panel/modal close → "?"   (caught by cropping a device screenshot)
 *   ✎  U+270E  profile avatar edit badge     → "?"
 *   ►  U+25BA  the Resume button             → "?"
 *   ＋ U+FF0B  follow / add-to-collection    → "?"   (a FULLWIDTH plus, in a Latin UI)
 *   🔥 U+1F525 the streak mark               → "?"
 *
 * Adding emoji/symbol faces to the font stacks did NOT fix any of them.
 *
 * Nothing else can catch this: these elements are `aria-hidden`, so they are invisible to both
 * Playwright and XCUITest — every one was found by a human looking at a picture. A denylist is the
 * only automated guard available, so it is deliberately EVIDENCE-BASED: a codepoint goes in only
 * once it has been observed failing on a real screen. Glyphs verified to render (‹ › → ✓ ● ↑ ↓)
 * are intentionally absent.
 */
import { describe, expect, it } from "vitest"
import { readdirSync, readFileSync, statSync } from "node:fs"
import { join } from "node:path"

/** codepoint → where it was seen failing, for the failure message. */
const TOFU: Record<string, string> = {
  "✕": "✕ U+2715 — sheet/panel close; use <CloseIcon />",
  "✎": "✎ U+270E — profile edit badge; draw it",
  "►": "► U+25BA — use ▶ U+25B6, which renders, or an SVG",
  "＋": "＋ U+FF0B — FULLWIDTH plus; use ASCII +",
  "\u{1F525}": "🔥 U+1F525 — emoji; drop it or draw it",
  "☷": "☷ U+2637 — board cover placeholder; draw it",
  "⠿": "⠿ U+283F — drag grip; draw it",
}

function vueFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) vueFiles(p, out)
    else if (p.endsWith(".vue")) out.push(p)
  }
  return out
}

describe("icon glyphs", () => {
  it("no component renders a codepoint known to be tofu on iOS", () => {
    const offenders: string[] = []
    for (const file of vueFiles("src")) {
      const src = readFileSync(file, "utf8")
      // Only the TEMPLATE renders — and within it, only non-comment content. The fixes for these
      // very glyphs left comments EXPLAINING them, so scanning raw template text flags the
      // documentation of a fix as if it were the bug.
      const template = src
        .slice(src.indexOf("<template>"))
        .replace(/<!--[\s\S]*?-->/g, "")
      for (const [glyph, why] of Object.entries(TOFU)) {
        if (template.includes(glyph)) {
          offenders.push(`${file.replace("src/", "")}: ${why}`)
        }
      }
    }
    expect(offenders, `Tofu glyph(s) in a rendered template:\n  ${offenders.join("\n  ")}`).toEqual(
      []
    )
  })
})
