import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Text-bearing token pairs meet WCAG 2.1 AA (4.5:1) in every theme (UXS-001, #2280).
 *
 * UXS-001 targets AA and says the `-foreground` convention exists so contrast is validated at the
 * token level. Until this test nothing validated it, and the 2026-09-13 design-system review found
 * the one place it had slipped: the dark theme — the declared baseline — shipped white on
 * `--ps-primary: #4c90f0`, 3.20:1, on every primary button. The spec's own value passed; the code's
 * did not, and a narrow, specific gap like that survives exactly because nothing measures it.
 *
 * `disabled` is deliberately not checked: WCAG 1.4.3 exempts inactive controls, and the token is
 * tuned to read as unavailable.
 */

const CSS = readFileSync(resolve(__dirname, '..', 'theme', 'tokens.css'), 'utf8').replace(
  /\/\*[\s\S]*?\*\//g,
  '',
)

/** `--ps-*` declarations of the first rule whose selector list contains `selector`. */
function themeBlock(selector: string): Map<string, string> {
  const at = CSS.indexOf(selector)
  if (at < 0) throw new Error(`tokens.css has no ${selector} block`)
  const body = CSS.slice(CSS.indexOf('{', at) + 1, CSS.indexOf('}', at))
  const out = new Map<string, string>()
  for (const m of body.matchAll(/--ps-([a-z0-9-]+)\s*:\s*([^;]+);/g)) out.set(m[1], m[2].trim())
  return out
}

/** WCAG 2.1 relative luminance of a `#rrggbb` colour. */
function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}

const THEMES = {
  dark: themeBlock(":root[data-theme='dark']"),
  light: themeBlock(":root[data-theme='light']"),
  'light (OS preference)': themeBlock(':root:not([data-theme])'),
}

/** Text tokens that are not `-foreground` pairs but are read on the shell backgrounds. */
const TEXT_ON = {
  muted: ['canvas', 'surface', 'elevated'],
  link: ['canvas', 'surface', 'elevated'],
  // Small uppercase label text in the Details panels. Hard-coded until #2280, when the light theme
  // measured 1.76:1 (theme) and 2.66:1 (related-topic) on white.
  theme: ['canvas', 'surface', 'elevated', 'overlay'],
  'related-topic': ['canvas', 'surface', 'elevated', 'overlay'],
}

describe('viewer token contrast meets WCAG AA (#2280)', () => {
  for (const [theme, t] of Object.entries(THEMES)) {
    it(`${theme}: every -foreground reads on its surface at 4.5:1`, () => {
      const pairs = [...t.keys()].filter((k) => k.endsWith('-foreground') && t.has(k.slice(0, -11)))
      expect(pairs.length, 'found no foreground pairs — the parser is broken').toBeGreaterThan(3)
      const failing = pairs
        .map((fg) => ({ fg, bg: fg.slice(0, -11) }))
        .map(({ fg, bg }) => ({ fg, bg, ratio: contrast(t.get(fg)!, t.get(bg)!) }))
        .filter(({ ratio }) => ratio < 4.5)
        .map(({ fg, bg, ratio }) => `${fg} on ${bg}: ${ratio.toFixed(2)}:1`)
      expect(failing).toEqual([])
    })

    it(`${theme}: muted and link text read on canvas, surface and elevated at 4.5:1`, () => {
      const failing = Object.entries(TEXT_ON)
        .flatMap(([fg, bgs]) => bgs.map((bg) => ({ fg, bg, ratio: contrast(t.get(fg)!, t.get(bg)!) })))
        .filter(({ ratio }) => ratio < 4.5)
        .map(({ fg, bg, ratio }) => `${fg} on ${bg}: ${ratio.toFixed(2)}:1`)
      expect(failing).toEqual([])
    })
  }

  it('the OS-preference light palette is the explicit light palette', () => {
    // tokens.css repeats the light palette for `prefers-color-scheme`; a value edited in one copy
    // and not the other gives users a different theme depending on how they reached it.
    expect(Object.fromEntries(THEMES['light (OS preference)'])).toEqual(
      Object.fromEntries(THEMES.light),
    )
  })
})
