import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * UXS-001's colour tables are checked against tokens.css, in both directions (#2280).
 *
 * The tables were written three days before the viewer shipped and never reconciled. By the
 * 2026-09-13 design-system review six values disagreed — including `kg`, marked "Frozen: identity
 * colors, do not change without a UXS revision", which proved that "frozen" meant nothing
 * mechanical — and eleven tabled tokens did not exist at all. A frozen value nothing checks is not
 * frozen; this is the check.
 *
 * Both themes are compared: the Dark column against `:root[data-theme='dark']`, the Light column
 * against `:root[data-theme='light']` (the OS-preference copy is held equal to that one by
 * `token-contrast.test.ts`). And every `--ps-*` the dark palette defines must appear in a table,
 * so a token cannot be added to the code without being documented.
 */

const VIEWER = resolve(__dirname, '..', '..')
const SPEC = readFileSync(resolve(VIEWER, '..', '..', 'docs', 'uxs', 'UXS-001-gi-kg-viewer.md'), 'utf8')
const CSS = readFileSync(resolve(VIEWER, 'src', 'theme', 'tokens.css'), 'utf8').replace(
  /\/\*[\s\S]*?\*\//g,
  '',
)

function themeBlock(selector: string): Map<string, string> {
  const at = CSS.indexOf(selector)
  const body = CSS.slice(CSS.indexOf('{', at) + 1, CSS.indexOf('}', at))
  const out = new Map<string, string>()
  for (const m of body.matchAll(/--ps-([a-z0-9-]+)\s*:\s*([^;]+);/g)) out.set(m[1], m[2].trim())
  return out
}

/** `| \`name\` | \`dark\` | \`light\` | usage |` rows of the "Semantic color tokens" section. */
function tableRows(): { name: string; dark: string; light: string }[] {
  const start = SPEC.indexOf('## Semantic color tokens')
  const section = SPEC.slice(start, SPEC.indexOf('\n## ', start + 1))
  return [...section.matchAll(/^\|\s*`([a-z0-9-]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|/gm)].map(
    (m) => ({ name: m[1], dark: m[2], light: m[3] }),
  )
}

/** Case and spacing never fail the check; the spec's prefix-free `var(--gi)` means `--ps-gi`. */
const normalise = (v: string) =>
  v.toLowerCase().replace(/\s+/g, '').replace(/var\(--(?!ps-)/g, 'var(--ps-')

describe('UXS-001 token tables match tokens.css (#2280)', () => {
  const dark = themeBlock(":root[data-theme='dark']")
  const light = themeBlock(":root[data-theme='light']")
  const rows = tableRows()

  it('finds the tables and both palettes it is meant to guard', () => {
    expect(rows.length).toBeGreaterThan(20)
    expect(dark.size).toBeGreaterThan(20)
    expect(light.size).toBeGreaterThan(20)
  })

  it('every tabled token is defined in both themes', () => {
    const missing = rows
      .filter((r) => !dark.has(r.name) || !light.has(r.name))
      .map((r) => r.name)
    expect(missing, 'tabled in UXS-001 but not defined in tokens.css').toEqual([])
  })

  it('every tabled value is the shipped value, dark and light', () => {
    const drifted = rows
      .filter((r) => dark.has(r.name) && light.has(r.name))
      .flatMap((r) => [
        normalise(r.dark) === normalise(dark.get(r.name)!)
          ? []
          : [`${r.name} (dark): spec ${r.dark} / code ${dark.get(r.name)}`],
        normalise(r.light) === normalise(light.get(r.name)!)
          ? []
          : [`${r.name} (light): spec ${r.light} / code ${light.get(r.name)}`],
      ])
      .flat()
    expect(drifted, 'UXS-001 table value differs from tokens.css').toEqual([])
  })

  it('every token tokens.css defines is in a table', () => {
    const tabled = new Set(rows.map((r) => r.name))
    const undocumented = [...dark.keys()].filter((k) => !tabled.has(k))
    expect(undocumented, 'defined in tokens.css but missing from UXS-001').toEqual([])
  })
})
