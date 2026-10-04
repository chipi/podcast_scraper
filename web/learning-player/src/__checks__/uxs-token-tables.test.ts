import { existsSync, readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"

/**
 * UXS-011's token tables are a contract, so they are checked against the file they describe (#2280).
 *
 * The 2026-09-13 design-system review found the tables had drifted from `tokens.css` while every
 * other rule in this app held: `primary` / `primary-foreground` and `insight` were tabled but never
 * existed, and the implementation paths named files that were never created. A month later the
 * `theme` row still carried the storyline's value after #2225 split the two. Every one of those was
 * a table nobody re-read after the code moved. "Token names are the frozen API" is only true if
 * something reads the table — this does.
 *
 * Scope is deliberately narrow: the colour tables under "Semantic color tokens", against the
 * default `:root` palette, plus the `web/learning-player/...` and `src/...` paths the spec names.
 * Directions repaint these tokens on purpose and are guarded by `directions.test.ts`.
 *
 * If this fails, fix the code OR amend the table — never delete the check (same rule as
 * `spec-conformance.test.ts`).
 */

const APP = resolve(__dirname, "..", "..")
const REPO = resolve(APP, "..", "..")
const SPEC = readFileSync(resolve(REPO, "docs", "uxs", "UXS-011-consumer-learning-app.md"), "utf8")
const TOKENS = readFileSync(resolve(APP, "src", "theme", "tokens.css"), "utf8")

/** `--lp-*` declarations of the default palette: the first `:root` block, comments stripped. */
function rootTokens(): Map<string, string> {
  const css = TOKENS.replace(/\/\*[\s\S]*?\*\//g, "")
  const root = css.slice(css.indexOf(":root"))
  const block = root.slice(root.indexOf("{") + 1, root.indexOf("}"))
  const out = new Map<string, string>()
  for (const m of block.matchAll(/--lp-([a-z0-9-]+)\s*:\s*([^;]+);/g)) out.set(m[1], m[2].trim())
  return out
}

/** `| \`name\` | \`value\` | usage |` rows between "Semantic color tokens" and "Typography". */
function tableRows(): { name: string; value: string }[] {
  const start = SPEC.indexOf("## Semantic color tokens")
  const end = SPEC.indexOf("## Typography", start)
  const section = SPEC.slice(start, end)
  return [...section.matchAll(/^\|\s*`([a-z0-9-]+)`\s*\|\s*`([^`]+)`\s*\|/gm)].map((m) => ({
    name: m[1],
    value: m[2],
  }))
}

/**
 * One spelling per colour so notation never fails the check: lower case, no spaces, a leading zero
 * on fractional alpha, and the spec's prefix-free `var(--accent)` read as the `--lp-` it means.
 */
function normalise(v: string): string {
  return v
    .toLowerCase()
    .replace(/\s+/g, "")
    .replace(/([,(])\./g, "$10.")
    .replace(/var\(--(?!lp-)/g, "var(--lp-")
}

describe("UXS-011 token tables match tokens.css (#2280)", () => {
  const tokens = rootTokens()
  const rows = tableRows()

  it("finds the tables and the token layer it is meant to guard", () => {
    // A parser that silently matches nothing would pass every assertion below.
    expect(rows.length).toBeGreaterThan(15)
    expect(tokens.size).toBeGreaterThan(15)
  })

  it("every tabled token exists in the default palette", () => {
    const missing = rows.filter((r) => !tokens.has(r.name)).map((r) => r.name)
    expect(missing, `tabled in UXS-011 but not declared in tokens.css :root`).toEqual([])
  })

  it("every tabled value is the shipped default value", () => {
    const drifted = rows
      .filter((r) => tokens.has(r.name))
      .filter((r) => normalise(r.value) !== normalise(tokens.get(r.name) as string))
      .map((r) => `${r.name}: spec ${r.value} / code ${tokens.get(r.name)}`)
    expect(drifted, "UXS-011 table value differs from tokens.css :root").toEqual([])
  })

  it("every implementation path the spec names exists", () => {
    const named = new Set<string>()
    for (const m of SPEC.matchAll(/`(web\/learning-player\/[^`\s]+)`/g))
      named.add(resolve(REPO, m[1]))
    for (const m of SPEC.matchAll(/`(src\/[A-Za-z0-9_./-]+\.(?:ts|vue|css|js))`/g))
      named.add(resolve(APP, m[1]))
    const missing = [...named].filter((p) => !existsSync(p)).map((p) => p.slice(REPO.length + 1))
    expect(missing, "UXS-011 names a path that does not exist").toEqual([])
  })
})
