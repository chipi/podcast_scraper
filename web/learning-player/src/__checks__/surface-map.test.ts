import { describe, expect, it } from 'vitest'
import surfaceMap from '../../e2e/E2E_SURFACE_MAP.md?raw'
import routerSrc from '../router/index.ts?raw'

/**
 * Guardrail (#1606) — keeps `e2e/E2E_SURFACE_MAP.md` true.
 *
 * The map is the **baseline of business rules the e2e suite is reconstructed from**, so a stale row
 * doesn't just mislead a reader: a test rebuilt from it is wrong, and wrong *silently* — a renamed
 * `data-testid` yields a selector that matches nothing rather than an error anyone notices.
 *
 * The map asks contributors to update it in the same PR (E2E_SURFACE_MAP.md:26-28). That request
 * was not being met: an audit on 2026-08-12 found four routes, five spec files and two retired
 * testids out of sync. This check makes the three mechanically-verifiable parts of that promise
 * fail the build instead.
 *
 * It is a RATCHET, not a gate: the violations that existed when it landed are listed in
 * `KNOWN_GAPS` with the issue that fixes each. New violations fail immediately. Remove an entry as
 * its issue lands — never add one to make a red test green.
 *
 * Deliberately NOT checked: that every testid in `src/` appears in the map. Not every testid is a
 * contract, and a noisy gate gets disabled.
 */

// --- sources -----------------------------------------------------------------

const components = import.meta.glob('../**/*.vue', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

const specFiles = Object.keys(
  import.meta.glob('../../e2e/*.spec.ts', { query: '?raw', import: 'default', eager: true }),
).map((p) => p.split('/').pop() as string)

const allComponentSrc = Object.values(components).join('\n')

// --- known gaps at the time this check landed --------------------------------
// Each entry is a violation the 2026-08-12 audit found. Remove entries as the issues land.

const KNOWN_GAPS = {
  /** Routes absent from the map. Empty — the `#1261-6` routes were added in #1609. */
  routes: [] as string[],
  /** Dead selectors in the map. Empty — `es-disagreements` corrected to `es-consensus` in #1608. */
  testids: [] as string[],
  /** Spec files the map never mentions. Empty — all five were added in #1609. */
  specs: [] as string[],
} as const

// Attribute names that look like testids in the map's prose but aren't.
const NOT_TESTIDS = new Set(['data-testid', 'aria-pressed', 'aria-expanded', 'aria-modal', 'aria-label'])

// --- helpers -----------------------------------------------------------------

/** Route names declared in the router. */
function routerRouteNames(): string[] {
  return [...new Set([...routerSrc.matchAll(/name:\s*'([a-z-]+)'/g)].map((m) => m[1]))].sort()
}

/**
 * Testids the map documents. Only tokens on a line that mentions "testid" count — the map also
 * backticks plenty of prose (component names, params) that is not a selector contract.
 */
function mapDocumentedTestids(): string[] {
  const found = new Set<string>()
  for (const line of surfaceMap.split('\n')) {
    if (!line.toLowerCase().includes('testid')) continue
    for (const m of line.matchAll(/`([a-z][a-z0-9]*(?:-[a-z0-9]+)+)`/g)) found.add(m[1])
    for (const m of line.matchAll(/data-testid="([a-z0-9-]+)"/g)) found.add(m[1])
  }
  for (const n of NOT_TESTIDS) found.delete(n)
  // A route row may carry both its route name and a testid on one line (e.g. the `#1261-6` rows).
  // Route names are not selectors — drop them rather than have the check report false positives.
  for (const name of routerRouteNames()) found.delete(name)
  return [...found].sort()
}

/** Static testids in components, plus prefixes of dynamically-built ones (`seg-${i}`). */
function componentTestids(): { exact: Set<string>; prefixes: string[] } {
  const raw = new Set<string>()
  for (const m of allComponentSrc.matchAll(/data-testid\s*=\s*"([^"]+)"/g)) raw.add(m[1])
  for (const m of allComponentSrc.matchAll(/data-testid\s*=\s*'([^']+)'/g)) raw.add(m[1])
  for (const m of allComponentSrc.matchAll(/data-testid\s*=\s*`([^`]+)`/g)) raw.add(m[1])
  // Bound expressions: `:data-testid="cond ? 'a-chip' : 'b-chip'"`. Pull every quoted literal out
  // of the expression — otherwise a conditionally-named testid reads as nonexistent and the map
  // row documenting it is reported as dead. (Found by this check failing on my own chips.)
  for (const m of allComponentSrc.matchAll(/:data-testid\s*=\s*"([^"]+)"/g)) {
    for (const lit of m[1].matchAll(/'([a-z0-9-]+)'/g)) raw.add(lit[1])
  }

  const exact = new Set<string>()
  const prefixes: string[] = []
  for (const id of raw) {
    const cleaned = id.replace(/`/g, '').trim()
    if (cleaned.includes('${')) {
      const prefix = cleaned.split('${')[0].replace(/-$/, '')
      if (prefix) prefixes.push(prefix)
    } else if (cleaned) {
      exact.add(cleaned)
    }
  }
  return { exact, prefixes }
}

// --- checks ------------------------------------------------------------------

describe('E2E surface map stays true to the app', () => {
  it('documents every route in the router', () => {
    const undocumented = routerRouteNames().filter(
      (name) => !surfaceMap.includes(`\`${name}\``) && !KNOWN_GAPS.routes.includes(name),
    )
    expect(
      undocumented,
      `Route(s) missing from e2e/E2E_SURFACE_MAP.md. Add a row to the routes table, or the ` +
        `feature is invisible to anyone rebuilding the suite from the map.`,
    ).toEqual([])
  })

  it('only documents data-testids that exist in the app', () => {
    const { exact, prefixes } = componentTestids()
    const dead = mapDocumentedTestids().filter(
      (id) =>
        !exact.has(id) &&
        !prefixes.some((p) => id.startsWith(p)) &&
        !KNOWN_GAPS.testids.includes(id as (typeof KNOWN_GAPS.testids)[number]),
    )
    expect(
      dead,
      `The map documents data-testid(s) that no component renders. A test rebuilt from these ` +
        `selectors matches nothing and fails for a reason that looks like a product bug. ` +
        `Update the map row, or restore the testid.`,
    ).toEqual([])
  })

  it('accounts for every e2e spec file', () => {
    const unclaimed = specFiles.filter(
      (f) =>
        !surfaceMap.includes(f) &&
        !surfaceMap.includes(f.replace('.spec.ts', '')) &&
        !KNOWN_GAPS.specs.includes(f as (typeof KNOWN_GAPS.specs)[number]),
    )
    expect(
      unclaimed,
      `Spec file(s) not referenced anywhere in the map. Either add the surface they cover, or ` +
        `list them under "coverage gaps" — silence reads as "covered".`,
    ).toEqual([])
  })

  it('documents every testid the e2e specs actually depend on', () => {
    // The three checks above run map → code. That direction alone has a hole, and I fell into it
    // within hours of writing them: five new contract selectors (section-*, player-open-insights,
    // yourweek-firstrun) were added to the app and the specs without reaching the map.
    //
    // Asserting that EVERY testid in src/ is documented would be noisy — not every testid is a
    // contract. But one an e2e spec selects on demonstrably is: that is the definition of a
    // selector the suite depends on, and the map exists to let the suite be rebuilt.
    const specSrc = Object.values(
      import.meta.glob('../../e2e/*.spec.ts', { query: '?raw', import: 'default', eager: true }),
    ).join('\n') as string

    const used = new Set<string>()
    for (const m of specSrc.matchAll(/getByTestId\(\s*['"`]([a-z0-9-]+)['"`]/g)) used.add(m[1])
    for (const m of specSrc.matchAll(/\[data-testid="([a-z0-9-]+)"\]/g)) used.add(m[1])

    // The map documents dynamic ids as templates — `momentum-rail-{kind}` covers
    // `momentum-rail-topic`. Honour that rather than forcing every concrete value to be listed,
    // which would be the noisy version of this check.
    const templatePrefixes = [...surfaceMap.matchAll(/`([a-z0-9-]+)-\{[a-z]+\}`/g)].map((m) => m[1])
    const documented = (id: string): boolean =>
      surfaceMap.includes(id) || templatePrefixes.some((p) => id.startsWith(`${p}-`))

    const undocumented = [...used].filter((id) => !documented(id)).sort()
    expect(
      undocumented,
      `An e2e spec selects on these data-testid(s), but the map does not document them. The map is ` +
        `the baseline the suite is reconstructed from, so a selector a spec depends on must appear ` +
        `in it — otherwise the rebuilt spec cannot be written.`,
    ).toEqual([])
  })

  it('has no stale entries in its own allowlist', () => {
    // Guards the ratchet: once an issue lands, its entry must go. A KNOWN_GAP that no longer
    // describes a real violation means someone fixed the app but left the exemption behind, which
    // would silently re-open the hole for a future regression.
    const routeNames = routerRouteNames()
    const staleRoutes = KNOWN_GAPS.routes.filter(
      (r) => !routeNames.includes(r) || surfaceMap.includes(`\`${r}\``),
    )
    const { exact, prefixes } = componentTestids()
    const staleTestids = KNOWN_GAPS.testids.filter(
      (id) => exact.has(id) || prefixes.some((p) => id.startsWith(p)),
    )
    const staleSpecs = KNOWN_GAPS.specs.filter(
      (f) => !specFiles.includes(f) || surfaceMap.includes(f),
    )

    expect(
      { staleRoutes, staleTestids, staleSpecs },
      'A KNOWN_GAPS entry no longer describes a real violation — delete it from the allowlist.',
    ).toEqual({ staleRoutes: [], staleTestids: [], staleSpecs: [] })
  })
})
