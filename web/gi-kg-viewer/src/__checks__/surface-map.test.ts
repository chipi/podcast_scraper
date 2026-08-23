import { describe, expect, it } from 'vitest'
import surfaceMap from '../../e2e/E2E_SURFACE_MAP.md?raw'
import appSrc from '../App.vue?raw'

/**
 * Guardrail (#1606) — keeps `e2e/E2E_SURFACE_MAP.md` true. Sibling of the same check in
 * `web/learning-player/src/__checks__/surface-map.test.ts`.
 *
 * The map is the **baseline of business rules the e2e suite is reconstructed from**, so a stale row
 * doesn't merely mislead a reader: a test rebuilt from it is wrong, and wrong *silently* — a retired
 * `data-testid` yields a selector that matches nothing rather than an error anyone notices.
 *
 * An audit on 2026-08-12 found this file citing a retired Explore surface in six places, documenting
 * FR5.3 buttons whose unit tests assert their absence, naming a spec file that does not exist, and
 * omitting the auth gate that 36 of 38 specs depend on to boot.
 *
 * RATCHET, not a gate: violations present when this landed are listed in `KNOWN_GAPS` with the issue
 * that fixes each. New violations fail immediately. Remove an entry as its issue lands — never add
 * one to turn a red test green.
 */

const components = import.meta.glob('../**/*.vue', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

/**
 * Top-level specs only. The `handoff/`, `handoff-production/`, `perf/`, `validation/` and `live/`
 * suites are documented **collectively by directory** in the Spec inventory (they have their own
 * configs and boot requirements), so checking those by filename would be the wrong contract — the
 * directory check below covers them instead.
 */
const specPaths = Object.keys(
  import.meta.glob('../../e2e/*.spec.ts', { query: '?raw', import: 'default', eager: true }),
)

const SPEC_DIRECTORIES = ['handoff', 'handoff-production', 'perf', 'validation', 'live'] as const

/**
 * Non-test `.ts` sources too: some testids are exported constants (e.g.
 * `transcriptSourceDisplay.ts`) rather than template attributes, and reading only `.vue` reports
 * those as dead.
 */
const tsSources = import.meta.glob('../**/*.ts', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

const allComponentSrc = [
  ...Object.values(components),
  ...Object.entries(tsSources)
    .filter(([p]) => !p.endsWith('.test.ts') && !p.includes('/__checks__/'))
    .map(([, src]) => src),
].join('\n')

const KNOWN_GAPS = {
  /**
   * Testids the map documents that no component renders.
   * Empty — the Explore citations and the FR5.3 profile buttons were corrected in #1614.
   */
  testids: [] as string[],
  /**
   * Spec files the map never references. The handoff/perf/validation/live suites are now described
   * collectively in the "Spec inventory" section rather than file-by-file, which the filename check
   * below cannot see, so they stay listed here deliberately. Tracked in #1616.
   */
  specs: [] as string[],
} as const

/** Attribute names that look like testids in prose but aren't. */
const NOT_TESTIDS = new Set([
  'data-testid',
  'aria-pressed',
  'aria-expanded',
  'aria-modal',
  'aria-label',
  'aria-selected',
  'aria-controls',
])

/** Main-tab names declared in App.vue's `mainTab` union — every one must be documented. */
function mainTabNames(): string[] {
  const m = appSrc.match(/const mainTab = ref<([^>]+)>/)
  if (!m) return []
  return [...m[1].matchAll(/'([a-z]+)'/g)].map((x) => x[1]).sort()
}

/**
 * Testids the map documents. Only tokens on a line mentioning "testid" count — the map backticks a
 * great deal of prose (component names, params, copy) that is not a selector contract.
 *
 * Struck-through rows (`~~...~~`) are retirement notes: they document that a selector is GONE, so
 * they must not be read as claims that it exists.
 */
function mapDocumentedTestids(): string[] {
  const found = new Set<string>()
  // Only the EXPLICIT `data-testid="x"` form counts. This map's rows are single paragraphs of
  // 3,000+ characters mixing CSS classes (`bg-elevated`), spec filenames (`auth-roles.spec.ts`) and
  // selectors, so the sibling check's looser "any backticked token on a testid line" heuristic
  // produces dozens of false positives here. The explicit form is what documented every real
  // violation the audit found (the Explore chip bar, the FR5.3 profile buttons), so precision costs
  // nothing that matters.
  const body = surfaceMap.replace(/~~[^~]*~~/g, '') // drop retirement notes — those say "gone"
  for (const m of body.matchAll(/data-testid="([a-z0-9-]+)"/g)) found.add(m[1])
  for (const n of NOT_TESTIDS) found.delete(n)
  return [...found].sort()
}

/** Static testids in components, plus prefixes of dynamically-built ones. */
function componentTestids(): { exact: Set<string>; prefixes: string[] } {
  const raw = new Set<string>()
  // This viewer threads testids through PROPS as well as literal attributes — `close-testid="…"`,
  // `chip-testid="…"`, `search-testid="…"` — so match any attribute or key ending in "testid",
  // not just `data-testid`. Missing this reports ~60 live selectors as dead.
  for (const m of allComponentSrc.matchAll(/[\w:-]*testid\s*=\s*"([^"]+)"/gi)) raw.add(m[1])
  for (const m of allComponentSrc.matchAll(/[\w:-]*testid\s*=\s*'([^']+)'/gi)) raw.add(m[1])
  for (const m of allComponentSrc.matchAll(/[\w:-]*testid\s*=\s*`([^`]+)`/gi)) raw.add(m[1])
  // Object-literal form: `testid: 'foo'` / `testId: \`foo-${x}\`` in setup or config arrays.
  for (const m of allComponentSrc.matchAll(/testid\s*:\s*['"`]([^'"`]+)['"`]/gi)) raw.add(m[1])
  // Exported constants: `export const FOO_TESTID = 'foo-bar'` (transcriptSourceDisplay.ts et al).
  for (const m of allComponentSrc.matchAll(/TESTID\s*(?::[^=]+)?=\s*\n?\s*['"`]([^'"`]+)['"`]/g)) {
    raw.add(m[1])
  }
  // Computed testid helpers that return the id as a bare literal, e.g. NodeDetail.vue's
  // `if (isInsightNode.value) return 'node-detail-full-insight'`.
  for (const m of allComponentSrc.matchAll(/return\s+'([a-z][a-z0-9]*(?:-[a-z0-9]+)+)'/g)) {
    raw.add(m[1])
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

describe('E2E surface map stays true to the app', () => {
  it('documents every main tab, including the role-gated ones', () => {
    const undocumented = mainTabNames().filter((t) => !surfaceMap.includes(`main-tab-${t}`))
    expect(
      undocumented,
      `Main tab(s) missing from e2e/E2E_SURFACE_MAP.md. Ops and Admin were absent for months, ` +
        `along with the auth gate that decides whether they render at all.`,
    ).toEqual([])
  })

  it('only documents data-testids that exist in the app', () => {
    const { exact, prefixes } = componentTestids()
    const dead = mapDocumentedTestids().filter(
      (id) =>
        !exact.has(id) &&
        !prefixes.some((p) => id.startsWith(p)) &&
        !KNOWN_GAPS.testids.includes(id as never),
    )
    expect(
      dead,
      `The map documents data-testid(s) no component renders. A spec rebuilt from these selectors ` +
        `matches nothing and fails for a reason that looks like a product bug. Either correct the ` +
        `row, or mark it retired with ~~strikethrough~~ so it reads as history, not contract.`,
    ).toEqual([])
  })

  it('references every spec file it should account for', () => {
    const unclaimed = specPaths
      .map((p) => p.split('/').pop() as string)
      .filter(
        (f) =>
          !surfaceMap.includes(f) &&
          !surfaceMap.includes(f.replace('.spec.ts', '')) &&
          !KNOWN_GAPS.specs.includes(f as never),
      )
    expect(
      unclaimed,
      `Spec file(s) not referenced anywhere in the map. Add the surface they cover, or list them ` +
        `under "Coverage gaps" — silence reads as "covered".`,
    ).toEqual([])
  })

  it('accounts for every spec directory', () => {
    const undocumented = SPEC_DIRECTORIES.filter((d) => !surfaceMap.includes(`e2e/${d}/`))
    expect(
      undocumented,
      `Spec directory/ies not mentioned in the map. The handoff suites (20 files) were unlinked ` +
        `for months despite the graph matrix explicitly asking for them.`,
    ).toEqual([])
  })

  it('states the mockSignIn boot invariant', () => {
    // 36 of 38 specs call mockSignIn. Omitting it boots the app to <LoginView> and every
    // getByTestId times out — the ci-ui-full failure of 2026-07-18 (152/158 specs). The map carried
    // zero mentions of auth for months. Interim until #1619 removes the mocks entirely.
    expect(
      surfaceMap.includes('mockSignIn'),
      'The map must document the mockSignIn boot invariant, or a rebuilt suite fails wholesale.',
    ).toBe(true)
  })

  it('has no stale entries in its own allowlist', () => {
    const { exact, prefixes } = componentTestids()
    const staleTestids = KNOWN_GAPS.testids.filter(
      (id) => exact.has(id) || prefixes.some((p) => id.startsWith(p)),
    )
    const specNames = specPaths.map((p) => p.split('/').pop() as string)
    const staleSpecs = KNOWN_GAPS.specs.filter(
      (f) => !specNames.includes(f) || surfaceMap.includes(f),
    )
    expect(
      { staleTestids, staleSpecs },
      'A KNOWN_GAPS entry no longer describes a real violation — delete it from the allowlist.',
    ).toEqual({ staleTestids: [], staleSpecs: [] })
  })
})
