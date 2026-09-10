import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * Read a project file regardless of where vitest was invoked from.
 *
 * `process.cwd()` is the app directory for `npm test` but the REPO ROOT for
 * `vitest --root web/learning-player`, which is how it runs from a script at the top level. A
 * bare `resolve(process.cwd(), 'src/…')` therefore throws ENOENT in exactly the invocation a
 * CI wrapper is most likely to use — the check does not report a violation, it fails to run.
 */
function projectFile(rel: string): string {
  for (const base of ['.', 'web/learning-player']) {
    const candidate = resolve(process.cwd(), base, rel)
    if (existsSync(candidate)) return candidate
  }
  throw new Error(`cannot locate ${rel} from ${process.cwd()}`)
}

/**
 * Every bottom sheet uses ONE canonical geometry (#2004 follow-up).
 *
 * ## The bug
 *
 * Four components — storyline, entity, queue, interests — each carried their own copy of the same
 * sheet utility string, and every copy set only a `max-height`. So a sheet was as tall as whatever
 * it happened to hold: a storyline with four topics opened as a 418px strip across the bottom half
 * of an 844px phone, while a long one opened nearly full-screen. Same control, same gesture,
 * different resting place — measured at top=426 before the fix and top=253 after.
 *
 * ## Why a source check
 *
 * The failure is a missing CSS property, which jsdom will not reveal: `getBoundingClientRect` is
 * all zeros there and no unit test can see a sheet's real height. The thing that actually has to
 * hold is that the geometry lives in exactly one place — that IS checkable in the source, and it
 * is what drifted.
 */

const vue = import.meta.glob('../**/*.vue', { query: '?raw', import: 'default', eager: true }) as Record<
  string,
  string
>

/** Strip comments — this file's own explanation and the CSS docblock both quote the old string. */
function code(src: string): string {
  return src.replace(/<!--[\s\S]*?-->/g, '').replace(/\/\*[\s\S]*?\*\//g, '')
}

// Read from disk rather than importing: vitest stubs CSS imports to an empty module, so
// `import '../style.css?raw'` silently yields nothing and every rule below would pass vacuously.
const css = code(readFileSync(projectFile('src/style.css'), 'utf8'))
const components = Object.entries(vue).map(([p, src]) => [p, code(src)] as const)

const SHEETS = ['EntityCard', 'QueuePanel', 'InterestsPicker']

describe('sheet geometry is defined once', () => {
  it('.lp-sheet sets a MIN height, not only a max', () => {
    // The whole bug: with a max alone the top edge lands wherever the content ends, so you cannot
    // learn where an opened sheet will be.
    const rule = css.match(/\.lp-sheet\s*\{[^}]*\}/)?.[0] ?? ''
    expect(rule, '.lp-sheet is not defined').not.toBe('')
    expect(rule, '.lp-sheet has no min-height — sheets go back to content height').toContain('min-height')
    expect(rule).toContain('max-height')
  })

  it('the phone floor is released on the centred (sm:) dialog', () => {
    // Above sm: the sheet is centred; a floor there leaves a four-row list in a tall empty box.
    expect(css, 'no sm: override releasing the min-height').toMatch(
      /@media\s*\(min-width:\s*640px\)\s*\{\s*\.lp-sheet\s*\{[^}]*min-height:\s*0/,
    )
  })

  it('no component hand-writes the sheet scrim', () => {
    const offenders = components
      .filter(([, src]) => /fixed inset-0 z-50 flex items-end/.test(src))
      .map(([p]) => p)
    expect(offenders, 'use the .lp-sheet-scrim class instead of re-spelling the utilities').toEqual([])
  })

  it('no SHEET hand-writes its own height', () => {
    // Scoped to files that actually open a sheet. PlayerView's docked transcript pane and its
    // summary dialog also use dvh and are neither bottom sheets nor drifting — a rule broad enough
    // to catch them would be policing layout in general, which is guesswork.
    const offenders = components
      .filter(([, src]) => src.includes('lp-sheet-scrim') && /max-h-\[\d+dvh\]/.test(src))
      .map(([p]) => p)
    expect(offenders, 'heights belong in .lp-sheet, which is the thing that drifted').toEqual([])
  })

  it('each known sheet actually uses the canonical classes', () => {
    // Guards the checks above from passing because the sheets were deleted or renamed away.
    for (const name of SHEETS) {
      const entry = components.find(([p]) => p.endsWith(`/${name}.vue`))
      expect(entry, `${name}.vue not found`).toBeTruthy()
      expect(entry![1], `${name} lost .lp-sheet-scrim`).toContain('lp-sheet-scrim')
      expect(entry![1], `${name} lost .lp-sheet`).toMatch(/class="lp-sheet[ "]/)
    }
  })

  it('the sweep reads real files — it is not passing on an empty glob', () => {
    expect(Object.keys(vue).length, 'the component glob found nothing').toBeGreaterThan(40)
    expect(css.length, 'the stylesheet read found nothing').toBeGreaterThan(500)
  })
})
