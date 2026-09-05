import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * The accent means "you can act on this", and nothing else.
 *
 * ## Why this is a test and not a convention
 *
 * A blind design critic reviewing this app named one finding above all others: *"the accent has
 * lost its meaning. Orange appears ~60 times in one scroll — every show kicker, every insights
 * chip, the tab underline, the active nav icon. When everything is accented, nothing is."* The
 * measured count was 158 `*-accent` usages across 44 files, of which 30 were decorative.
 *
 * That was fixed once already, narrowly, and it grew back. `#1963` correctly diagnosed the same
 * problem in dense lists and added a `.lp-kicker--muted` modifier for repeating rows — which fixed
 * three call sites and left the other 52 alone, because `.lp-kicker` itself was still defined as
 * `color: var(--lp-accent)`. A rule that has to be remembered at 55 call sites is not a rule.
 *
 * So the invariant is enforced where it can actually be enforced: in `style.css`, which is the only
 * place a single declaration can colour the whole app at once.
 *
 * ## What "interactive" means here
 *
 * Not "is a `<button>`". An insights chip, a "distinctive topic" pill and a "downloaded" badge are
 * all real buttons whose accent was signalling INFORMATION rather than affordance. The test is
 * status-vs-toggle: a control that reports a fact is not an action, even when it is clickable.
 * Focus rings are exempt in the other direction — `:focus-visible` is an accessibility contract,
 * not decoration, and must keep the accent.
 */

const RAW = readFileSync(resolve(__dirname, '..', 'style.css'), 'utf8')

/**
 * Comments are stripped before parsing. Without this the "selector" captured for a failing rule is
 * the entire doc-comment preceding it, and the assertion message — the only thing a future reader
 * sees when this fires — becomes an unreadable wall. The rules this file guards are heavily
 * commented by design, so that is not a hypothetical.
 */
const CSS = RAW.replace(/\/\*[\s\S]*?\*\//g, '')

/**
 * The complete set of selectors permitted to spend the accent, each with the reason it qualifies.
 * Adding an entry here is a deliberate act; adding one to `style.css` without touching this list
 * fails, which is the point.
 */
const MAY_SPEND_ACCENT: Array<[RegExp, string]> = [
  [/:focus-visible/, 'focus ring — an accessibility contract, not decoration'],
  [/\[aria-selected='true'\]/, 'active state of an exclusive-choice control'],
  [/\.lp-fav:hover/, 'hover state of an action'],
  [/\.lp-fav--on/, 'pressed/active state of a toggle'],
  [/\.lp-check:checked/, 'checked state of a checkbox — a control, and this is its ON state'],
]

/** Every rule in `style.css` that references the accent, as `selector { …decls… }` pairs. */
function accentRules(): Array<{ selector: string; body: string }> {
  const out: Array<{ selector: string; body: string }> = []
  for (const m of CSS.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    const body = m[2]
    if (/var\(--lp-accent\)/.test(body)) out.push({ selector: m[1].trim(), body })
  }
  return out
}

describe('accent discipline (#2013)', () => {
  it('finds the stylesheet it is meant to guard', () => {
    // A guard that matches nothing passes vacuously and reads as "the rule holds" forever after.
    expect(CSS.length, 'style.css should not be empty').toBeGreaterThan(1000)
    expect(accentRules().length, 'expected some rules to legitimately use the accent').toBeGreaterThan(1)
  })

  it('is spent only by selectors that represent an action, a focus ring, or an active state', () => {
    const offenders = accentRules()
      .filter(({ selector }) => !MAY_SPEND_ACCENT.some(([re]) => re.test(selector)))
      .map(({ selector }) => selector.replace(/\s+/g, ' '))
    expect(
      offenders,
      `these colour something with --lp-accent that a finger cannot act on. If one of them IS an ` +
        `action, add it to MAY_SPEND_ACCENT with its reason; do not delete this assertion: ${offenders.join(' | ')}`,
    ).toEqual([])
  })

  it('keeps the kicker in the instrument voice, not the accent', () => {
    // The specific regression this file exists for. `.lp-kicker` is used 55 times across 25
    // components; it is a label, and you cannot tap a label.
    const kicker = accentRules().find(({ selector }) => /^\.lp-kicker$/.test(selector.trim()))
    expect(kicker, '.lp-kicker must not use the accent — it is a label, not a control').toBeUndefined()
    expect(CSS).toMatch(/\.lp-kicker\s*\{[^}]*font-family:\s*var\(--lp-font-mono\)/)
    expect(CSS).toMatch(/\.lp-kicker\s*\{[^}]*color:\s*var\(--lp-muted\)/)
  })

  it('has no no-op modifier restating a base rule', () => {
    // `.lp-kicker--muted` set the colour the base rule already had once the base went muted. A class
    // that changes nothing is how dead code is born, so it was retired rather than left as a
    // harmless-looking alias for future readers to propagate.
    expect(CSS).not.toMatch(/\.lp-kicker--muted\s*\{/)
  })
})
