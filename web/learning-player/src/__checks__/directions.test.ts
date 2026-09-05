import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * A visual direction must repaint EVERY ground-dependent colour, not just the ones it thought about.
 *
 * ## The bug this exists to prevent
 *
 * A direction is a block of `--lp-*` value overrides (`theme/directions.css`). Anything it does not
 * override keeps the shipping value — and the shipping values were chosen for a near-black ground.
 * So a light direction that forgets one token does not fail, does not warn, and does not look
 * obviously wrong in the one screenshot anybody checks: it just renders that one element at a
 * contrast nobody can read.
 *
 * That is not hypothetical. All four directions omitted `--lp-person` and `--lp-theme`. Under
 * `paper` — bone-white ground — the inherited person orange `#ffb37a` landed on `#e5e0d7` at
 * **1.33:1** against a 4.5:1 requirement, and the a11y specs produced 269 contrast violations
 * across the suite. `terminal` had exactly the same omission and passed everything, because it is
 * also dark and the inherited values happened to still work. The difference between the two was
 * luck, and luck is not a property you can build eight more directions on.
 *
 * ## Why a static check rather than an a11y run
 *
 * The a11y specs DO catch it — after a full browser suite, per direction, in minutes, reported as
 * hundreds of nodes that have to be traced back to one missing line. Phase 3 authors directions in
 * batches. The feedback has to arrive in milliseconds and name the token, or authors will keep
 * paying that cost.
 */

const CSS = readFileSync(resolve(__dirname, '..', 'theme', 'directions.css'), 'utf8')
const TOKENS = readFileSync(resolve(__dirname, '..', 'theme', 'tokens.css'), 'utf8')

/**
 * Tokens a direction may leave alone, and why each one is safe.
 *
 * `accent` and `link` are ALIASES (`--lp-accent: var(--lp-brand-default)`, `--lp-link:
 * var(--lp-accent)`), so they follow the brand colour a direction already set — overriding them
 * would be the mistake. The three fonts are not ground-dependent: keeping the shipping typeface is
 * a legitimate thing for a direction to do, and a direction that wants a new one says so.
 */
const MAY_INHERIT = new Set([
  '--lp-accent',
  '--lp-link',
  '--lp-font-display',
  '--lp-font-ui',
  '--lp-font-mono',
])

/** Posture tokens are opt-in: a direction that only repaints is a valid direction. */
const POSTURE = new Set(['--lp-radius', '--lp-density', '--lp-motion'])

function tokensIn(block: string): Set<string> {
  return new Set(Array.from(block.matchAll(/(--lp-[a-z-]+)\s*:/g), (m) => m[1]))
}

/** Every `:root[data-direction='<name>'] { … }` block, by name. */
function directionBlocks(): Array<[string, string]> {
  return Array.from(
    CSS.matchAll(/:root\[data-direction='([a-z-]+)'\]\s*\{([^}]*)\}/g),
    (m) => [m[1], m[2]] as [string, string],
  )
}

describe('visual directions are complete', () => {
  const blocks = directionBlocks()
  const required = [...tokensIn(TOKENS)].filter((t) => !MAY_INHERIT.has(t) && !POSTURE.has(t))

  it('finds the directions and the token contract it is meant to guard', () => {
    // A guard that matches nothing passes vacuously, which is worse than no guard: it reads as
    // "the directions are complete" on every future run.
    expect(blocks.length, 'expected direction blocks in theme/directions.css').toBeGreaterThan(1)
    expect(required.length, 'expected ground-dependent tokens in theme/tokens.css').toBeGreaterThan(
      10,
    )
    // The aliases must actually BE aliases — if `--lp-accent` ever became a literal, exempting it
    // above would start hiding the exact bug this file is about.
    expect(TOKENS).toMatch(/--lp-accent:\s*var\(--lp-brand-default\)/)
    expect(TOKENS).toMatch(/--lp-link:\s*var\(--lp-accent\)/)
  })

  it.each(blocks)('direction "%s" overrides every ground-dependent token', (_name, block) => {
    const have = tokensIn(block)
    const missing = required.filter((t) => !have.has(t))
    expect(
      missing,
      `these keep a value chosen for the SHIPPING dark ground, which is only safe by accident: ${missing.join(', ')}`,
    ).toEqual([])
  })

  it.each(blocks)('direction "%s" invents no token the app does not read', (_name, block) => {
    // The other direction of the same mistake: a typo (`--lp-bordr`) is inert CSS. Nothing renders
    // differently, nothing errors, and the direction quietly keeps the shipping border colour.
    const known = tokensIn(TOKENS)
    const unknown = [...tokensIn(block)].filter((t) => !known.has(t) && !POSTURE.has(t))
    expect(unknown, `not defined in tokens.css — a typo here is silent: ${unknown.join(', ')}`)
      .toEqual([])
  })
})
