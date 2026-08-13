import { describe, expect, it } from 'vitest'

/**
 * Executable acceptance criteria (#1604).
 *
 * The spec-drift audit found ~60 divergences, and its central lesson was that **social rules do not
 * hold**: `UXS-011` and `UXS-014` both carry "living document" instructions that were not followed
 * for months, and in two cases a test comment became the only record of a decision.
 *
 * Most spec claims cannot be checked mechanically. A few can, and every one converted is one that
 * cannot rot. This file holds that subset — deliberately small, so it stays trusted. The rest stays
 * human-verified; see #1604 for the social half (PR checklist, status honesty in front-matter, never
 * recording a decision only in a test comment).
 *
 * If a check here starts failing, the fix is the code OR an explicit spec amendment — not deleting
 * the check.
 */

const components = import.meta.glob('../**/*.vue', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

/** Strip comments — a rule must not fail on prose explaining the rule. */
const strip = (s: string): string =>
  s.replace(/<!--[\s\S]*?-->/g, '').replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '')

describe('UXS-011 acceptance criteria that can be executed', () => {
  it('no one-off hex in components — the single token layer holds (:329)', () => {
    // Violated in spirit until #1598: PlayerControls hard-coded rgba(255,106,61,0.4), which IS the
    // Ember brand colour, so the play button's glow would have stayed orange even if the per-show
    // accent had shipped.
    const offenders: string[] = []
    for (const [path, src] of Object.entries(components)) {
      const body = strip(src)
      // Hex literals and rgb()/rgba() with concrete channel numbers. Allowed: var(--lp-*),
      // color-mix over tokens, and rgba over a var.
      const hex = [...body.matchAll(/#[0-9a-fA-F]{3,8}\b/g)].map((m) => m[0])
      // Only CHROMATIC rgb()/rgba() counts. Greyscale — rgba(0,0,0,.5) scrims over artwork,
      // rgba(255,255,255,.2) hairlines — is not a theme colour and has no token to use instead;
      // flagging it would make the rule noisy and get it deleted. A brand colour is never
      // greyscale, so the Ember glow this rule exists to catch still fails.
      const rgb = [...body.matchAll(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/g)]
        .filter((m) => !(m[1] === m[2] && m[2] === m[3]))
        .map((m) => m[0])
      // hsl() is used for deliberately generated fallback artwork gradients, not for theme colour.
      if (hex.length || rgb.length) {
        offenders.push(`${path.replace('../', '')}: ${[...hex, ...rgb].slice(0, 3).join(', ')}`)
      }
    }
    expect(
      offenders,
      'Components must use semantic tokens (var(--lp-*)), not literal colours. Use color-mix over ' +
        'a token if you need transparency.',
    ).toEqual([])
  })

})

describe('UXS-014 pattern rules that can be executed', () => {
  it('show names only clamp inside fixed-width tiles (:70, scoped #1604)', () => {
    // The rule holds where width is elastic. In a fixed-width grid/rail tile it cannot: a freely
    // wrapping name makes the row as tall as its longest member, which is the #1584 defect. Those
    // tiles may clamp WITH a reserved height and a title attribute.
    //
    // These are the tiles. Everything else must let a show name wrap.
    const FIXED_WIDTH_TILES = [
      'components/ShowTile.vue', // reserved 2-line box + title attribute
      'components/TrendingShowsRail.vue', // rail slice; its comment notes "truncates only at the edge"
      'components/EntityCardBody.vue', // "Host of" chips
      'views/HomeView.vue', // Recommended grid kicker — clamped for the same reserved-height reason
    ]
    const KNOWN_VIOLATIONS = FIXED_WIDTH_TILES
    const offenders: string[] = []
    for (const [path, src] of Object.entries(components)) {
      const rel = path.replace('../', '')
      if (KNOWN_VIOLATIONS.includes(rel)) continue
      const body = strip(src)
      // A truncate class on the same element as a show/podcast title binding.
      for (const line of body.split('\n')) {
        if (!line.includes('truncate')) continue
        if (/podcast_title|show\.title|feed_title/.test(line)) offenders.push(`${rel}: ${line.trim().slice(0, 90)}`)
      }
    }
    expect(
      offenders,
      'UXS-014:70 says a show name wraps rather than truncating. Fix the component, or amend the ' +
        'spec and add it to KNOWN_VIOLATIONS with the reason.',
    ).toEqual([])
  })
})
