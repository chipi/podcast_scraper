import { readFileSync } from 'node:fs'
import { join } from 'node:path'
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
    // #1598 closed this: PlayerControls once hard-coded rgba(255,106,61,0.4) — Ember inlined — so
    // the play button's glow stayed orange regardless of the accent. It now derives from the token
    // (`color-mix(... var(--lp-accent) ...)`), and the per-show accent is live (theme/accent.ts).
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

describe('WCAG contrast cannot depend on scroll position', () => {
  /**
   * A fixed bar that overlays page content must be OPAQUE.
   *
   * The bottom nav shipped as `bg-canvas/95 backdrop-blur`. axe composites text against what is
   * actually behind an element, so with a tinted storyline chip scrolled underneath, both the
   * active label (`text-accent`) and the inactive labels (`text-muted`) measured 4.28:1 — under the
   * 4.5:1 AA floor. Conformance became a function of what the user had scrolled to, which surfaced
   * as an INTERMITTENT e2e failure: the kind that gets retried away rather than fixed.
   *
   * Scrims (`bg-black/40` behind a modal) are exempt: translucency is their purpose and they carry
   * no text of their own.
   */
  const BARS = ['BottomNav.vue', 'MiniPlayer.vue']

  it.each(BARS)('%s paints an opaque background', (name) => {
    const text = readFileSync(join(__dirname, '..', 'components', name), 'utf8')
    const bar = text.split('<template>')[1] ?? ''
    const translucent = bar.match(/class="[^"]*fixed[^"]*"/g)?.filter(
      (c) => /bg-[a-z-]+\/\d+/.test(c) || c.includes('backdrop-blur'),
    )
    expect(
      translucent ?? [],
      `${name} overlays page content, so a translucent background makes its text contrast depend ` +
        `on what is scrolled behind it. Use an opaque token.`,
    ).toEqual([])
  })
})

describe('reduced motion is honoured at every call site (S9)', () => {
  /**
   * `prefers-reduced-motion` is an accessibility setting — for users with vestibular disorders or
   * migraine triggers, an unexpected smooth scroll can cause actual nausea.
   *
   * The check existed inline in TranscriptList and nowhere else, so four other call sites animated
   * regardless. The advisor found three of them; this guard found a fourth (CardRail) that nobody
   * had looked at. That is the point: a literal `behavior: 'smooth'` IS the missed call site, so
   * make the literal the thing that fails.
   */
  it("no component hard-codes behavior: 'smooth'", () => {
    const offenders = Object.entries(components)
      .filter(([path]) => !path.includes('utils/motion'))
      .filter(([, text]) => /behavior:\s*'smooth'/.test(text ?? ''))
      .map(([path]) => path)

    expect(
      offenders,
      `Use scrollBehavior() from utils/motion instead — it returns 'auto' when the user has asked ` +
        `for reduced motion. A hard-coded 'smooth' animates regardless of that setting.`,
    ).toEqual([])
  })
})
