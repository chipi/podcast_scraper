import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The design rules, enforced against what the browser actually renders (#1946).
 *
 * ## Why this exists, and why it is not a pixel diff
 *
 * #1946 asks for visual-regression detection so "the next person to edit a redesigned surface gets
 * no warning if they degrade it" stops being true. The obvious answer is committed screenshot
 * baselines. That answer does not survive this repo: CI runs `ubuntu-latest` and development
 * happens on macOS, and the two rasterise type differently, so a baseline generated in one place
 * fails forever in the other. The usual workarounds — generate baselines inside a Linux container,
 * or run the diff only in CI — trade a real maintenance burden for coverage of a risk we can
 * attack more directly.
 *
 * The risk is not "some pixel moved". It is "the rules that make this design coherent got eroded
 * one commit at a time". Those rules are checkable as PROPERTIES of the rendered DOM, which is
 * platform-independent, fast, and fails with the offending element rather than a diff image a
 * human has to interpret.
 *
 * ## What this covers that the static guards cannot
 *
 * `src/__checks__/accent-discipline.test.ts` reads `style.css` and can only see rules written
 * there. Components apply the accent through Tailwind classes (`text-accent`, `bg-accent`) directly
 * in markup — 128 legitimate usages — and a 129th added to a label would be invisible to it. This
 * runs against the composed page, so it sees the accent wherever it came from.
 *
 * ## Deliberately NOT covered
 *
 * Layout and spacing drift. If someone doubles a margin or reorders a section, nothing here fails.
 * That is the residual risk of choosing invariants over pixels, and it is recorded on #1946 rather
 * than left implied.
 */

/**
 * Surfaces worth holding to the rules — the ones the redesign actually governs.
 *
 * The Player was missing from this list, which is the surface the redesign spent the most effort
 * on: Zone D, the insight panel, the NEXT row and the artwork treatment all live there and none of
 * them were covered. It cannot be reached by `goto` because its route is `/episode/:slug` and
 * hard-coding a slug makes the test date- and corpus-dependent, so it is reached by navigation
 * (see `openPlayer`) and handled alongside the rest rather than left out for being awkward.
 *
 * Catalog and Search are here for the same reason — both were redesigned, neither was checked.
 * Highlights needs no entry: it renders inside `/library` as a tab, so the library sweep reaches
 * it once the tab is open.
 */
const SURFACES: Array<[name: string, path: string]> = [
  ['home', '/'],
  ['browse', '/browse'],
  ['library', '/library'],
  ['profile', '/profile'],
  ['catalog', '/catalog'],
  ['search', '/search'],
]

/**
 * Open a real episode's Player, without pinning a slug.
 *
 * Via a show page rather than Home, because Home's hero is date-dependent and this test must not
 * start failing when the corpus rolls over (#1148 made the same fix elsewhere).
 */
async function openPlayer(page: Page): Promise<void> {
  await page.goto('/podcast/p05')
  await page.waitForLoadState('networkidle')
  await page.locator('a[href^="/episode/"]').first().click()
  await page.waitForURL(/\/episode\//)
  await page.waitForLoadState('networkidle')
}

/**
 * An element may wear the accent if a finger can act on it.
 *
 * Deliberately generous: a control nested inside a button (an icon, a label span) inherits the
 * colour and is not itself interactive, so ancestry counts. The goal is to catch a kicker or a
 * badge that is accent-coloured while sitting in no control at all — the specific regression that
 * made "orange appears ~60 times in one scroll" true.
 */
const INTERACTIVE = 'a, button, [role="button"], [role="tab"], [role="switch"], input, select, textarea, summary'

/**
 * Every element painting the accent outside a control, plus the counts that prove the sweep ran.
 *
 * `scanned` and `painted` exist because the assertion is `offenders === []`, which an empty page
 * satisfies perfectly. A surface that failed to render, or a selector that stopped matching, would
 * report success forever. `painted` additionally proves the accent is REACHABLE — if the token
 * resolved to something no element ever uses, every offender check would be trivially clean.
 */
interface AccentScan {
  offenders: string[]
  scanned: number
  painted: number
}

async function accentOffenders(page: Page): Promise<AccentScan> {
  return page.evaluate((interactiveSel) => {
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--lp-accent').trim()
    if (!accent) return { offenders: ['--lp-accent is not defined at all'], scanned: 0, painted: 0 }

    // Resolve the token to the rgb() the browser reports, so comparison is like-for-like.
    const probe = document.createElement('span')
    probe.style.color = accent
    document.body.appendChild(probe)
    const target = getComputedStyle(probe).color
    probe.remove()

    /**
     * Compare by RGB channels, ignoring alpha.
     *
     * String equality against `rgb(r, g, b)` missed every translucent use: `rgba(239, 168, 67, 0.5)`
     * is the accent at half strength — visibly the same colour doing the same signalling job — and
     * did not match. Spending the accent through an alpha is not a loophole the rule intends.
     */
    const rgb = (v: string): string | null => {
      const m = v.match(/^rgba?\(([^)]+)\)$/)
      if (!m) return null
      const n = m[1].split(/[,/\s]+/).filter(Boolean).map(Number)
      if (n.length < 3 || n.slice(0, 3).some(Number.isNaN)) return null
      // Fully transparent paints nothing, whatever its channels say.
      if (n.length > 3 && n[3] === 0) return null
      return `${n[0]},${n[1]},${n[2]}`
    }
    const targetRgb = rgb(target)
    const isAccent = (v: string): boolean => targetRgb != null && rgb(v) === targetRgb

    const out: string[] = []
    let scanned = 0
    let painted = 0
    for (const el of Array.from(document.querySelectorAll<HTMLElement>('body *'))) {
      const cs = getComputedStyle(el)
      if (cs.visibility === 'hidden' || cs.display === 'none') continue
      scanned++

      /**
       * Text and background were the only two properties checked. An accent border, an accent SVG
       * fill and an accent focus outline all paint the colour just as visibly — an accent-bordered
       * badge is precisely the decorative use this rule exists to stop, and it was invisible here.
       * Border and outline only count when they have width and a style, or every element on the
       * page would report its inherited-but-unpainted border colour.
       */
      const paints: string[] = []
      if (isAccent(cs.color)) paints.push('text')
      if (isAccent(cs.backgroundColor)) paints.push('background')
      // `getPropertyValue` takes KEBAB-case CSS names. Passing `borderTopWidth` returns '' for
      // every element, so the whole border branch silently detected nothing — caught only because
      // the falsification probe injected an accent-bordered div and expected to see it reported.
      for (const side of ['top', 'right', 'bottom', 'left']) {
        const w = parseFloat(cs.getPropertyValue(`border-${side}-width`))
        const style = cs.getPropertyValue(`border-${side}-style`)
        if (w > 0 && style !== 'none' && isAccent(cs.getPropertyValue(`border-${side}-color`))) {
          paints.push(`border-${side}`)
        }
      }
      if (
        parseFloat(cs.outlineWidth) > 0 &&
        cs.outlineStyle !== 'none' &&
        isAccent(cs.outlineColor)
      ) {
        paints.push('outline')
      }
      if (isAccent(cs.fill)) paints.push('fill')
      if (parseFloat(cs.strokeWidth || '0') > 0 && isAccent(cs.stroke)) paints.push('stroke')

      if (!paints.length) continue
      painted++

      // Text colour is inherited, so only flag the element that introduces it. Borders, outlines
      // and SVG paint are not inherited in a way that produces this duplication.
      const inheritedOnly =
        paints.length === 1 &&
        paints[0] === 'text' &&
        el.parentElement != null &&
        isAccent(getComputedStyle(el.parentElement).color)
      if (inheritedOnly) continue
      if (el.closest(interactiveSel)) continue

      const id = el.getAttribute('data-testid')
      const cls = (el.className || '').toString().split(/\s+/).slice(0, 3).join('.')
      out.push(`<${el.tagName.toLowerCase()}${id ? ` data-testid="${id}"` : ''} class="${cls}"> [${paints.join('+')}]`)
    }
    return { offenders: out, scanned, painted }
  }, INTERACTIVE)
}

/** Every surface including the Player, whose route needs navigation rather than a path. */
const ALL_SURFACES: Array<[name: string, open: (page: Page) => Promise<void>]> = [
  ...SURFACES.map(
    ([name, path]) =>
      [
        name,
        async (page: Page) => {
          await page.goto(path)
          await page.waitForLoadState('networkidle')
        },
      ] as [string, (page: Page) => Promise<void>],
  ),
  ['player', openPlayer],
]

test.describe('design invariants', () => {
  for (const [name, open] of ALL_SURFACES) {
    test(`${name}: the accent is spent only on things a finger can act on`, async ({ page }, testInfo) => {
      await signInIsolated(page, `invariants-${name}`, testInfo)
      await open(page)

      const { offenders, scanned } = await accentOffenders(page)

      // `offenders === []` is satisfied perfectly by a page that rendered nothing. Without this,
      // a broken route or a changed selector reports "the accent is disciplined" forever.
      expect(scanned, `${name} rendered almost nothing — this assertion would be vacuous`).toBeGreaterThan(20)

      expect(
        offenders,
        `these paint --lp-accent while sitting inside no control. The accent means "you can act ` +
          `on this" (UXS-011, #2013). If one of these IS interactive, give it a real role rather ` +
          `than relaxing this test: ${offenders.join(' | ')}`,
      ).toEqual([])
    })
  }

  test('the accent is actually reachable — the offender sweep is not checking for nothing', async ({
    page,
  }, testInfo) => {
    // The other vacuity risk, and the subtler one. If `--lp-accent` ever resolved to a colour no
    // element uses — a rename, a broken alias, a direction that drops it — every offender check
    // above would pass while checking for a colour that is not on screen. Swept across surfaces
    // rather than pinned to one, because which surface shows an accent is a design decision that
    // may legitimately change.
    await signInIsolated(page, 'invariants-reach', testInfo)
    let painted = 0
    for (const [, open] of ALL_SURFACES) {
      await open(page)
      painted += (await accentOffenders(page)).painted
    }
    expect(
      painted,
      'no element on any surface paints --lp-accent — either the token is broken or every accent ' +
        'check in this file is passing against a colour that never renders',
    ).toBeGreaterThan(0)
  })

  test('every rendered kicker carries the instrument voice, not the accent', async ({ page }, testInfo) => {
    // The specific regression that shipped 55 times. Asserted on the RENDERED page because a
    // component could override `.lp-kicker` with a utility class and the CSS-level guard in
    // src/__checks__/accent-discipline.test.ts would never see it.
    //
    // Swept across surfaces rather than pinned to one. The first version asserted a kicker existed
    // on Home and failed — a fresh signed-in account gets a different hero than I assumed, and
    // hard-coding "Home has a kicker" tested my assumption about the page rather than the rule.
    // The rule is about kickers wherever they are; the vacuity guard is the total count.
    await signInIsolated(page, 'invariants-kicker', testInfo)

    let checked = 0
    for (const [name, open] of ALL_SURFACES) {
      await open(page)

      const found = await page.evaluate(() => {
        const probe = document.createElement('span')
        probe.style.color = getComputedStyle(document.documentElement).getPropertyValue('--lp-accent')
        document.body.appendChild(probe)
        const accent = getComputedStyle(probe).color
        probe.remove()

        return Array.from(document.querySelectorAll<HTMLElement>('.lp-kicker'))
          .filter((el) => {
            const cs = getComputedStyle(el)
            return cs.display !== 'none' && cs.visibility !== 'hidden'
          })
          .map((el) => {
            const cs = getComputedStyle(el)
            return {
              text: (el.textContent || '').trim().slice(0, 40),
              mono: /mono/i.test(cs.fontFamily),
              isAccent: cs.color === accent,
            }
          })
      })

      const wrong = found.filter((k) => !k.mono || k.isAccent)
      expect(
        wrong,
        `on ${name}, these kickers break the instrument-voice rule (mono, never accent): ` +
          wrong.map((k) => `"${k.text}"${k.isAccent ? ' [accent]' : ''}${k.mono ? '' : ' [not mono]'}`).join(' | '),
      ).toEqual([])
      checked += found.length
    }

    // A sweep that found nothing would pass forever while asserting nothing.
    expect(checked, 'no kicker rendered on any surface — this test would be vacuous').toBeGreaterThan(0)
  })
})
