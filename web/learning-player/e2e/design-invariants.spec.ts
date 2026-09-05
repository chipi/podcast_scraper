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

/** Surfaces worth holding to the rules — the ones the redesign actually governs. */
const SURFACES: Array<[name: string, path: string]> = [
  ['home', '/'],
  ['browse', '/browse'],
  ['library', '/library'],
  ['profile', '/profile'],
]

/**
 * An element may wear the accent if a finger can act on it.
 *
 * Deliberately generous: a control nested inside a button (an icon, a label span) inherits the
 * colour and is not itself interactive, so ancestry counts. The goal is to catch a kicker or a
 * badge that is accent-coloured while sitting in no control at all — the specific regression that
 * made "orange appears ~60 times in one scroll" true.
 */
const INTERACTIVE = 'a, button, [role="button"], [role="tab"], [role="switch"], input, select, textarea, summary'

async function accentOffenders(page: Page): Promise<string[]> {
  return page.evaluate((interactiveSel) => {
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--lp-accent').trim()
    if (!accent) return ['--lp-accent is not defined at all']

    // Resolve the token to the rgb() the browser reports, so comparison is like-for-like.
    const probe = document.createElement('span')
    probe.style.color = accent
    document.body.appendChild(probe)
    const target = getComputedStyle(probe).color
    probe.remove()

    const out: string[] = []
    for (const el of Array.from(document.querySelectorAll<HTMLElement>('body *'))) {
      const cs = getComputedStyle(el)
      if (cs.visibility === 'hidden' || cs.display === 'none') continue
      // Only elements that actually paint the accent as their own text or fill.
      const paintsAccent = cs.color === target || cs.backgroundColor === target
      if (!paintsAccent) continue
      // Text colour is inherited, so only flag the element that introduces it.
      if (cs.color === target && el.parentElement && getComputedStyle(el.parentElement).color === target) {
        continue
      }
      if (el.closest(interactiveSel)) continue
      const id = el.getAttribute('data-testid')
      const cls = (el.className || '').toString().split(/\s+/).slice(0, 3).join('.')
      out.push(`<${el.tagName.toLowerCase()}${id ? ` data-testid="${id}"` : ''} class="${cls}">`)
    }
    return out
  }, INTERACTIVE)
}

test.describe('design invariants', () => {
  for (const [name, path] of SURFACES) {
    test(`${name}: the accent is spent only on things a finger can act on`, async ({ page }, testInfo) => {
      await signInIsolated(page, `invariants-${name}`, testInfo)
      await page.goto(path)
      await page.waitForLoadState('networkidle')

      const offenders = await accentOffenders(page)
      expect(
        offenders,
        `these paint --lp-accent while sitting inside no control. The accent means "you can act ` +
          `on this" (UXS-011, #2013). If one of these IS interactive, give it a real role rather ` +
          `than relaxing this test: ${offenders.join(' | ')}`,
      ).toEqual([])
    })
  }

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
    for (const [name, path] of SURFACES) {
      await page.goto(path)
      await page.waitForLoadState('networkidle')

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
