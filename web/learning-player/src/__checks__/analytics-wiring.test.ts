import { describe, expect, it } from 'vitest'
import landingSrc from '../views/LandingView.vue?raw'
import routerSrc from '../router/index.ts?raw'

/**
 * Static guards on WHERE analytics calls are attached (#2267).
 *
 * The registry test in `services/analytics.test.ts` proves every tracked NAME exists. These prove
 * something it cannot: that a call is on the right element with the right props.
 *
 * They exist because of a real mistake made while wiring this slice. A mechanical edit attached
 * `position: 'closing'` to the HERO call-to-action — both are `:to="signupTo()"`, so a
 * first-match replacement hit the wrong one. Nothing failed: the event fired, with a plausible
 * value, and the onboarding funnel would have reported that nobody is ever convinced by the hero
 * and everybody is convinced by the footer. A wrong enum value is invisible in a way a missing
 * event is not, which is why these are anchored on each element's own `data-testid`.
 */

/** The attributes of the element carrying `testid`, up to the closing `>`. */
function elementWith(src: string, testid: string): string {
  const at = src.indexOf(`data-testid="${testid}"`)
  expect(at, `no element with data-testid="${testid}"`).toBeGreaterThan(-1)
  // Back up to the start of this tag, forward to the end of its attribute list.
  const open = src.lastIndexOf('<', at)
  const close = src.indexOf('>', at)
  return src.slice(open, close)
}

describe('landing funnel wiring (#2267)', () => {
  it('fires landing_view on mount, before the teaser fetch can fail', () => {
    // A listener whose network drops must still count as having SEEN the landing, or the funnel
    // under-counts exactly the people who had the worst first experience.
    expect(landingSrc).toMatch(/track\('landing_view'\)/)
    const viewAt = landingSrc.indexOf("track('landing_view')")
    const fetchAt = landingSrc.indexOf('getDiscover(')
    expect(viewAt, 'landing_view must be fired before the teaser fetch').toBeLessThan(fetchAt)
  })

  it('pairs each CTA with its own position — hero is hero, footer is closing', () => {
    // The distinction is the entire purpose of the property: "convinced by the hero" versus
    // "convinced after reading". Swap them and the funnel tells a confident, inverted story.
    const hero = elementWith(landingSrc, 'landing-cta-primary')
    expect(hero).toContain("cta: 'create_account'")
    expect(hero).toContain("position: 'hero'")
    expect(hero, 'the hero CTA must not report itself as the closing one').not.toContain(
      "position: 'closing'",
    )

    const foot = elementWith(landingSrc, 'landing-cta-foot')
    expect(foot).toContain("cta: 'create_account'")
    expect(foot).toContain("position: 'closing'")
    expect(foot).not.toContain("position: 'hero'")

    const signin = elementWith(landingSrc, 'landing-cta-signin')
    expect(signin).toContain("cta: 'sign_in'")
  })

  it('distinguishes a show teaser from a topic chip', () => {
    // Both funnel to signup, and the spec asks which kind of bait worked.
    const card = elementWith(landingSrc, 'landing-card')
    expect(card).toContain("kind: 'show'")
    const chip = elementWith(landingSrc, 'landing-chip')
    expect(chip).toContain("kind: 'topic'")
  })
})

describe('screen_view wiring (#2267)', () => {
  it('reports the route NAME, never the path', () => {
    // The path carries slugs, ids and — on /search — the query string that `data-exclude-search`
    // exists to keep out of analytics. Sending it here would reintroduce the leak through a custom
    // event, where the script-tag attribute cannot help.
    expect(routerSrc).toMatch(/track\('screen_view',\s*\{\s*screen:\s*to\.name/)
    const call = routerSrc.slice(routerSrc.indexOf("track('screen_view'"))
    const stmt = call.slice(0, call.indexOf('\n'))
    expect(stmt).not.toMatch(/to\.(path|fullPath)/)
  })

  it('is in afterEach, so a redirected navigation is not reported as a view', () => {
    const hookAt = routerSrc.lastIndexOf('router.afterEach')
    const callAt = routerSrc.indexOf("track('screen_view'")
    expect(callAt).toBeGreaterThan(hookAt)
  })
})
