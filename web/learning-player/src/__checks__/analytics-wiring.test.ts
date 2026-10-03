import { describe, expect, it } from 'vitest'
import landingSrc from '../views/LandingView.vue?raw'
import routerSrc from '../router/index.ts?raw'
import analyticsSrc from '../services/analytics.ts?raw'
import mainSrc from '../main.ts?raw'
import searchSrc from '../views/SearchView.vue?raw'

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


describe('no telemetry target is hardcoded (operator rule, 2026-10-03)', () => {
  /**
   * Both the Umami website id and the GlitchTip DSN used to be literals in the source, and both
   * were WRONG in a way nothing could notice.
   *
   * The Umami one referenced `30384fd4-b22b-406c-b5f6-054a0e0d16d1`, a website that does not exist
   * in the instance — measured by posting it to `/api/send`, which answered
   * `{"error":{"message":"Website not found.","code":"bad-request","status":400}}`. So every event
   * sent from `vite dev` was rejected, silently, because `track()` is fire-and-forget. The DSN was
   * the same shape of mistake: a tailnet hostname that does not resolve from every account and a
   * project id nothing verifies, with a transport that also swallows its own errors.
   *
   * A wrong target is invisible in exactly the way a missing one is not: the dashboard is empty,
   * and empty reads as "nobody used it".
   */
  /**
   * Comments are stripped before scanning, deliberately.
   *
   * `analytics.ts` names the dead id in prose, because "this exact id was wrong and here is the
   * response that proved it" is the most useful thing a future reader can be told. What must not
   * come back is an id the CODE reads. Scanning the raw file would force the history out to keep
   * the guard green, which trades a real record for a mechanical one.
   */
  function withoutComments(src: string): string {
    return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '')
  }

  it('no UUID website id appears in analytics.ts code', () => {
    const uuids = withoutComments(analyticsSrc).match(
      /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi,
    )
    expect(
      uuids,
      'a website id belongs in the environment (VITE_UMAMI_WEBSITE_ID), never in the source',
    ).toBeNull()
  })

  it('still records WHICH id was dead, so the lesson survives the guard', () => {
    // If someone "fixes" the test above by deleting the prose, this one goes red instead.
    expect(analyticsSrc).toContain('30384fd4-b22b-406c-b5f6-054a0e0d16d1')
    expect(analyticsSrc).toMatch(/Website not found/i)
  })

  it('main.ts contains no Sentry DSN literal', () => {
    expect(
      withoutComments(mainSrc),
      'a DSN belongs in VITE_SENTRY_DSN_PLAYER / _DEV, never in the source',
    ).not.toMatch(/https?:\/\/[0-9a-f]{16,}@/i)
  })

  it('neither file hardcodes the tailnet host as a telemetry target', () => {
    // `homelab` does not resolve from every account on this machine — it did not resolve from the
    // one that found these bugs, which is half of why the dev defaults could never have worked.
    for (const [name, src] of [
      ['analytics.ts', withoutComments(analyticsSrc)],
      ['main.ts', withoutComments(mainSrc)],
    ] as const) {
      expect(src, `${name} must not hardcode a homelab URL`).not.toMatch(
        /['"]https?:\/\/homelab[:/]/,
      )
    }
  })
})


describe('search_result_click wiring (#2267)', () => {
  /**
   * Covered HERE because the browser tier cannot reach it on this machine.
   *
   * The episode variant is fired by `openEpisode`, whose only callers are the per-hit "Play from
   * {time}" jump controls. Those render on `hitStartSeconds(hit) != null && g.slug`, and this host has
   * no `lancedb` wheel (no x86_64 build), so the API answers `no_index`, the search page renders its
   * error state, and no grouped episode row with a jump control ever appears. The topic variant needs
   * related-topic chips, which are derived from topics attached to the returned hits — and the
   * committed fixture attached none for every query tried.
   *
   * Neither absence is a wiring problem, so skipping in the e2e tier and asserting nothing would have
   * left the event with no coverage at all. This is the part that can be checked anywhere.
   */
  it('separates the topic chip from an episode result, and pairs each with its rank rule', () => {
    const topicCall = searchSrc.slice(searchSrc.indexOf('function openTopicChip'))
    const topicBody = topicCall.slice(0, topicCall.indexOf('\n}'))
    expect(topicBody).toContain("track(\"search_result_click\", { result_kind: \"topic\", rank: \"1\" })")
    // The chip row is a short unranked set. A literal '1' is correct precisely because inventing an
    // ordinal would make the same bucket mean two different things across events.
    expect(topicBody, 'a chip must not be given a computed rank').not.toContain('toRankBucket')
    // And it must still emit the pivot event — the two answer different questions.
    expect(topicBody).toContain('track("entity_open"')

    const epCall = searchSrc.slice(searchSrc.indexOf('function openEpisode'))
    const epBody = epCall.slice(0, epCall.indexOf('\n}'))
    expect(epBody).toContain('result_kind: "episode"')
    // An episode result DOES have a position, and it must be bucketed rather than raw: a raw ordinal
    // over a small beta turns the event into a per-query fingerprint.
    expect(epBody, 'an episode rank must go through toRankBucket').toContain('toRankBucket')
  })
})
