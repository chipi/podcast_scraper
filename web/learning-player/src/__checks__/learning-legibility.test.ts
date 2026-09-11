import { describe, expect, it } from "vitest"
import en from "../i18n/locales/en.json"
import personContentSrc from "../components/PersonCardContent.vue?raw"
import topicContentSrc from "../components/TopicCardContent.vue?raw"
import momentumSrc from "../components/MomentumRail.vue?raw"
import playerSrc from "../views/PlayerView.vue?raw"

/**
 * Guardrail (#1595, #1603) — the learning layer is the reason to choose this over Spotify, and each
 * of these regressions made it *less* legible while changing nothing functional. They are cheap to
 * reintroduce and invisible in review, so they're pinned here.
 */

describe("the learning differentiator stays legible", () => {
  it("insights is a labelled control, not an emoji in the stats cluster (#1595)", () => {
    // It read "💡 3", styled like a statistic, sitting between listener and open counts — the least
    // legible control on the page, for the product's central feature.
    expect(playerSrc).toContain('data-testid="player-open-insights"')
    // Comments are stripped first: the docblock explaining this change names the old emoji, and a
    // check that fails on its own rationale is a check people delete.
    const rendered = playerSrc.replace(/<!--[\s\S]*?-->/g, "").replace(/\/\*[\s\S]*?\*\//g, "")
    expect(rendered).not.toContain("💡")
  })

  it("uses ONE consumer word for co-occurrence clusters: storyline (#1603)", () => {
    // The code said "Theme ·" for co-occurrence while UXS-013 mandates "Theme" for the SEMANTIC
    // cluster — exactly backwards — and "Storyline" (Home's word) appeared in no spec at all. One
    // word wins, and it is the one users already meet on Home.
    expect(en.kp.theme).toContain("Storyline")
    expect(en.ec.themeMembers).toContain("storyline")
    expect(en.ec.singleTopic).toContain("storyline")

    // "Similar" stays distinct: semantic similarity is a different idea, not a second name.
    expect(en.kp.similar).toContain("Similar")
    expect(en.ec.clusterMembers).toContain("similar")

    // No consumer string may reintroduce "Theme ·" for either concept.
    for (const v of [en.kp.theme, en.kp.similar, en.ec.themeMembers, en.ec.clusterMembers]) {
      expect(v).not.toMatch(/Theme ·/)
    }
  })

  it("explains the × metric on screen, not only in a title attribute (#1595)", () => {
    // title attributes do not exist on touch, so on the primary platform the core trending metric
    // was a bare "1.6×" with no way to decode it.
    expect(en.home.trendingHint).toMatch(/×/)
    expect(en.home.momentumHint).toMatch(/×/)
    // Quote-agnostic: the hint is rendered on screen (not only in a title attr). Asserting a
    // specific quote style made this break on a prettier reformat rather than a real regression.
    expect(momentumSrc).toMatch(/t\(['"]home\.momentumHint['"]\)/)
  })

  it("does not describe the two momentum measures with the same words (#1668)", () => {
    // Home carries two independent measures and both are deliberately kept — "Rising now" is a
    // read-time EWMA against recent weeks, "Trending topics" is last month against its own
    // 6-month average. They legitimately disagree: on the validation corpus systems thinking
    // reads 1.78x on one and 0.86 on the other.
    //
    // That is only confusing because both used to be explained with the SAME phrase, "its usual
    // rate" — so the page said "twice its usual rate" directly above "nothing is above its usual
    // rate", which is a flat contradiction no reader can resolve. Each hint must name its own
    // comparison window.
    const usualRate = /usual rate/i
    expect(en.home.momentumHint).not.toMatch(usualRate)
    expect(en.home.trendingHint).not.toMatch(usualRate)
    expect(en.home.trendingQuiet).not.toMatch(usualRate)

    // Each names the window it compares against.
    expect(en.home.trendingHint).toMatch(/6-month/i)
    expect(en.home.trendingQuiet).toMatch(/6-month/i)
    expect(en.home.momentumHint).toMatch(/recent weeks|right now/i)

    // And the quiet state must not read as a claim about the other rail's metric.
    expect(en.home.trendingQuiet).not.toMatch(/^nothing is (rising|trending)\b/i)
  })

  it("puts synthesis before search on the entity card (#1595)", () => {
    // The card exists for perspectives/consensus/arc/momentum. A prominent "Search every episode"
    // button ABOVE all of it made the most prominent control the one that navigates AWAY. The body
    // is per-kind now (Person/TopicCardContent), so assert it in each — matching the template
    // `@click="searchLibrary"` (the `"` avoids the script's function definition): the person's
    // signals and the topic's momentum both precede the search affordance.
    const pSignals = personContentSrc.indexOf("<EntitySignals")
    const pSearch = personContentSrc.indexOf('searchLibrary"')
    expect(pSignals).toBeGreaterThan(-1)
    expect(pSearch).toBeGreaterThan(-1)
    expect(pSignals).toBeLessThan(pSearch)

    const tMomentum = topicContentSrc.indexOf("<TrendMomentum")
    const tSearch = topicContentSrc.indexOf('searchLibrary"')
    expect(tMomentum).toBeGreaterThan(-1)
    expect(tSearch).toBeGreaterThan(-1)
    expect(tMomentum).toBeLessThan(tSearch)
  })
})
