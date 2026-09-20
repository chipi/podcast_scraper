import { describe, expect, it } from "vitest"
import en from "../i18n/locales/en.json"
import personContentSrc from "../components/PersonCardContent.vue?raw"
import topicContentSrc from "../components/TopicCardContent.vue?raw"
import discoverySrc from "../components/DiscoveryList.vue?raw"
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
    expect(en.kp.storyline).toContain("Storyline")
    expect(en.ec.singleTopic).toContain("storyline")

    // "Similar" stays distinct WHERE IT STILL APPEARS: semantic similarity is a different idea
    // from co-occurrence, not a second name for it.
    //
    // `kp.similar` is gone (operator 2026-09-19). The Knowledge Panel stacked "Storyline · X" above
    // an inert "Similar · Y" in identical positions — one navigated, the other could not, and
    // nothing said why. A semantic cluster has no card and no route, so the line named an internal
    // mechanism and offered nowhere to go. The entity card still surfaces the concept as
    // "N similar topics", where it sits beside a count rather than impersonating a link.
    // Indexed access, not `en.kp.similar`: the key is GONE, so the typed read is a compile error —
    // which the new test typecheck correctly refuses. Asserting its absence has to go through a
    // widened view of the object.
    expect(
      (en.kp as Record<string, unknown>).similar,
      "kp.similar should stay retired — see the issue on where semantic clusters belong",
    ).toBeUndefined()
    expect(en.ec.clusterMembers).toContain("similar")

    // No consumer string may reintroduce "Theme ·" for either concept.
    for (const v of [en.kp.storyline, en.ec.clusterMembers]) {
      expect(v).not.toMatch(/Theme ·/)
    }
  })

  it("explains the × metric on screen, not only in a title attribute (#1595)", () => {
    // title attributes do not exist on touch, so on the primary platform the core trending metric
    // was a bare "1.6×" with no way to decode it.
    // The × metric example is rendered ON SCREEN by DiscoveryList (a bold "2×" beside the hint),
    // not buried in a title attr (title attrs don't exist on touch). Both measures' hints render —
    // whichever the active sort selects.
    expect(discoverySrc).toMatch(/×/)
    expect(discoverySrc).toMatch(/t\(['"]home\.momentumHint['"]\)/)
    expect(discoverySrc).toMatch(/t\(['"]home\.trendingHint['"]\)/)
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
