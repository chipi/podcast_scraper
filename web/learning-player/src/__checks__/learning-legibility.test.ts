import { describe, expect, it } from 'vitest'
import en from '../i18n/locales/en.json'
import entityCardSrc from '../components/EntityCardBody.vue?raw'
import momentumSrc from '../components/MomentumRail.vue?raw'
import playerSrc from '../views/PlayerView.vue?raw'

/**
 * Guardrail (#1595, #1603) — the learning layer is the reason to choose this over Spotify, and each
 * of these regressions made it *less* legible while changing nothing functional. They are cheap to
 * reintroduce and invisible in review, so they're pinned here.
 */

describe('the learning differentiator stays legible', () => {
  it('insights is a labelled control, not an emoji in the stats cluster (#1595)', () => {
    // It read "💡 3", styled like a statistic, sitting between listener and open counts — the least
    // legible control on the page, for the product's central feature.
    expect(playerSrc).toContain('data-testid="player-open-insights"')
    // Comments are stripped first: the docblock explaining this change names the old emoji, and a
    // check that fails on its own rationale is a check people delete.
    const rendered = playerSrc.replace(/<!--[\s\S]*?-->/g, '').replace(/\/\*[\s\S]*?\*\//g, '')
    expect(rendered).not.toContain('💡')
  })

  it('uses ONE consumer word for co-occurrence clusters: storyline (#1603)', () => {
    // The code said "Theme ·" for co-occurrence while UXS-013 mandates "Theme" for the SEMANTIC
    // cluster — exactly backwards — and "Storyline" (Home's word) appeared in no spec at all. One
    // word wins, and it is the one users already meet on Home.
    expect(en.kp.theme).toContain('Storyline')
    expect(en.ec.themeMembers).toContain('storyline')
    expect(en.ec.singleTopic).toContain('storyline')

    // "Similar" stays distinct: semantic similarity is a different idea, not a second name.
    expect(en.kp.similar).toContain('Similar')
    expect(en.ec.clusterMembers).toContain('similar')

    // No consumer string may reintroduce "Theme ·" for either concept.
    for (const v of [en.kp.theme, en.kp.similar, en.ec.themeMembers, en.ec.clusterMembers]) {
      expect(v).not.toMatch(/Theme ·/)
    }
  })

  it('explains the × metric on screen, not only in a title attribute (#1595)', () => {
    // title attributes do not exist on touch, so on the primary platform the core trending metric
    // was a bare "1.6×" with no way to decode it.
    expect(en.home.trendingHint).toMatch(/×/)
    expect(en.home.momentumHint).toMatch(/×/)
    expect(momentumSrc).toContain("t('home.momentumHint')")
  })

  it('puts synthesis before search on the entity card (#1595)', () => {
    // The card exists for perspectives/consensus/arc. A full-width accent "Search every episode"
    // button above all of it made the most prominent control the one that navigates AWAY.
    const signals = entityCardSrc.indexOf('<EntitySignals')
    const search = entityCardSrc.indexOf('searchLibrary"')
    expect(signals).toBeGreaterThan(-1)
    expect(search).toBeGreaterThan(-1)
    expect(signals).toBeLessThan(search)
  })
})
