import { expect, test } from '@playwright/test'
import { attachSink, DEV_UMAMI_WEBSITE_ID } from './sink'
import './settle'

/**
 * The onboarding funnel's first step, end to end (#2267).
 *
 * Property values, not just names. The hero and the closing call-to-action are the SAME component
 * with the same `:to`, and a mechanical edit in this arc attached `position: 'closing'` to the hero
 * one. Nothing failed: the event fired with a plausible value, and the funnel would have reported
 * that nobody is ever convinced by the hero and everybody by the footer. Only an assertion that
 * ties a specific element to a specific value catches that — which is why each click here is made
 * through the element's own `data-testid`.
 */

test.describe('landing funnel', () => {
  test('landing_view fires once, before any teaser content can fail', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/welcome')
    const view = await sink.waitForEvent('landing_view')
    expect(view.website).toBe(DEV_UMAMI_WEBSITE_ID)

    // Once per visit. A view counted twice makes every downstream conversion rate look half as good.
    await expect
      .poll(() => sink.byName('landing_view').length, { timeout: 5_000 })
      .toBe(1)
  })

  test('each CTA reports its OWN position and cta kind', async ({ page }) => {
    const sink = attachSink(page)

    // Hero — create account
    await page.goto('/welcome')
    await sink.waitForEvent('landing_view')
    await page.getByTestId('landing-cta-primary').click()
    const hero = await sink.waitForEvent('landing_cta_click')
    expect(hero.data).toMatchObject({ cta: 'create_account', position: 'hero' })

    // Hero — sign in
    await page.goto('/welcome')
    await page.getByTestId('landing-cta-signin').click()
    await expect.poll(() => sink.byName('landing_cta_click').length).toBeGreaterThanOrEqual(2)
    const signin = sink.byName('landing_cta_click')[1]
    expect(signin.data).toMatchObject({ cta: 'sign_in', position: 'hero' })

    // Closing — create account. The one that must NOT say 'hero'.
    await page.goto('/welcome')
    await page.getByTestId('landing-cta-foot').scrollIntoViewIfNeeded()
    await page.getByTestId('landing-cta-foot').click()
    await expect.poll(() => sink.byName('landing_cta_click').length).toBeGreaterThanOrEqual(3)
    const foot = sink.byName('landing_cta_click')[2]
    expect(foot.data).toMatchObject({ cta: 'create_account', position: 'closing' })

    // The distinction actually survived the wire — three clicks, two distinct positions.
    const positions = sink.byName('landing_cta_click').map((b) => b.data?.position)
    expect(positions).toEqual(['hero', 'hero', 'closing'])
  })

  test('a show teaser and a topic chip are distinguishable', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/welcome')
    await sink.waitForEvent('landing_view')

    // The teasers come from the real /api/app/discover over the committed corpus. If the corpus
    // ever stops producing them the section is `v-if`-ed away, so assert presence explicitly
    // rather than letting a missing element read as "the event does not fire".
    const card = page.getByTestId('landing-card').first()
    await expect(card, 'the landing needs real discover content to have a teaser to click').toBeVisible()
    await card.click()
    const show = await sink.waitForEvent('landing_teaser_click')
    expect(show.data).toMatchObject({ kind: 'show' })

    await page.goto('/welcome')
    const chip = page.getByTestId('landing-chip').first()
    await expect(chip).toBeVisible()
    await chip.click()
    await expect.poll(() => sink.byName('landing_teaser_click').length).toBeGreaterThanOrEqual(2)
    expect(sink.byName('landing_teaser_click')[1].data).toMatchObject({ kind: 'topic' })
  })
})

test.describe('screen_view', () => {
  test('reports the route NAME and never a path, slug or query', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/welcome')
    const sv = await sink.waitForEvent('screen_view')

    const screen = String(sv.data?.screen ?? '')
    expect(screen.length, 'screen_view must carry a screen name').toBeGreaterThan(0)
    expect(screen, 'a route NAME, not a path').not.toContain('/')

    // Every screen_view across the session: no path-shaped or query-shaped value anywhere. The
    // router hook could regress to `to.path` or `to.fullPath` and still fire a plausible event.
    await page.goto('/welcome')
    await page.getByTestId('landing-cta-signin').click()
    await page.waitForLoadState('networkidle')
    for (const b of sink.byName('screen_view')) {
      const s = String(b.data?.screen ?? '')
      expect(s, `screen_view carried a path-like value: ${s}`).not.toMatch(/[/?=]/)
    }
  })
})
