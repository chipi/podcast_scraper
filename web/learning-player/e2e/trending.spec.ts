import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * RFC-103 momentum "Rising now" rail on Home — REAL API over the COMMITTED validation corpus, NO
 * mocks. The e2e webServer pins APP_TRENDING_NOW=2026-07-20 (just after the corpus's newest episode)
 * so the read-time momentum is deterministic and the risk/systems content reads as rising; GET
 * /api/app/trending?kind=topic then returns those topics and the rail renders.
 *
 * Trending topics from the committed corpus at that anchor: "systems thinking" / "risk management"
 * (the cross-domain storyline the newest episodes carry).
 */
test('Home shows the Rising-now momentum rail with rising topics', async ({ page }, testInfo) => {
  // RFC-120: home is login-first; sign in so the HomeView renders rather than the lure landing.
  await signInIsolated(page, 'trending-rising', testInfo)
  await page.goto('/')

  // #4 folded the three "what's hot" rails into one tabbed area; "Rising now" is now the default
  // discovery TAB label (not a duplicate rail heading), and its panel holds the momentum rail.
  const risingTab = page.getByTestId('discovery-tab-rising')
  await expect(risingTab).toHaveText('Rising now')
  await expect(risingTab).toHaveAttribute('aria-selected', 'true')

  const rail = page.locator('[data-testid="momentum-rail-topic"]')
  await expect(rail).toBeVisible()

  // At least one trending chip, and it carries a velocity multiplier (↑N×) — the momentum signal.
  const chips = rail.locator('[data-testid="momentum-chip"]')
  await expect(chips.first()).toBeVisible()
  await expect(rail).toContainText('×')
  // The risk/systems storyline is what's freshest at the pinned anchor.
  await expect(rail.getByText(/risk management|systems thinking/i).first()).toBeVisible()
})

test('signed in: following a trending topic from the rail toggles to followed', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'trending', testInfo)
  await page.goto('/')

  const rail = page.locator('[data-testid="momentum-rail-topic"]')
  await expect(rail).toBeVisible()
  const follow = rail.locator('[data-testid="momentum-follow"]').first()
  await expect(follow).toBeVisible()

  // Idempotent by construction: assert the button TOGGLES, whichever state it starts in.
  //
  // This used to require `aria-pressed="false"` up front, which made the test pass exactly once
  // per api container. The follow persists server-side under this user, so a second run of the
  // suite against the same long-lived container met a topic it had already followed and failed on
  // its very first assertion — before clicking anything, so a retry could never recover either.
  // The bug looked like a product regression and was a test that had eaten its own precondition.
  //
  // Toggling from whatever is there also tests MORE: it covers the unfollow path, which the
  // one-way version never touched.
  const before = await follow.getAttribute('aria-pressed')
  const after = before === 'true' ? 'false' : 'true'

  await follow.click()
  await expect(follow).toHaveAttribute('aria-pressed', after) // persisted to the interests store

  // Back to where we found it, so this spec leaves no trace for the next run — the property whose
  // absence caused the failure above.
  await follow.click()
  await expect(follow).toHaveAttribute('aria-pressed', before ?? 'false')
})
