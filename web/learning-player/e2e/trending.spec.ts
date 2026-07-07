import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * RFC-103 momentum "Trending now" rail on Home — REAL API over the COMMITTED validation corpus, NO
 * mocks. The e2e webServer pins APP_TRENDING_NOW=2026-07-20 (just after the corpus's newest episode)
 * so the read-time momentum is deterministic and the risk/systems content reads as rising; GET
 * /api/app/trending?kind=topic then returns those topics and the rail renders.
 *
 * Trending topics from the committed corpus at that anchor: "systems thinking" / "risk management"
 * (the cross-domain storyline the newest episodes carry).
 */
test('Home shows the Trending-now momentum rail with rising topics', async ({ page }) => {
  await page.goto('/')

  const rail = page.locator('[data-testid="momentum-rail-topic"]')
  await expect(rail).toBeVisible()
  await expect(rail.getByRole('heading', { name: 'Trending now' })).toBeVisible()

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
  await expect(follow).toHaveAttribute('aria-pressed', 'false')

  await follow.click()
  await expect(follow).toHaveAttribute('aria-pressed', 'true') // persisted to the interests store
})
