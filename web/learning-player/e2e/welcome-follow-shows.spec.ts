import { expect, test } from '@playwright/test'
import { navTo, signInIsolated } from './helpers'

/**
 * The welcome card's second way in (operator 2026-10-07): "Follow shows" opens Discover on Shows,
 * and the show followed there makes What's new the listener's own on the way back — without a
 * reload. Home is kept alive, and What's new used to load once, so it kept saying "across all
 * shows" until the app restarted. REAL API over the committed corpus, NO mocks.
 */
test("Follow shows from the welcome card, and What's new becomes yours on return", async ({
  page,
}, testInfo) => {
  await signInIsolated(page, `welcome-follow-${Date.now()}`, testInfo)
  await page.goto('/')
  await expect(page.getByTestId('interests-welcome')).toBeVisible()
  await expect(page.getByText('New across all shows', { exact: false })).toBeVisible()

  await page.getByTestId('interests-follow-shows').click()
  await expect(page).toHaveURL(/\/browse\?tab=shows/)

  // Open a show from the Shows list and follow it.
  await page.locator('main a[href^="/podcast/"]:visible').first().click()
  await expect(page).toHaveURL(/\/podcast\//)
  const follow = page.getByTestId('follow-show')
  await expect(follow).toHaveAttribute('aria-pressed', 'false')
  await follow.click()
  await expect(follow).toHaveAttribute('aria-pressed', 'true')

  // Back to Home IN-APP (kept alive, no reload): What's new is now from the followed show.
  await navTo(page, 'home')
  await expect(page.getByText('New in your shows and topics', { exact: false })).toBeVisible()
  await expect(page.getByText('New across all shows', { exact: false })).toHaveCount(0)
})

test('with no listening yet, Recommended appears once there is something to base it on', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, `rec-from-follows-${Date.now()}`, testInfo)
  await page.goto('/')
  // No basis yet: no Recommended at all (operator 2026-10-07 — not a section saying nothing).
  await expect(page.getByTestId('interests-welcome')).toBeVisible()
  await expect(page.getByTestId('home-recommended')).toHaveCount(0)

  // Follow a topic the corpus carries, through the real API, and come back to Home.
  const follow = await page.request.post(`/api/app/interests/${encodeURIComponent('topic:risk-management')}`)
  expect(follow.ok()).toBeTruthy()
  await page.reload()
  const rec = page.getByTestId('home-recommended')
  await expect(rec).toBeVisible()
  await expect(rec).toContainText('Picked from what you follow')
})
