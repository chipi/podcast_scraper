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

  // The step's action, "Skip step" and "Not now" share one row, at phone width too (operator
  // 2026-10-08).
  const card = page.getByTestId('interests-welcome')
  const tops = await Promise.all(
    ['interests-choose', 'guided-skip', 'interests-not-now'].map(async (id) =>
      Math.round((await card.getByTestId(id).boundingBox())!.y),
    ),
  )
  expect(new Set(tops).size).toBe(1)

  // Shows are step 2 of the guided start (operator 2026-10-07): skip interests to reach it. The
  // shows come from the server's ranking (`/podcasts/suggested`, operator 2026-10-08), fetched as the
  // step opens.
  const [suggested] = await Promise.all([
    page.waitForResponse((r) => r.url().includes('/api/app/podcasts/suggested')),
    page.getByTestId('guided-skip').click(),
  ])
  await expect(page.getByTestId('interests-welcome')).toHaveAttribute('data-step', '2')
  const ranked = ((await suggested.json()).items as Array<{ feed_id: string }>).map((p) => p.feed_id)
  expect(ranked.length).toBeGreaterThan(0)
  const tiles = page.getByTestId('guided-shows').locator('li')
  await expect(tiles).toHaveCount(ranked.length)
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

test('Settings brings the getting-started guide back, from step 1, after "Not now"', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, `guided-restart-${Date.now()}`, testInfo)
  await page.goto('/')
  await expect(page.getByTestId('interests-welcome')).toBeVisible()
  await page.getByTestId('interests-not-now').click()
  await expect(page.getByTestId('interests-welcome')).toHaveCount(0)
  // Snoozed, not gone for the session only: a reload keeps it away (operator 2026-10-08).
  await page.reload()
  await expect(page.getByRole('heading', { level: 2 }).first()).toBeVisible()
  await expect(page.getByTestId('interests-welcome')).toHaveCount(0)

  await page.goto('/settings')
  await page.getByTestId('settings-guided-restart').click()
  await expect(page).toHaveURL(/\/$/)
  await expect(page.getByTestId('interests-welcome')).toHaveAttribute('data-step', '1')
})

test('saving interests from the guided start leaves Home at the top', async ({ page }, testInfo) => {
  // Operator 2026-10-08, on the phone: after choosing interests and saving, Home came back scrolled
  // down. The guide moves on to step 2 at the top of Home, so the listener must land there.
  await signInIsolated(page, `guided-scroll-${Date.now()}`, testInfo)
  await page.goto('/')
  await page.getByTestId('interests-choose').click()
  await page.getByTestId('interest-add-topic').click()
  const suggestions = page.getByTestId('interest-add-panel-topic').getByTestId('interest-suggestion')
  for (let i = 0; i < 3; i++) {
    await suggestions.first().click()
  }
  // On iOS, lifting the search box above the keyboard (and the keyboard itself) also scrolls the
  // PAGE behind the sheet. Chrome has no on-screen keyboard, so do what iOS does to the page.
  await page.evaluate(() => window.scrollTo(0, 600))
  expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(0)
  await page.getByTestId('interests-save').click()
  await expect(page.getByTestId('interests-welcome')).toHaveAttribute('data-step', '2')
  await page.waitForTimeout(1500) // let What's new / Recommended finish re-rendering
  expect(await page.evaluate(() => window.scrollY)).toBe(0)
})
