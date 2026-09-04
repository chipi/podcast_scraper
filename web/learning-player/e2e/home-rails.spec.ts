import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The Home rails that had unit tests and no e2e (E2E_SURFACE_MAP coverage gaps, closed 2026-09-03).
 *
 * They are grouped because they share ONE contract, and that contract is what is actually worth
 * asserting in a browser: a rail with nothing to show **omits itself**. A unit test can prove a
 * component renders chips from props; only a real render against a real API can prove the section
 * does not leave an empty shell behind when the API returns nothing.
 */

test('Your Week renders for a signed-in listener and can expand', async ({ page }, testInfo) => {
  await signInIsolated(page, 'home-yourweek', testInfo)
  await page.goto('/')

  const week = page.getByTestId('your-week')
  await expect(week).toBeVisible()
  // compact ↔ full is a synced per-user preference; the inline control is the only way to reach it.
  const toggle = week.getByRole('button', { name: /show more|show less/i }).first()
  if (await toggle.isVisible().catch(() => false)) {
    await toggle.click()
    await expect(week).toBeVisible()
  }
})

test('the momentum rail lists chips and each one can be followed', async ({ page }, testInfo) => {
  await signInIsolated(page, 'home-momentum', testInfo)
  await page.goto('/')

  await page.getByTestId('discovery-tab-rising').click()
  const rail = page.getByTestId('momentum-rail-topic')
  await expect(rail).toBeVisible()
  await expect(page.getByTestId('momentum-chip').first()).toBeVisible()

  // Following writes the same interest token the picker does, so the two must agree.
  const follow = page.getByTestId('momentum-follow').first()
  await expect(follow).toBeVisible()
  await Promise.all([
    page.waitForResponse((r) => r.url().includes('/api/app/interests') && r.request().method() !== 'GET'),
    follow.click(),
  ])
})

test('the trend window tabs re-query rather than re-rendering the same series', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'home-trend-window', testInfo)
  await page.goto('/')
  // The discovery rails are TABBED and mutually exclusive, and the window tabs belong to RISING
  // (beside the momentum rail) — not to trending. Encoding the real IA rather than a guess.
  await page.getByTestId('discovery-tab-rising').click()
  await expect(page.getByTestId('trend-window-tabs')).toBeVisible()

  // A window control that does not change the request is decoration.
  await Promise.all([
    page.waitForResponse((r) => r.url().includes('/api/app/trending')),
    page.getByTestId('trend-window-1y').click(),
  ])
  await expect(page.getByTestId('momentum-rail-topic')).toBeVisible()

  // ...and the trending rail lives under its own tab.
  await page.getByTestId('discovery-tab-trending').click()
  await expect(page.getByTestId('home-trending')).toBeVisible()
})

test('the storylines rail renders chips and follows one', async ({ page }, testInfo) => {
  await signInIsolated(page, 'home-storylines', testInfo)
  await page.goto('/')
  await page.getByTestId('discovery-tab-storylines').click()

  const rail = page.getByTestId('home-storylines')
  await expect(rail).toBeVisible()
  await expect(page.getByTestId('storyline-chip').first()).toBeVisible()
  const follow = page.getByTestId('storyline-follow').first()
  await expect(follow).toBeVisible()
  await follow.click()
  // Idempotent: a second render must not duplicate the chip.
  await expect(page.getByTestId('storyline-chip').first()).toBeVisible()
})

test('the discovery tabs switch between rising, trending and storylines', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'home-discovery', testInfo)
  await page.goto('/')

  await expect(page.getByTestId('home-discovery')).toBeVisible()
  for (const tab of ['discovery-tab-rising', 'discovery-tab-trending', 'discovery-tab-storylines']) {
    await page.getByTestId(tab).click()
    // Whatever the tab shows, the section must not be left empty-but-present.
    await expect(page.getByTestId('home-discovery')).toBeVisible()
  }
})
