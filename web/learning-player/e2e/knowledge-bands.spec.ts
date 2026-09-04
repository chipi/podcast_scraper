import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The knowledge bands that had unit tests and no e2e (E2E_SURFACE_MAP coverage gaps, 2026-09-03).
 *
 * All of them render the knowledge layer where the listener already is, and all of them follow one
 * rule that only a browser can check: **absent intelligence omits cleanly**. A unit test feeds a
 * component props and sees it draw; it cannot see a band left half-rendered against a real corpus,
 * which is the failure these guard.
 */

test('the podcast signals band separates DISTINCTIVE topics from the rest', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'podcast-signals', testInfo)
  await page.goto('/podcast/p05')

  const band = page.getByTestId('podcast-signals')
  await expect(band).toBeVisible()

  // The split is the point (UXS-013): topics with `lift` above the corpus base rate are what set
  // this show apart, and they are listed under their own heading. Collapsing the two groups is how
  // a show's signature topic lost an alphabetical tiebreak to wallpaper every show covers.
  await expect(page.getByTestId('ps-distinctive-heading')).toBeVisible()
  await expect(page.getByTestId('ps-distinctive-topic').first()).toBeVisible()
  await expect(page.getByTestId('ps-topics-heading')).toBeVisible()
})

test('the show activity chart renders one bar per period', async ({ page }, testInfo) => {
  await signInIsolated(page, 'show-activity', testInfo)
  await page.goto('/podcast/p05')

  await expect(page.getByTestId('show-activity')).toBeVisible()
  // "Is this show alive?" is the question, so more than one bucket has to render for the shape to
  // mean anything.
  expect(await page.locator('[data-testid^="show-activity-bar-"]').count()).toBeGreaterThan(1)
})

test('the player insight density band shows where the insights sit', async ({ page }, testInfo) => {
  await signInIsolated(page, 'insight-density', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  await expect(page.getByTestId('player-insight-density')).toBeVisible()
  // One band element per tick — `.first()` because the testid is the SERIES, not a singleton.
  await expect(page.getByTestId('player-density-band').first()).toBeVisible()
  await expect(page.getByTestId('player-density-tick').first()).toBeVisible()

  // Ticks mark WHERE the insights are; more than one is what makes the band informative rather
  // than decorative.
  expect(await page.getByTestId('player-density-tick').count()).toBeGreaterThan(1)

  // NOTE: `density-early|mid|late` belong to `EpisodeDensity`, a DIFFERENT component on a
  // different surface (asserted by `consolidation.spec.ts`). Conflating the two here would have
  // this spec fail for a reason that has nothing to do with the player band.
})

test('the knowledge panel opens in place and closes with Escape', async ({ page }, testInfo) => {
  await signInIsolated(page, 'knowledge-panel', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()

  const open = page.getByTestId('player-open-insights')
  await expect(open).toBeVisible()
  await open.click()

  const panel = page.getByTestId('knowledge-panel')
  await expect(panel).toBeVisible()
  await expect(page.getByTestId('kp-insights')).toBeVisible()

  // UXS-014: on MOBILE the panel is a modal dialog and must be dismissible from the keyboard; on
  // desktop it is a side column that is always present and has nothing to dismiss. Asserting
  // Escape unconditionally would demand the desktop layout behave like the phone one.
  const isDialog = (await panel.getAttribute('role')) === 'dialog'
  if (isDialog) {
    await page.keyboard.press('Escape')
    await expect(panel).toBeHidden()
  } else {
    await expect(panel).toBeVisible()
  }
})
