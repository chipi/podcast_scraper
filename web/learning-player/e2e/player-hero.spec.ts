import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * Player hero geometry (UXS-014 "Player hero"): 5:4 on phones, 1:1 from `lg`.
 *
 * The reason for 5:4 is a measured outcome, so the outcome is what is asserted: on an iPhone-sized
 * screen the WHOLE transport — buttons, scrubber, density strip, timestamps — sits above the tab
 * bar without scrolling. A full-width square put the timestamps under it (2026-09-30).
 */
async function openPlayer(page: import('@playwright/test').Page, testInfo: import('@playwright/test').TestInfo, tag: string) {
  await signInIsolated(page, tag, testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page.getByTestId('player-hero')).toBeVisible()
  await expect(page.getByTestId('player-times')).toBeVisible()
}

test('phone: the hero is 5:4 and the whole transport is above the tab bar', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'mobile-chrome', 'phone geometry')
  // iPhone 14/15 CSS viewport — the device class the report came from.
  await page.setViewportSize({ width: 390, height: 844 })
  await openPlayer(page, testInfo, 'player-hero-phone')

  const hero = (await page.getByTestId('player-hero').boundingBox())!
  expect(hero.width / hero.height, `hero is ${hero.width}x${hero.height}`).toBeCloseTo(1.25, 1)

  const times = (await page.getByTestId('player-times').boundingBox())!
  const nav = (await page.getByTestId('bottom-nav').boundingBox())!
  expect(
    times.y + times.height,
    `the timestamps end at ${Math.round(times.y + times.height)}px, under the tab bar at ${Math.round(nav.y)}px`,
  ).toBeLessThanOrEqual(nav.y)

  // The timeline reads scrubber → density strip → times (UXS-011): the two strips annotate the
  // same span and belong together; the numbers label their ends.
  const scrub = (await page.locator('input[type="range"]').first().boundingBox())!
  const density = (await page.getByTestId('player-insight-density').boundingBox())!
  expect(scrub.y).toBeLessThan(density.y)
  expect(density.y).toBeLessThan(times.y)
})

test('desktop: the hero stays square', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop-chrome', 'desktop geometry')
  await openPlayer(page, testInfo, 'player-hero-desktop')
  const hero = (await page.getByTestId('player-hero').boundingBox())!
  expect(hero.width / hero.height, `hero is ${hero.width}x${hero.height}`).toBeCloseTo(1, 1)
})
