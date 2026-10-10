import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * The publisher's episode description in the player (operator 2026-10-10): cards clamped it and the
 * player never showed it. The Description door on the obi opens the whole text in a sheet.
 *
 * Links in it are covered by the unit tests (`EpisodeDescriptionSheet.test.ts`, `linkify.test.ts`):
 * no fixture description carries a URL.
 */
async function openEpisode(
  page: import('@playwright/test').Page,
  testInfo: import('@playwright/test').TestInfo,
  show: string,
  title: string,
) {
  await signInIsolated(page, `episode-description-${show}`, testInfo)
  await page.goto(`/podcast/${show}`)
  await page.getByText(title).first().click()
  await expect(page.getByTestId('player-hero')).toBeVisible()
}

test('the Description door opens the full publisher text, and the sheet closes again', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'p05', 'Index Investing Without the Myths')

  const pill = page.getByTestId('player-open-description')
  await expect(pill).toBeVisible()
  // Inside the artwork, on the obi down its right edge, below Brief — never pushed off a phone screen.
  const hero = (await page.getByTestId('player-hero').boundingBox())!
  const box = (await pill.boundingBox())!
  expect(box.x).toBeGreaterThanOrEqual(hero.x)
  expect(box.x + box.width).toBeLessThanOrEqual(hero.x + hero.width)

  await pill.click()
  const sheet = page.getByTestId('episode-description')
  await expect(sheet).toBeVisible()
  await expect(page.getByTestId('episode-description-text')).toHaveText(
    'What indexing does well, where people still make mistakes, and how to think about fees and behavior.',
  )
  await expect(page.getByTestId('episode-description-episode')).toHaveText('Index Investing Without the Myths')

  await page.getByTestId('episode-description-close').click()
  await expect(sheet).toBeHidden()

  await pill.click()
  await expect(sheet).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(sheet).toBeHidden()
})

test('an episode whose feed gave no description has no Description door', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'p10', 'Construyendo Senderos Que Duran')
  await expect(page.getByTestId('player-open-description')).toHaveCount(0)
})

test('the obi: a 44px band flush to the artwork\'s right edge, Moments · Brief · About, equal', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'p05', 'Index Investing Without the Myths')
  const hero = (await page.getByTestId('player-hero').boundingBox())!
  const obi = (await page.getByTestId('player-obi').boundingBox())!
  expect(Math.round(obi.width), 'the band is a 44px touch target').toBe(44)
  expect(Math.abs(obi.x + obi.width - (hero.x + hero.width))).toBeLessThanOrEqual(1.5)
  const doors = [
    (await page.getByTestId('player-open-moments').boundingBox())!,
    (await page.getByTestId('player-open-insights').boundingBox())!,
    (await page.getByTestId('player-open-description').boundingBox())!,
  ]
  // Top to bottom, equal heights (operator 2026-10-10: no doors sized to their words), each >= 44.
  expect(doors[0].y).toBeLessThan(doors[1].y)
  expect(doors[1].y).toBeLessThan(doors[2].y)
  for (const d of doors) {
    expect(d.height).toBeGreaterThanOrEqual(44)
    expect(Math.abs(d.height - doors[0].height)).toBeLessThanOrEqual(1.5)
  }
  await expect(page.getByTestId('player-open-moments')).toHaveText('Moments')
  await expect(page.getByTestId('player-open-insights')).toHaveText('Brief')
  await expect(page.getByTestId('player-open-description')).toHaveText('About')
})
