import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * The publisher's episode description in the player (operator 2026-10-10): cards clamped it and the
 * player never showed it. A Description pill beside Episode notes opens the whole text in a sheet.
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

test('the Description pill opens the full publisher text, and the sheet closes again', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'p05', 'Index Investing Without the Myths')

  const pill = page.getByTestId('player-open-description')
  await expect(pill).toBeVisible()
  // Inside the artwork, beside Episode notes — never pushed off a phone screen.
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

test('an episode whose feed gave no description has no Description pill', async ({ page }, testInfo) => {
  await openEpisode(page, testInfo, 'p10', 'Construyendo Senderos Que Duran')
  await expect(page.getByTestId('player-open-description')).toHaveCount(0)
})
