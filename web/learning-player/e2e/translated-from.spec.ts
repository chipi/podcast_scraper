import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * D-36 slice b (operator 2026-10-13): a translated quote says so — REAL API over the COMMITTED
 * corpus, NO mocks.
 *
 * The fixture's non-English shows (p10 es … p14 pt) carry the English GI a real translated episode
 * has: English quotes, each with the translation provenance the pipeline writes. The Brief labels
 * those quotes "Translated from Spanish"; an English episode's quotes carry no label; and a saved
 * translated insight keeps the label in Saved, because a capture is a copy of the words.
 *
 * Idempotency: the save is guarded on its pressed state (never toggled off), so parallel projects
 * sharing a store only ever add.
 */
test('a translated quote is labelled in the Brief, and keeps the label when saved', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'translated-from', testInfo)

  await page.goto('/podcast/p10')
  await page.getByText('Construyendo Senderos Que Duran').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await page.getByTestId('player-open-insights').click()

  const label = page.getByTestId('translated-from').first()
  await expect(label).toHaveText('Translated from Spanish')
  await expect(label).toHaveAttribute('data-lang', 'es')

  // Save the insight whose quote carries the label.
  const card = page.locator('li').filter({ has: label }).first()
  const save = card.getByTestId('highlight-toggle').first()
  if ((await save.getAttribute('aria-pressed')) !== 'true') await save.click()
  await expect(save).toHaveAttribute('aria-pressed', 'true')

  await page.goto('/library')
  const saved = page
    .getByTestId('highlight-card')
    .filter({ has: page.getByTestId('translated-from') })
    .first()
  await expect(saved.getByTestId('translated-from')).toHaveText('Translated from Spanish')
})

test('an English episode carries no translation label', async ({ page }, testInfo) => {
  await signInIsolated(page, 'translated-from-en', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await page.getByTestId('player-open-insights').click()
  await expect(page.getByTestId('insight-type').first()).toBeVisible()
  await expect(page.getByTestId('translated-from')).toHaveCount(0)
})
