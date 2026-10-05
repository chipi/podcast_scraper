import { expect, test, type Page } from '@playwright/test'
import { expectSignedIn } from '../helpers'

/**
 * Every kind's Share card, as the app's own "Share card" button produces it (operator 2026-10-05).
 *
 * The card is the SERVER's (`server/og/card.py`) — one design for every kind. On a desktop browser
 * there is no file share, so the card arrives as a PNG DOWNLOAD: exactly the image a phone would
 * hand to its share sheet. One file per kind, so the set can be reviewed side by side.
 *
 *   npm run design:shots -- share-cards --project=desktop
 */
const VARIANT = process.env.DESIGN_VARIANT || 'baseline'
const out = (name: string) => `design-results/${VARIANT}/${test.info().project.name}/share-${name}.png`

/** Open the page's Share menu and save what "Share card" produces. */
async function saveCard(page: Page, name: string): Promise<void> {
  const trigger = page.getByTestId('share-menu').first()
  await expect(trigger).toBeVisible()
  await trigger.click()
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByTestId('share-card').first().click(),
  ])
  await download.saveAs(out(name))
}

test.beforeEach(async ({ page }) => {
  await page.goto(`/api/app/auth/login?as=design-share-${Date.now().toString(36)}`)
  await expectSignedIn(page)
})

test('episode', async ({ page }) => {
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await expect(page.getByRole('heading', { name: /Index Investing Without the Myths/ }).first()).toBeVisible()
  await saveCard(page, 'episode')
})

for (const [name, path] of [
  ['show', '/podcast/p05'],
  ['topic', '/topic/topic:risk-management'],
  ['person', '/person/person:nora'],
  ['storyline', '/storyline/topic:risk-management'],
  ['theme', '/theme/tc:broadcast-format'],
] as const) {
  test(name, async ({ page }) => {
    await page.goto(path)
    await page.waitForLoadState('networkidle')
    await saveCard(page, name)
  })
}
