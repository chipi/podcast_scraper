import { expect, test, type Page } from '@playwright/test'
import { expectSignedIn } from '../helpers'

/**
 * Every kind's Share card — and a highlight's quote card — as the app's own buttons produce them
 * (operator 2026-10-05).
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

test('highlight', async ({ page }) => {
  // A real capture on a real episode, then the Saved row's own "Share as card" button.
  const eps = await (await page.request.get('/api/app/podcasts/p05/episodes')).json()
  const ep = (eps as { items: { slug: string; title: string }[] }).items.find((e) =>
    e.title.startsWith('Index Investing'),
  )!
  const created = await page.request.post('/api/app/highlights', {
    data: {
      episode_slug: ep.slug,
      kind: 'span',
      start_ms: 65_000,
      quote_text:
        "Index funds are not a strategy — they're the absence of one. You stop asking what you're " +
        'trying to achieve and start optimising fees on a goal you never named.',
      speaker: 'Daniel Cho',
    },
  })
  expect(created.ok()).toBe(true)
  await page.goto('/library?tab=saved')
  await page.waitForLoadState('networkidle')
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Share as card' }).first().click(),
  ])
  await download.saveAs(out('highlight'))
})

