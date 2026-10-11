import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Licence and copyright, as each show's feed declares it (#2325), against the fixture corpus:
 * p05 carries a copyright notice (the shape 79 of 80 production feeds have) and one episode,
 * p05_e05, its own item-level <podcast:license>; p02 a recognised Creative Commons licence; p03
 * nothing. The line under the show artwork and the line at the foot of the player's About both
 * open the SAME licence page; from an episode with its own declaration that page leads with it.
 */

test('the show line opens the licence page with the notice in full', async ({ page }, testInfo) => {
  await signInIsolated(page, 'licence-show', testInfo)
  await page.goto('/podcast/p05')
  const link = page.getByTestId('podcast-licence-link')
  await expect(link).toHaveText('© 2025 Long Horizon Media. All rights reserved.')
  await link.click()
  await expect(page).toHaveURL(/\/podcast\/p05\/licence$/)
  await expect(page.getByTestId('podcast-licence-raw')).toHaveText(
    '© 2025 Long Horizon Media. All rights reserved.',
  )
  await expect(page.getByTestId('podcast-licence-page')).toContainText('not a licence grant')
})

test('a recognised licence shows its conditions; a silent feed says so', async ({ page }, testInfo) => {
  await signInIsolated(page, 'licence-kinds', testInfo)
  await page.goto('/podcast/p02/licence')
  const cards = page.getByTestId('podcast-licence-declaration')
  await expect(cards).toHaveCount(2)
  await expect(cards.nth(1)).toContainText('Recognised as CC BY-NC-SA 4.0')
  await expect(cards.nth(1)).toContainText('Commercial use: no')
  await expect(cards.nth(1)).toContainText('Applies to: audio and video')

  await page.goto('/podcast/p03')
  await expect(page.getByTestId('podcast-licence-link')).toHaveText('No explicit licence found')
  await page.getByTestId('podcast-licence-link').click()
  await expect(page.getByTestId('podcast-licence-none')).toContainText('not the same as unrestricted')
})

test("About's line opens the same page, led by the episode's own declaration", async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'licence-episode', testInfo)
  const eps = (await (await page.request.get('/api/app/podcasts/p05/episodes?page_size=50')).json()) as {
    items: { slug: string; title: string }[]
  }
  // Find p05_e05 by its own declaration, not by position.
  let slugWithOwn = ''
  for (const e of eps.items) {
    const d = (await (await page.request.get(`/api/app/episodes/${e.slug}`)).json()) as {
      licence_scope?: string
    }
    if (d.licence_scope === 'episode') slugWithOwn = e.slug
  }
  expect(slugWithOwn, 'p05_e05 carries its own <podcast:license>').not.toBe('')

  await page.goto(`/episode/${slugWithOwn}`)
  await page.getByTestId('player-open-description').click()
  const line = page.getByTestId('episode-description-licence')
  await expect(line).toHaveText('CC BY 4.0')
  await line.click()
  await expect(page).toHaveURL(new RegExp(`/podcast/p05/licence\\?episode=${slugWithOwn}$`))
  await expect(page.getByTestId('podcast-licence-episode')).toContainText('cc-by-4.0')
  // The show's notice is still on the same page, below.
  await expect(page.getByTestId('podcast-licence-raw').last()).toHaveText(
    '© 2025 Long Horizon Media. All rights reserved.',
  )
})
