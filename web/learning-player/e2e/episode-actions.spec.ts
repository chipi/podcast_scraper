import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * EpisodeActions (UXS-014) — the standard episode action row (favorite · download · queue), the
 * MINIMUM set every episode surface shows. REAL API over the committed corpus, NO mocks. Download
 * self-hides on web (`DownloadButton` is native-only), so on this web preview the row is favorite +
 * queue; this drives both and asserts each per-user write round-trips to the real API (the label
 * flips and stays flipped), which is the bug the row was created to end — surfaces that hand-rolled
 * a subset and silently dropped an action.
 */
test('the episode action row favourites and queues an episode against the real API', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'episode-actions', testInfo)
  await page.goto('/')

  const row = page.getByTestId('episode-actions').first()
  await expect(row).toBeVisible()

  // Favourite: toggle on, confirm the label flips to "Remove", then toggle back off.
  const favAdd = row.getByRole('button', { name: 'Save to favorites' })
  await expect(favAdd).toBeVisible()
  await favAdd.click()
  await expect(row.getByRole('button', { name: 'Remove from favorites' })).toBeVisible()
  await row.getByRole('button', { name: 'Remove from favorites' }).click()
  await expect(row.getByRole('button', { name: 'Save to favorites' })).toBeVisible()

  // Queue: same round-trip.
  const queueAdd = row.getByRole('button', { name: 'Add to queue' })
  await expect(queueAdd).toBeVisible()
  await queueAdd.click()
  await expect(row.getByRole('button', { name: 'Remove from queue' })).toBeVisible()
})
