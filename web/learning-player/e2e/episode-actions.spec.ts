import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * EpisodeActions (UXS-014) — the standard episode action row, the fixed set (favorite, queue,
 * download, add-to-collection) every episode surface shows. REAL API over the committed corpus, NO mocks. Download
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
  //
  // Normalise FIRST. This identity is stable across runs and its queue lives in the API's own data
  // dir, which `globalSetup` does not clear (it only wipes the local APP_DATA_DIR) — so anything
  // else that queued this episode against the same fixture API leaves the row already showing
  // "Remove from queue", and an unconditional wait for "Add to queue" then times out on a row that
  // is working perfectly. Assert the ROUND TRIP, not the starting state (2026-09-16).
  const queued = row.getByRole('button', { name: 'Remove from queue' })
  if (await queued.isVisible()) {
    await queued.click()
    await expect(row.getByRole('button', { name: 'Add to queue' })).toBeVisible()
  }

  const queueAdd = row.getByRole('button', { name: 'Add to queue' })
  await expect(queueAdd).toBeVisible()
  await queueAdd.click()
  await expect(row.getByRole('button', { name: 'Remove from queue' })).toBeVisible()
})
