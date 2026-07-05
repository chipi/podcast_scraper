import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Library hub (this session): the Saved tab shows per-kind sections (Episodes, Insights) instead of
 * one flat list, and every tab has a real empty state. Real API + committed corpus.
 *
 * Two isolated users so the assertions stay idempotent across retries / both projects: one user is
 * NEVER written to (empty states), the other only ever ADDS favourites (sections present).
 */
test('Library tabs show real empty states for a fresh user', async ({ page }, testInfo) => {
  await signInIsolated(page, 'library-empty', testInfo)
  await page.goto('/library')

  // Saved is the default tab — empty for a fresh user.
  await page.getByRole('button', { name: 'Saved' }).click()
  await expect(page.getByText('Nothing saved yet.', { exact: false })).toBeVisible()

  await page.getByRole('button', { name: 'Highlights' }).click()
  await expect(page.getByText('No highlights yet.', { exact: false })).toBeVisible()

  await page.getByRole('button', { name: 'Queue' }).click()
  await expect(page.getByText('Your queue is empty.', { exact: false })).toBeVisible()
})

test('favouriting an episode + an insight fills the Saved per-kind sections', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'library-fill', testInfo)

  // Favourite the episode from its player screen (the heart). Guarded: only ever ADD.
  await page.goto('/')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page.getByText(/Index funds are not a strategy/).first()).toBeVisible()
  const epFav = page.getByRole('button', { name: 'Save to favorites' }).first()
  if (await epFav.isVisible().catch(() => false)) await epFav.click()
  await expect(page.getByRole('button', { name: 'Remove from favorites' }).first()).toBeVisible()

  // Favourite an insight from the Knowledge Panel (a different favourite KIND → its own section).
  await page.getByRole('button', { name: 'Insights' }).first().click()
  const insFav = page.getByRole('button', { name: 'Save to favorites' }).first()
  if (await insFav.isVisible().catch(() => false)) await insFav.click()

  // The Saved tab now shows BOTH per-kind sections, and the empty state is gone.
  await page.goto('/library')
  await page.getByRole('button', { name: 'Saved' }).click()
  await expect(page.getByRole('heading', { name: 'Episodes' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Insights' })).toBeVisible()
  await expect(page.getByText('Nothing saved yet.', { exact: false })).toHaveCount(0)
})
