import { expect, test } from '@playwright/test'
import { openTranscript, signInIsolated } from './helpers'

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
  await page.goto('/podcast/p05') // #1148: reach the episode via its show page (date-independent)
  await page.getByText('Index Investing Without the Myths').first().click()
  await openTranscript(page) // transcript is opt-in on mobile — reveal it (no-op on desktop)
  await expect(page.getByText(/Index funds are not a strategy/).first()).toBeVisible()
  const epFav = page.getByRole('button', { name: 'Save to favorites' }).first()
  if (await epFav.isVisible().catch(() => false)) await epFav.click()
  await expect(page.getByRole('button', { name: 'Remove from favorites' }).first()).toBeVisible()

  // #1593 — an insight has exactly ONE save, and it goes to Highlights. The favourite heart used to
  // sit alongside it, writing the SAME insight to a SECOND list (Saved › Insights); this spec used
  // to exercise that duplicate path. Insight favourites are now legacy read-only: nothing writes
  // them, so there is no heart on an insight row to click.
  await page.getByRole('button', { name: 'Insights' }).first().click()
  const kp = page.getByTestId('kp-insights')
  await expect(kp).toBeVisible()
  await expect(kp.getByRole('button', { name: 'Save to favorites' })).toHaveCount(0)

  // The bookmark is the one save, and it lands in Highlights.
  await kp.getByRole('button', { name: 'Save to highlights' }).first().click()

  // Saved holds the favourited EPISODE; the insight went to Highlights, not here.
  await page.goto('/library')
  await page.getByRole('button', { name: 'Saved' }).click()
  await expect(page.getByRole('heading', { name: 'Episodes' })).toBeVisible()
  await expect(page.getByText('Nothing saved yet.', { exact: false })).toHaveCount(0)

  await page.getByRole('button', { name: 'Highlights' }).click()
  await expect(page.getByText('No highlights yet.', { exact: false })).toHaveCount(0)
})
