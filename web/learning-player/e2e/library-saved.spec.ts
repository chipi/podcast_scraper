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

  // Saved holds three things — favourited episodes, kept insights, and marked moments — and when it
  // holds none of them it says so ONCE (#1962). It used to render only the Highlights section's own
  // "No highlights yet.", because Highlights was the single unconditional section: a fresh account
  // met one orphan heading naming a third of the tab, and read the tab as redundant. Every section
  // is conditional now, and the tab speaks for itself.
  // Library's tab strip is `Tabs.vue` now, so these are `role="tab"` (#1594 item 7). They
  // previously carried NO role at all — which is why `getByRole('button')` matched them, and
  // why the strip did not announce as tabs to anyone using one.
  await page.getByRole('tab', { name: 'Saved' }).click()
  await expect(page.getByText('No highlights yet.', { exact: false })).toHaveCount(0)
  await expect(page.getByText('Episodes you favourite', { exact: false })).toBeVisible()
  // An empty state with nothing to do is a dead end.
  await expect(page.getByRole('link', { name: /Find something to listen to/ })).toBeVisible()

  await page.getByRole('tab', { name: 'Boards' }).click()
  await expect(page.getByText('No collections yet', { exact: false })).toBeVisible()
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

  // #1593 + F2.1 — an insight has exactly ONE save, the shared "Save to favorites" heart (F2.1
  // unified the save wording to "favorite"). It still routes to HIGHLIGHTS, not the /favorites list:
  // an insight is not a favourite there (the API 422s it), so the heart writes it to the
  // capture/highlights corpus. The old duplicate favourite path is gone; this is the one save.
  //
  // Guarded the same way the episode favourite above is: the save persists server-side under this
  // user, so on a second run against the same api container the insight is ALREADY saved and the
  // heart reads "Remove from favorites". The contract is "the insight ends up in Highlights", so
  // only ever ADD and assert the outcome.
  await page.getByRole('button', { name: 'Insights' }).first().click()
  const kp = page.getByTestId('kp-insights')
  await expect(kp).toBeVisible()
  const insightSave = kp.getByRole('button', { name: 'Save to favorites' }).first()
  if (await insightSave.isVisible().catch(() => false)) await insightSave.click()
  await expect(kp.getByRole('button', { name: 'Remove from favorites' }).first()).toBeVisible()

  // Saved (default tab) holds the favourited EPISODE in its "Episodes" section; the insight went to
  // the Highlights section (also inside Saved now), so the Highlights empty state is gone. Both live
  // in the one Saved tab after the beta consolidation — no separate Highlights tab to click.
  await page.goto('/library')
  await page.getByRole('tab', { name: 'Saved' }).click()
  await expect(page.getByRole('heading', { name: 'Episodes' })).toBeVisible()
  // The Highlights SECTION appears now that there is something in it — the inverse of the empty
  // case above, so "hide it always" could not pass both.
  await expect(page.getByRole('heading', { name: 'Highlights' })).toBeVisible()
  await expect(page.getByText('Episodes you favourite', { exact: false })).toHaveCount(0)
})
