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

  await page.getByRole('tab', { name: 'Collections' }).click()
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

  // #1593 — an insight has exactly ONE save, and it goes to Highlights. The favourite heart used to
  // sit alongside it, writing the SAME insight to a SECOND list (Saved › Insights); this spec used
  // to exercise that duplicate path. Insight favourites are now legacy read-only: nothing writes
  // them, so there is no heart on an insight row to click.
  await page.getByRole('button', { name: 'Insights' }).first().click()
  const kp = page.getByTestId('kp-insights')
  await expect(kp).toBeVisible()
  await expect(kp.getByRole('button', { name: 'Save to favorites' })).toHaveCount(0)

  // The bookmark is the one save, and it lands in Highlights.
  //
  // Guarded the same way the episode favourite above is, and for the same reason: the save
  // persists server-side under this user, so on a second run against the same api container the
  // insight is ALREADY saved, the button no longer reads "Save to highlights", and the click hangs
  // for the full timeout. The spec's contract is "the insight ends up in Highlights", not "the
  // button was clicked" — so only ever ADD, and assert the outcome either way.
  const save = kp.getByRole('button', { name: 'Save to highlights' }).first()
  if (await save.isVisible().catch(() => false)) await save.click()

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
