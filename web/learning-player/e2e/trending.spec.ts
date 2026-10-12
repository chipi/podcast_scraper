import { expect, test } from '@playwright/test'
import { signInIsolated, showEveryonesTrends } from './helpers'

/**
 * Discovery list on Discover (Trends left Home 2026-10-07) — REAL API over the COMMITTED validation corpus, NO mocks. The e2e
 * webServer pins APP_TRENDING_NOW=2026-07-20 (just after the corpus's newest episode) so the
 * read-time momentum is deterministic and the risk/systems content reads as rising; GET
 * /api/app/trending?kind=topic then returns those topics and the list renders.
 *
 * Rising topics from the committed corpus at that anchor (natural topics since 2026-10-12): the
 * newest episodes' own subjects, all new in the window and so tied at the top momentum, in id
 * order — "canonical labeling", "code switching", … The stamped "risk management" that led before
 * was on every show's newest episodes because the generator put it there.
 */
test('Discover shows the Rising-now discovery list with rising topics', async ({ page }, testInfo) => {
  await signInIsolated(page, 'trending-rising', testInfo)
  await page.goto('/explore')
  await showEveryonesTrends(page)

  // #4 folded the three "what's hot" rails into one tabbed area; the default discovery TAB is
  // Topics, and Rising sort is the default — the discovery list renders immediately.
  const topicTab = page.getByTestId('discovery-tab-topic')
  await expect(topicTab).toBeVisible()
  await expect(topicTab).toHaveAttribute('aria-selected', 'true')

  const list = page.getByTestId('discovery-list-topic')
  await expect(list).toBeVisible()

  // At least one discovery row, and it carries a velocity multiplier (↑N×) — the momentum signal.
  await expect(page.getByTestId('discovery-row').first()).toBeVisible()
  await expect(list).toContainText('×')
  // The newest episodes' own subjects are what's rising at the pinned anchor.
  await expect(list.getByText(/canonical labeling|code switching/i).first()).toBeVisible()
})

test('signed in: following a trending topic from the list toggles to followed', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'trending', testInfo)
  await page.goto('/explore')
  await showEveryonesTrends(page)

  const list = page.getByTestId('discovery-list-topic')
  await expect(list).toBeVisible()
  const follow = list.locator('[data-testid="discovery-follow"]').first()
  await expect(follow).toBeVisible()

  // Idempotent by construction: assert the button TOGGLES, whichever state it starts in.
  //
  // This used to require `aria-pressed="false"` up front, which made the test pass exactly once
  // per api container. The follow persists server-side under this user, so a second run of the
  // suite against the same long-lived container met a topic it had already followed and failed on
  // its very first assertion — before clicking anything, so a retry could never recover either.
  // The bug looked like a product regression and was a test that had eaten its own precondition.
  //
  // Toggling from whatever is there also tests MORE: it covers the unfollow path, which the
  // one-way version never touched.
  const before = await follow.getAttribute('aria-pressed')
  const after = before === 'true' ? 'false' : 'true'

  await follow.click()
  await expect(follow).toHaveAttribute('aria-pressed', after) // persisted to the interests store

  // Back to where we found it, so this spec leaves no trace for the next run — the property whose
  // absence caused the failure above.
  await follow.click()
  await expect(follow).toHaveAttribute('aria-pressed', before ?? 'false')
})
