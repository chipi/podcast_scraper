import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Queue reorder (this session): the ↑/↓ chevrons moved into the card's icon row. Real API +
 * committed corpus. Add two distinct episodes, then move the first one down and assert the order
 * actually changed — robust across retries (swapping two items always changes which is first).
 */
// Two DISTINCT episodes, each reached via its own show page (#1148: date-independent — no reliance
// on catalog ordering / the landing hero). Persisting each add before navigating away is essential:
// a full-page goto resets the Pinia queue store and can abort an in-flight PUT /queue.
const EP_A = 'Index Investing Without the Myths'
const EP_B = 'Wreck Diving Fundamentals'

async function firstQueuedTitle(page: import('@playwright/test').Page): Promise<string> {
  return (await page.locator('article').first().innerText()).trim()
}

test('reorder the queue with the ↑/↓ chevrons', async ({ page }, testInfo) => {
  await signInIsolated(page, 'queue-reorder', testInfo)

  // Add two distinct episodes from the catalog (guarded: only ever ADD).
  for (const { title, show } of [
    { title: EP_A, show: 'p05' },
    { title: EP_B, show: 'p03' },
  ]) {
    // A full-page goto resets the Pinia queue store; it re-hydrates via GET /queue. Adding before
    // that GET resolves lets the stale fetch (server state WITHOUT this episode) clobber the
    // optimistic add → the button never flips to Remove and the episode drops. Wait for the
    // hydration GET (armed before the goto so it can't be missed) before touching the button.
    const hydrated = page.waitForResponse(
      (r) => r.url().includes('/api/app/queue') && r.request().method() === 'GET',
    )
    await page.goto(`/podcast/${show}`) // #1148: reach each episode via its show page
    await hydrated
    const card = page.locator('article').filter({ hasText: title }).first()
    const btn = card.getByRole('button', { name: /queue/i }).first()
    await expect(btn).toBeVisible()
    if ((await btn.getAttribute('aria-label')) === 'Add to queue') {
      // Wait for the PUT /queue write to land before the next full-page goto, which would otherwise
      // reset the store and abort the in-flight persist → the episode silently drops from the queue.
      await Promise.all([
        page.waitForResponse(
          (r) => r.url().includes('/queue') && r.request().method() === 'PUT' && r.ok(),
        ),
        btn.click(),
      ])
    }
    await expect(card.getByRole('button', { name: 'Remove from queue' }).first()).toBeVisible()
  }

  await page.goto('/queue')
  await expect(page).toHaveURL(/\/queue/)
  const items = page.locator('article')
  await expect(items.nth(1)).toBeVisible() // two distinct episodes queued

  // Move the first item down → the item that was first is no longer first (swapping two always
  // changes which is first, so this holds regardless of the starting order / on retries).
  const firstBefore = await firstQueuedTitle(page)
  await items.first().getByRole('button', { name: 'Move down' }).click()
  await expect.poll(async () => firstQueuedTitle(page)).not.toBe(firstBefore)
})
