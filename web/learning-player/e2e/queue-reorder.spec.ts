import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Queue reorder (this session): the ↑/↓ chevrons moved into the card's icon row. Real API +
 * committed corpus. Add two distinct episodes, then move the first one down and assert the order
 * actually changed — robust across retries (swapping two items always changes which is first).
 */
// Two DISTINCT, non-featured episodes (the newest, "Index Investing", is the catalog hero and would
// dedupe against its own list card → only one unique slug queued). These are different episodes.
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
    await page.goto(`/podcast/${show}`) // #1148: reach each episode via its show page
    const card = page.locator('article').filter({ hasText: title }).first()
    const btn = card.getByRole('button', { name: /queue/i }).first()
    await expect(btn).toBeVisible()
    if ((await btn.getAttribute('aria-label')) === 'Add to queue') await btn.click()
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
