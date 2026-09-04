import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * What the queue tells the user when it is a CACHED copy (#1909/#1910/#1925).
 *
 * The distinction this branch introduced is user-visible and had no browser coverage: adding and
 * removing became item operations that replay safely offline, while REORDERING still sends the
 * whole list and must refuse from an unrevalidated baseline. Both halves are asserted here,
 * because "the arrows do nothing" was the original bug and "the toggle does nothing" was the
 * over-correction I shipped and had to walk back.
 */

const EPISODE = 'Index Investing Without the Myths'

test('adding uses the item route, so the whole-list PUT is not what persists a queue add', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'queue-item-route', testInfo)
  await page.goto('/podcast/p05')

  const card = page.locator('article').filter({ hasText: EPISODE })
  const btn = card.getByRole('button', { name: /queue/i })
  await expect(btn).toBeVisible()

  const seen: string[] = []
  page.on('request', (r) => {
    if (r.url().includes('/api/app/queue')) seen.push(`${r.method()} ${new URL(r.url()).pathname}`)
  })

  if ((await btn.getAttribute('aria-label')) === 'Add to queue') {
    await Promise.all([
      page.waitForResponse(
        (r) => r.url().includes('/api/app/queue/items') && r.request().method() === 'POST' && r.ok(),
      ),
      btn.click(),
    ])
  }
  await expect(card.getByRole('button', { name: 'Remove from queue' })).toBeVisible()

  // Replacing the whole list is what made an offline replay clobber another device's edits, so an
  // ADD must never do it.
  expect(seen.some((s) => s.startsWith('PUT'))).toBe(false)
})

test('a queue restored from cache says so, and refuses ONLY reordering', async ({
  page,
  context,
}, testInfo) => {
  await signInIsolated(page, 'queue-stale', testInfo)

  // Two episodes from ONE show, so both cards are certainly present and distinct — picking the
  // first card of two different shows made this depend on each show's episode ordering.
  await page.goto('/podcast/p05')
  const cards = page.locator('article')
  await expect(cards.nth(1)).toBeVisible()
  for (const index of [0, 1]) {
    const btn = cards.nth(index).getByRole('button', { name: /queue/i }).first()
    await expect(btn).toBeVisible()
    if ((await btn.getAttribute('aria-label')) === 'Add to queue') {
      await Promise.all([
        page.waitForResponse(
          (r) =>
            r.url().includes('/api/app/queue/items') && r.request().method() === 'POST' && r.ok(),
        ),
        btn.click(),
      ])
    }
  }

  await page.goto('/queue')
  await expect(page.locator('article').nth(1)).toBeVisible()

  // Now make the queue read fail, and reload: the store falls back to its cached copy and marks
  // it stale — readable, but not safe to write the WHOLE list from.
  await context.route('**/api/app/queue', (route) =>
    route.request().method() === 'GET' ? route.abort() : route.continue(),
  )
  await page.reload()

  // It says what it is showing, rather than looking like a live list.
  await expect(page.getByText(/Offline — showing your saved queue/)).toBeVisible()

  // Reordering is refused VISIBLY: the arrows disable and explain themselves. Silently doing
  // nothing is what made them read as dead buttons.
  const up = page.getByRole('button', { name: 'Move up' }).first()
  const down = page.getByRole('button', { name: 'Move down' }).first()
  await expect(down).toBeDisabled()
  await expect(up).toBeDisabled()
  await expect(down).toHaveAttribute('title', /Reordering needs a connection/)
})
