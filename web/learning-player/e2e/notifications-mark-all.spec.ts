import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * "Mark all read" clears every dot, even while a refresh is in flight (UXS-011 NotificationsBell).
 *
 * The defect (2026-09-30): opening the bell starts a list refresh. A "Mark all read" tapped before
 * that refresh landed was undone by it — the refresh's response had been computed before the
 * write, so it re-lit every item's dot. The badge said nothing was unread while the list said
 * everything was.
 *
 * A FOCUSED MOCK of the two notification calls, the same exception `perspectives.spec.ts` makes:
 * the corpus cannot create an in-app notification (there is no write route; they come from
 * server-side delivery jobs), and the race needs a response held back on purpose. Everything else
 * on the page is the real backend.
 */
test('mark all read survives a refresh that was already in flight', async ({ page }, testInfo) => {
  const item = (id: string, title: string) => ({
    id,
    type: 'new_episodes',
    title,
    body: null,
    deep_link: '/',
    read: false,
    created_at: Math.floor(Date.now() / 1000) - 120,
  })
  const unreadList = { items: [item('n1', 'Two new episodes'), item('n2', 'Your weekly digest')], unread: 2 }

  let calls = 0
  let answered = 0
  await page.route('**/api/app/notifications', async (route) => {
    calls++
    // The first load (app start) answers at once; the refresh on opening the bell is held back,
    // so the user's "Mark all read" lands while it is still in flight — the exact ordering of the
    // bug. It answers with the stale, all-unread list the server had before the write.
    if (calls > 1) await new Promise((r) => setTimeout(r, 1500))
    await route.fulfill({ json: unreadList })
    answered++
  })
  await page.route('**/api/app/notifications/read-all', (route) => route.fulfill({ json: { unread: 0 } }))

  await signInIsolated(page, 'notifications-mark-all', testInfo)
  await page.goto('/')
  await expect(page.getByTestId('notifications-badge')).toHaveText('2')

  const before = calls
  await page.getByTestId('notifications-bell').click()
  // Opening the bell issued a refresh, and it is still held back when "Mark all read" is tapped.
  await expect.poll(() => calls).toBeGreaterThan(before)
  const inFlight = calls
  expect(answered, 'the refresh must still be in flight when the user marks read').toBeLessThan(inFlight)
  const items = page.getByTestId('notification-item')
  await expect(items).toHaveCount(2)
  await page.getByTestId('notifications-mark-all').click()

  // Let that refresh land, then check nothing was re-lit by it.
  await expect.poll(() => answered, { timeout: 10_000 }).toBeGreaterThanOrEqual(inFlight)
  await expect(page.getByTestId('notifications-badge')).toHaveCount(0)
  for (const i of await items.all()) await expect(i).toHaveAttribute('data-read', 'true')
})
