import { expect, test, type Page, type TestInfo } from '@playwright/test'
import { navTo, signInIsolated } from './helpers'

/**
 * Switching accounts INSIDE a running app never shows one account's data to the other
 * (operator 2026-10-07, on device).
 *
 * The native app signs in without a page load: the OAuth deep link comes back to the running
 * WebView and the shell refreshes `/auth/status`. Two things then went wrong, both because the tab
 * views are kept alive and had mounted for the PREVIOUS account:
 *  - a brand-new Apple account opened Profile and saw the Google account's recap — its half hour,
 *    six episodes and saved line;
 *  - back on Google, Library › Boards said "No collections yet" while the add-to-board pop-up,
 *    which fetches fresh, listed all of them.
 *
 * `signInIsolated` is a full page load, which rebuilds the app and so cannot reproduce this. The
 * switch here is the in-app path: the session cookie changes underneath the running SPA, and the
 * shell's reconnect handler re-reads who is signed in — the same `auth.refresh()` the deep link runs.
 */

function accountId(who: string, testInfo: TestInfo): string {
  return `${who}-${testInfo.project.name}`.toLowerCase().replace(/[^a-z0-9-]/g, '')
}

async function switchInApp(page: Page, who: string, testInfo: TestInfo): Promise<void> {
  const resp = await page.request.get(`/api/app/auth/login?as=${accountId(who, testInfo)}`)
  expect(resp.ok(), `mock sign-in as ${who} failed: ${resp.status()}`).toBeTruthy()
  await page.evaluate(() => window.dispatchEvent(new Event('online')))
}

async function seedListening(page: Page): Promise<void> {
  // Two saves of one episode two minutes apart in position: the recap counts the accrued delta.
  const eps = await page.request.get('/api/app/episodes?page_size=1')
  const slug = ((await eps.json()).items as Array<{ slug: string }>)[0].slug
  const tz = -new Date().getTimezoneOffset()
  for (const position_seconds of [10, 130]) {
    const r = await page.request.put(`/api/app/playback/${slug}`, {
      data: { position_seconds, tz_offset_minutes: tz },
    })
    expect(r.ok(), `seeding playback failed: ${r.status()}`).toBeTruthy()
  }
}

test("an in-app account switch never shows the other account's boards or recap", async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'switch-a', testInfo)
  const board = `A board ${testInfo.project.name} ${Date.now()}`
  expect((await page.request.post('/api/app/collections', { data: { name: board } })).ok()).toBeTruthy()
  await seedListening(page)

  // A: boards and recap both render, so both tabs have mounted for A.
  await page.goto('/library?tab=collections')
  await expect(page.locator('li', { hasText: board })).toHaveCount(1)
  await navTo(page, 'profile')
  await page.getByRole('tab', { name: 'Stats' }).click()
  await expect(page.getByText('Listened')).toBeVisible()

  // B, a brand-new account: none of A's listening, none of A's boards.
  await switchInApp(page, 'switch-b', testInfo)
  await expect(page.getByText('Listened')).toHaveCount(0)
  await navTo(page, 'library')
  await expect(page.locator('li', { hasText: board })).toHaveCount(0)

  // Back to A: A's boards come back — the half of the bug the operator hit on the Google account.
  //
  // The native sign-in path does NOT reload boards; this web switch rides the reconnect handler,
  // which does, and so hid the bug (the spec passed with the fix removed). Holding that one reload
  // open — never answered, so neither the server's list nor A's on-device cache arrives through it;
  // an ABORT fell back to the cache and hid the bug too — makes this switch behave like the native
  // one: only a fresh Library mount can bring A's boards back.
  let swallowed = false
  await page.route('**/api/app/collections', async (route) => {
    if (!swallowed && route.request().method() === 'GET') {
      swallowed = true
      return // held: never continued
    }
    return route.fallback()
  })
  await switchInApp(page, 'switch-a', testInfo)
  await expect.poll(() => swallowed).toBe(true)
  await navTo(page, 'library')
  await expect(page.locator('li', { hasText: board })).toHaveCount(1)
  await page.unrouteAll({ behavior: 'ignoreErrors' })
})
