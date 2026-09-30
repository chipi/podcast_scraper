import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * A paragraph-length show title cannot break a layout (operator 2026-09-30).
 *
 * "BRAVE Southeast Asia Tech: Singapore, Indonesia, …" ran eleven lines down the player kicker and
 * the show page heading on a phone. The fix is a line cap (`.lp-show-name`), not a shortened name.
 * jsdom does no layout, so only a real browser can say whether the cap actually holds.
 *
 * The committed corpus has no title this long, so the REAL responses are fetched and only the show
 * name is swapped — everything else on the page is the real backend, per the suite's contract. The
 * same focused-rewrite exception `perspectives.spec.ts` makes for a scale the corpus cannot reach.
 */

const LONG =
  'BRAVE Southeast Asia Tech: Singapore, Indonesia, Vietnam, Philippines, Thailand & Malaysia ' +
  'Startups, Founders & Venture Capital VC (English)'

async function withLongTitle(page: Page): Promise<void> {
  await page.route(/\/api\/app\/podcasts(\?|$)/, async (route) => {
    const resp = await route.fetch()
    const body = await resp.json()
    const list = Array.isArray(body) ? body : body.items ?? body.podcasts
    for (const p of list) if (p.feed_id === 'p05') p.title = LONG
    await route.fulfill({ response: resp, json: body })
  })
  await page.route(/\/api\/app\/episodes\/[^/?]+(\?|$)/, async (route) => {
    const resp = await route.fetch()
    const body = await resp.json()
    if (body && typeof body === 'object' && 'podcast_title' in body) body.podcast_title = LONG
    await route.fulfill({ response: resp, json: body })
  })
}

async function lines(page: Page, selector: string): Promise<number> {
  return page.locator(selector).first().evaluate((el) => {
    const lh = parseFloat(getComputedStyle(el).lineHeight)
    return Math.round(el.getBoundingClientRect().height / lh)
  })
}

test('the show page heading is capped at three lines, and Show more reveals all of it', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'long-show-title', testInfo)
  await withLongTitle(page)
  await page.goto('/podcast/p05')

  const heading = page.getByTestId('podcast-title')
  await expect(heading).toHaveText(LONG)
  expect(await lines(page, '[data-testid="podcast-title"]')).toBeLessThanOrEqual(3)
  // The full name stays in the DOM for screen readers and search — the cap is visual only.
  await expect(heading).toHaveAttribute('title', LONG)

  // Only a title the cap actually cuts earns "Show more". On a desktop viewport this one fits in
  // three lines, and a toggle there would expand nothing; on a phone it does not fit, which is the
  // reported case, so there it must be cut AND recoverable.
  const cut = await heading.evaluate((el) => el.scrollHeight - el.clientHeight > 1)
  if (testInfo.project.name.startsWith('mobile')) expect(cut).toBe(true)
  if (!cut) return
  await page.getByRole('button', { name: 'Show more' }).click()
  expect(await lines(page, '[data-testid="podcast-title"]')).toBeGreaterThan(3)
})

test('the player kicker caps the show name at two lines', async ({ page }, testInfo) => {
  await signInIsolated(page, 'long-show-title', testInfo)
  await withLongTitle(page)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  const kicker = 'a.lp-show-name[href*="/podcast/"]'
  await expect(page.locator(kicker).first()).toHaveText(LONG, { ignoreCase: true })
  expect(await lines(page, kicker)).toBeLessThanOrEqual(2)
})
