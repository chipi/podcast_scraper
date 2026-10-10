import { expect, test } from '@playwright/test'
import { listenToOne, signInIsolated } from './helpers'

/**
 * Browse › Episodes: "Which episodes" (All · Shows I follow · Mine) is its own row, so it combines
 * with the state filter (operator 2026-10-09). What's new's "all ›" and Discover 2 open
 * `?from=following&state=unplayed` — your shows' episodes, all of them, not the top 5.
 * REAL API over the committed corpus, NO mocks.
 */
test('Shows I follow + Unplayed lists every episode of the followed show, and nothing else', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'browse-from-following', testInfo)
  const shows = (await (await page.request.get('/api/app/trending?kind=show&scope=corpus&limit=5')).json()).items
  const feed: string = shows[0].entity_id
  expect((await page.request.post('/api/app/library', { data: { feed_id: feed } })).ok()).toBeTruthy()
  const theirs = (await (await page.request.get(`/api/app/podcasts/${encodeURIComponent(feed)}/episodes?page_size=100`)).json()).items as Array<{ title: string }>
  expect(theirs.length).toBeGreaterThan(0)

  await page.goto('/browse?tab=episodes&from=following&state=unplayed#catalog')
  await expect(page.getByTestId('catalog-from-following')).toHaveAttribute('aria-checked', 'true')
  const cards = page.getByTestId('episode-card')
  await expect(cards.first()).toBeVisible()
  await expect(cards).toHaveCount(Math.min(theirs.length, 20))
  for (const t of theirs.slice(0, 3)) await expect(page.getByText(t.title, { exact: true }).first()).toBeVisible()

  // "All" puts the rest of the catalogue back.
  await page.getByTestId('catalog-from-all').click()
  await expect.poll(async () => cards.count()).toBeGreaterThan(Math.min(theirs.length, 20))
})

test('In progress lists the episode you started and stopped, against the real server', async ({ page }, testInfo) => {
  await signInIsolated(page, 'browse-in-progress', testInfo)
  const slug = await listenToOne(page) // two saves, 10 s then 130 s: started, not finished
  await page.goto('/browse?tab=episodes&state=inprogress#catalog')
  const cards = page.getByTestId('episode-card')
  await expect(cards).toHaveCount(1)
  await expect(cards.first().locator(`a[href*="/episode/${slug}"]`).first()).toBeVisible()
})
