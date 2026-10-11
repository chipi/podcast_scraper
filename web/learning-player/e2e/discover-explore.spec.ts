import { expect, test } from '@playwright/test'
import { signInIsolated, showEveryonesTrends } from './helpers'

/**
 * Discover is "what to hear next"; Explore is for looking around (ADR-163 §1, operator 2026-10-11).
 * Real API over the committed corpus.
 *
 * - Discover leads with the quote cards: each says WHY it is here (`RecommendationWhy`) in the
 *   transcript's own words, and "Play from" opens the player at the quote.
 * - Trends and the Episodes · Shows catalog are on Explore, one tap from Discover's header; old
 *   `/browse?trends=` / `?tab=` / `#catalog` links land there.
 */

test('Discover leads with quote cards that say why, and play from the quote', async ({ page }, testInfo) => {
  await signInIsolated(page, 'discover-quotes', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  const cards = page.getByTestId('quote-card')
  await expect(cards.first()).toBeVisible()
  const first = cards.first()
  await expect(first.getByTestId('why-reason')).not.toBeEmpty()
  await expect(first.getByTestId('why-quote')).toContainText('“')
  // Episode-first (ADR-163 §2): the quote cards sit above trending shows.
  const quotesTop = (await page.getByTestId('quote-cards').boundingBox())!.y
  const showsTop = (await page.getByTestId('trending-shows-rail').boundingBox())!.y
  expect(quotesTop).toBeLessThan(showsTop)

  const play = first.getByTestId('why-play-from')
  await expect(play).toHaveAttribute('href', /\/episode\/[^?]+\?t=\d+&play=1$/)
  await play.click()
  await expect(page).toHaveURL(/\/episode\/[^?]+\?t=\d+/)
})

test('Explore holds Trends and the catalog, one tap from Discover', async ({ page }, testInfo) => {
  await signInIsolated(page, 'explore-home', testInfo)
  await page.goto('/browse')
  await expect(page.getByTestId('browse-tab-episodes')).toHaveCount(0)
  await page.getByTestId('discover-explore-link').click()
  await expect(page).toHaveURL(/\/explore$/)
  await expect(page.getByTestId('explore-view')).toBeVisible()
  await expect(page.getByTestId('discovery-explorer')).toBeVisible()
  await expect(page.getByTestId('browse-tab-episodes')).toHaveAttribute('aria-selected', 'true')
  await page.getByTestId('browse-tab-shows').click()
  await expect(page.getByTestId('browse-panel-shows')).toBeVisible()
})

test('old Discover links into Trends and the catalog land on Explore', async ({ page }, testInfo) => {
  await signInIsolated(page, 'explore-redirects', testInfo)
  await page.goto('/browse?trends=person')
  await expect(page).toHaveURL(/\/explore\?trends=person/)
  await expect(page.getByTestId('discovery-tab-person')).toHaveAttribute('aria-selected', 'true')
  await page.goto('/browse?tab=shows#catalog')
  await expect(page).toHaveURL(/\/explore\?tab=shows#catalog/)
  await expect(page.getByTestId('browse-tab-shows')).toHaveAttribute('aria-selected', 'true')
})

test("trending shows' See all opens Explore's Shows, sorted by trending", async ({ page }, testInfo) => {
  await signInIsolated(page, 'explore-see-all', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  await page.getByTestId('trending-shows-seeall').click()
  await expect(page).toHaveURL(/\/explore\?tab=shows&sort=trending&view=list#catalog/)
  await expect(page.getByTestId('browse-tab-shows')).toHaveAttribute('aria-selected', 'true')
})

test('a highlight seeds "More from what you captured": elsewhere, saying which highlight', async ({
  page,
}, testInfo) => {
  // Discover arc Step 2.2. The highlight is made through the API a listener's tap uses; the rail
  // must then recommend an episode on ANOTHER show, citing the listener's own words.
  await signInIsolated(page, 'discover-captured', testInfo)
  const slug = 'p07-769d04686d' // a fixture episode whose insights are ABOUT topics shared elsewhere
  const insights = await (await page.request.get(`/api/app/episodes/${slug}/insights`)).json()
  const insight = insights.insights[0]
  const made = await page.request.post('/api/app/highlights', {
    data: {
      episode_slug: slug,
      kind: 'insight',
      source_insight_id: insight.id,
      quote_text: 'kept for later',
      start_ms: insight.quotes?.[0]?.start_ms ?? 0,
    },
  })
  expect(made.status()).toBe(201)

  await page.goto('/browse') // Mine is the default for a signed-in listener
  const rail = page.getByTestId('captured-cards')
  await expect(rail).toBeVisible()
  const first = rail.getByTestId('captured-card').first()
  await expect(first.getByTestId('why-reason')).toHaveText('Because you highlighted “kept for later”')
  await expect(first.getByTestId('why-quote')).toContainText('“')
  await expect(first.getByTestId('captured-card-open')).not.toHaveAttribute('href', /\/episode\/p07-/)
})

test('following a storyline brings "Continue the thread": its episodes, saying which storyline', async ({
  page,
}, testInfo) => {
  // Discover arc Step 2.3. The follow is made through the API the storyline page's Follow uses.
  await signInIsolated(page, 'discover-thread', testInfo)
  const followed = await page.request.post(`/api/app/interests/${encodeURIComponent('thc:managing-risk')}`)
  expect(followed.ok()).toBe(true)

  await page.goto('/browse')
  const rail = page.getByTestId('thread-cards')
  await expect(rail).toBeVisible()
  const first = rail.getByTestId('thread-card').first()
  await expect(first.getByTestId('why-reason')).toHaveText('New in Managing risk across domains')
  await expect(first.getByTestId('why-quote')).toContainText('“')
  await expect(first.getByTestId('thread-card-open')).toHaveAttribute('href', /\/episode\/p\d\d-/)
})

test('a guest heard twice brings "People you keep hearing": their episodes on another show', async ({
  page,
}, testInfo) => {
  // Discover arc Step 3. Dr. Elena Fischer guests twice on The Long View (p07) and also on
  // Cross-Show (p09); highlights in her two p07 episodes make her someone the listener keeps hearing.
  await signInIsolated(page, 'discover-people', testInfo)
  for (const slug of ['p07-a5eb82076a', 'p07-769d04686d']) {
    const made = await page.request.post('/api/app/highlights', {
      data: { episode_slug: slug, kind: 'moment', quote_text: 'kept', start_ms: 1000 },
    })
    expect(made.status()).toBe(201)
  }

  await page.goto('/browse')
  const rail = page.getByTestId('people-cards')
  await expect(rail).toBeVisible()
  const first = rail.getByTestId('people-card').first()
  await expect(first.getByTestId('why-reason')).toHaveText('Dr. Elena Fischer, whom you keep hearing')
  await expect(first.getByTestId('people-card-open')).toHaveAttribute('href', /\/episode\/p09-/)
})
