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
  // Cross-Show (p09); hearing her two p07 episodes makes her someone the listener keeps hearing.
  // LISTENED, not highlighted: highlights would also seed "More from what you captured", which sits
  // higher on the page and — one card per episode (ADR-164) — would show her p09 episodes first.
  await signInIsolated(page, 'discover-people', testInfo)
  for (const slug of ['p07-a5eb82076a', 'p07-769d04686d']) {
    // Heard = the saved position reaches 30% of the episode (these run under ten minutes).
    const heard = await page.request.put(`/api/app/playback/${slug}`, {
      data: { position_seconds: 300, finished: true },
    })
    expect(heard.ok()).toBe(true)
  }

  await page.goto('/browse')
  const rail = page.getByTestId('people-cards')
  await expect(rail).toBeVisible()
  const first = rail.getByTestId('people-card').first()
  await expect(first.getByTestId('why-reason')).toHaveText('Dr. Elena Fischer, whom you keep hearing')
  await expect(first.getByTestId('people-card-open')).toHaveAttribute('href', /\/episode\/p09-/)
})

test('topics a listener keeps hearing bring "Hear another view": a speaker not yet heard on one', async ({
  page,
}, testInfo) => {
  // Discover arc Step 3. Three highlighted episodes about risk management and systems thinking;
  // the rail offers someone the listener has not heard on either.
  await signInIsolated(page, 'discover-another-view', testInfo)
  for (const slug of ['p07-a5eb82076a', 'p07-2aceab172c', 'p02-8fdd7b7f79']) {
    const made = await page.request.post('/api/app/highlights', {
      data: { episode_slug: slug, kind: 'moment', quote_text: 'kept', start_ms: 1000 },
    })
    expect(made.status()).toBe(201)
  }

  await page.goto('/browse')
  const rail = page.getByTestId('another-view-cards')
  await expect(rail).toBeVisible()
  const first = rail.getByTestId('another-view-card').first()
  // Which topic leads depends on what the rails above already show (one card per episode), so the
  // promise is asserted, not the order: a speaker the listener has not heard, on one of their topics.
  await expect(first.getByTestId('why-reason')).toHaveText(/^Another view on (risk management|systems thinking)$/)
  for (const heard of ['Dr. Elena Fischer', 'Skanda Amarnath', 'Alex Morgan', 'Priya Nair', 'Ethan']) {
    await expect(first.getByTestId('why-quote')).not.toContainText(heard)
  }
})

test('cold start: Everyone, labelled, until a few topics are picked — then Mine is theirs', async ({
  page,
}, testInfo) => {
  // ADR-163 §6. A fresh listener's Mine has nothing in it: the quote cards are Everyone's and say so,
  // with "Pick topics". Following a topic widens Mine (world_episode_set), so after saving, the
  // note is gone and the cards are the listener's own.
  await signInIsolated(page, 'discover-cold-start', testInfo)
  await page.goto('/browse')
  await expect(page.getByTestId('discover-cold-start')).toBeVisible()
  await expect(page.getByTestId('quote-card').first()).toBeVisible()

  await page.getByTestId('discover-pick-topics').click()
  await page.getByTestId('interest-add-topic').click()
  await page.getByTestId('interest-search-topic').fill('systems thinking')
  await page.getByTestId('interest-add-panel-topic').getByTestId('interest-result').first().click()
  await page.getByTestId('interests-save').click()

  await expect(page.getByTestId('discover-cold-start')).toHaveCount(0)
  await expect(page.getByTestId('quote-card').first()).toBeVisible()
})

test('"Less like this" hides a storyline from Discover; Profile › Interests undoes it', async ({
  page,
}, testInfo) => {
  // ADR-163 §6 (operator 2026-10-11: one tap hides the subject the card's reason names).
  await signInIsolated(page, 'discover-suggest-less', testInfo)
  const followed = await page.request.post(`/api/app/interests/${encodeURIComponent('thc:managing-risk')}`)
  expect(followed.ok()).toBe(true)

  await page.goto('/browse')
  const rail = page.getByTestId('thread-cards')
  await expect(rail.getByTestId('thread-card').first()).toBeVisible()
  await rail.getByTestId('thread-card').first().getByTestId('rail-less').click()
  await expect(rail.getByTestId('rail-less-done')).toContainText('Less about Managing risk across domains')
  await expect(rail.getByTestId('thread-card')).toHaveCount(0)

  await page.goto('/profile?tab=interests')
  const hidden = page.getByTestId('suggested-less-item')
  await expect(hidden).toHaveText(/Managing risk across domains/)
  await hidden.click()
  await expect(page.getByTestId('suggested-less')).toHaveCount(0)

  await page.goto('/browse')
  await expect(page.getByTestId('thread-cards').getByTestId('thread-card').first()).toBeVisible()
})

test('"15 min on" a topic: clips from different episodes, back to back, then keep listening to one', async ({
  page,
}, testInfo) => {
  // ADR-163 §8 amendment (operator 2026-10-11: goal-shaped listening = a cross-episode reel).
  await signInIsolated(page, 'discover-topic-reel', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  const chip = page.getByTestId('reel-topic-chip').filter({ hasText: 'systems thinking' })
  await chip.click()
  await expect(page).toHaveURL(/\/listen\/topic\/topic(:|%3A)systems-thinking\?minutes=15/)
  await expect(page.getByTestId('topic-reel-title')).toHaveText('15 minutes on systems thinking')
  // The small fixture: the page says what the reel runs, not the fifteen minutes asked for.
  await expect(page.getByTestId('topic-reel-length')).toHaveText('11 clips from different episodes · 2:12')

  await page.getByTestId('topic-reel-play').click()
  const current = page.getByTestId('topic-reel-current')
  await expect(current).toContainText('Clip 1 of 11')
  // The clip's episode loads asynchronously (resolve, then load): wait for it before comparing.
  const audioSrc = () => page.evaluate(() => document.querySelector('audio')?.src ?? '')
  await expect.poll(audioSrc).toContain('/audio/')
  const first = await audioSrc()
  await page.getByTestId('topic-reel-next').click()
  await expect(current).toContainText('Clip 2 of 11')
  await expect(page.getByTestId('topic-reel-quote')).toContainText('Systems thinking in macro means watching feedback loops')
  await expect.poll(audioSrc).not.toBe(first) // the next clip's episode was loaded

  await page.getByTestId('topic-reel-keep').click()
  await expect(page).toHaveURL(/\/episode\/p07-3f1584920f/)
})

test('Discover is a taste: two cards a rail, "Show more" pages from the server, one card per episode', async ({
  page,
}, testInfo) => {
  // Operator 2026-10-11: no long scroll; more on request, paged on the server.
  await signInIsolated(page, 'discover-taste', testInfo)
  const followed = await page.request.post(`/api/app/interests/${encodeURIComponent('thc:managing-risk')}`)
  expect(followed.ok()).toBe(true)
  await page.goto('/browse')
  const rail = page.getByTestId('thread-cards')
  await expect(rail.getByTestId('thread-card')).toHaveCount(2)
  const paged = page.waitForRequest((r) => r.url().includes('/api/app/discover/rails/thread?'))
  await rail.getByTestId('show-all-toggle').click()
  const req = await paged
  expect(new URL(req.url()).searchParams.get('offset')).toBe('2')
  await expect.poll(() => rail.getByTestId('thread-card').count()).toBeGreaterThan(2)
  // One card per episode across the page.
  const hrefs = await page.locator('[data-testid$="-card-open"]').evaluateAll((els) =>
    els.map((e) => e.getAttribute('href')),
  )
  expect(new Set(hrefs).size).toBe(hrefs.length)
  // The show is named, so it is a way there.
  await expect(rail.getByTestId('rail-show-link').first()).toHaveAttribute('href', /\/podcast\//)
})
