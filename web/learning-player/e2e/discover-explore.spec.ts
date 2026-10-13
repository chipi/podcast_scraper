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
  // Explore is its own nav place (phone tab, desktop masthead icon), not a link in Discover's header.
  const tab = page.getByTestId('bottom-nav-explore')
  await ((await tab.isVisible()) ? tab : page.getByTestId('masthead-explore')).click()
  await expect(page).toHaveURL(/\/explore$/)
  await expect(page.getByTestId('explore-view')).toBeVisible()
  await expect(page.getByTestId('discovery-explorer')).toBeVisible()
  await expect(page.getByTestId('browse-tab-episodes')).toHaveAttribute('aria-selected', 'true')
  // Browse: the FULL lists with their filter bars, as in the last release (operator 2026-10-11).
  const episodes = page.getByTestId('browse-panel-episodes')
  await expect(episodes.getByPlaceholder('Filter episodes…')).toBeVisible()
  await expect(episodes.getByTestId('catalog-from')).toBeVisible()
  await page.getByTestId('browse-tab-shows').click()
  const shows = page.getByTestId('browse-panel-shows')
  await expect(shows.getByTestId('show-browse-search')).toBeVisible()
  await expect(shows.getByTestId('show-browse-sort')).toBeVisible()
  await expect(shows.getByTestId('show-view')).toBeVisible()
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

test("trending shows' See all opens the full show list, sorted by trending", async ({ page }, testInfo) => {
  await signInIsolated(page, 'explore-see-all', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  await page.getByTestId('trending-shows-seeall').click()
  await expect(page).toHaveURL(/\/shows\?sort=trending&view=list/)
  await expect(page.getByTestId('show-browse-sort')).toBeVisible()
  // An old link carrying those filters goes to the same full list.
  await page.goto('/explore?tab=shows&sort=trending&view=list#catalog')
  await expect(page).toHaveURL(/\/shows\?tab=shows&sort=trending&view=list/)
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

test('a short listen on a topic: the tap that opens it starts clip 1; next, jump, keep listening', async ({
  page,
}, testInfo) => {
  // ADR-163 §8 amendment (operator 2026-10-11: goal-shaped listening = a cross-episode reel; no start
  // page, one bar, the clips to come as a list; "real length everywhere").
  await signInIsolated(page, 'discover-topic-reel', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  const chip = page.getByTestId('reel-topic-chip').filter({ hasText: 'systems thinking' })
  // The pill names the subject and its kind (operator 2026-10-11, kind word 2026-10-12); its
  // length is in the accessible name.
  await expect(chip).toHaveText('▶ Topic systems thinking')
  // 1 minute since 2026-10-12: its clips stop before an ad break and where their speaker does.
  await expect(chip).toHaveAttribute('aria-label', 'Play 1 minute on systems thinking')
  await chip.click()
  await expect(page).toHaveURL(/\/listen\/topic\/topic(:|%3A)systems-thinking\?minutes=15/)
  await expect(page.getByTestId('topic-reel-title')).toHaveText('systems thinking')
  // The small fixture: the page says what the reel runs, not the fifteen minutes asked for.
  // 1:18 since 2026-10-12: clips stop before an ad break and where their speaker does, and a
  // tenth clip (a second from p07-3f15) fits. 1:24 and eleven since 2026-10-13: the corpus rebuilt on
  // main's claim ranking (test_app_goal_reel pins the same reel).
  await expect(page.getByTestId('topic-reel-length')).toHaveText('1:24 · 11 clips · 9 episodes')

  // Started by the chip's tap: clip 1, playing, with the ten to come listed.
  const current = page.getByTestId('topic-reel-current')
  await expect(current).toContainText('Clip 1 of 11')
  await expect(page.getByTestId('topic-reel-pause')).toBeVisible()
  // The transport sits centred in its card (operator 2026-10-11): equal room left and right.
  const row = (await page.getByTestId('topic-reel-transport').boundingBox())!
  const prevBox = (await page.getByTestId('topic-reel-prev').boundingBox())!
  const nextBox = (await page.getByTestId('topic-reel-next').boundingBox())!
  const leftRoom = prevBox.x - row.x
  const rightRoom = row.x + row.width - (nextBox.x + nextBox.width)
  expect(Math.abs(leftRoom - rightRoom)).toBeLessThan(2)
  await expect(page.getByTestId('topic-reel-up-next')).toHaveCount(10)
  // The clip's episode loads asynchronously (resolve, then load): wait for it before comparing.
  const audioSrc = () => page.evaluate(() => document.querySelector('audio')?.src ?? '')
  await expect.poll(audioSrc).toContain('/audio/')
  const first = await audioSrc()
  await page.getByTestId('topic-reel-next').click()
  await expect(current).toContainText('Clip 2 of 11')
  await expect(page.getByTestId('topic-reel-quote')).toContainText('Sustainability is a systems problem')
  await expect.poll(audioSrc).not.toBe(first) // the next clip's episode was loaded
  await expect(page.getByTestId('topic-reel-up-next')).toHaveCount(9)

  // Any clip to come is a jump.
  await page.getByTestId('topic-reel-up-next').nth(2).click() // clip 5
  await expect(current).toContainText('Clip 5 of 11')
  await expect(page.getByTestId('topic-reel-up-next')).toHaveCount(6)

  // Left unfinished, Home's "In progress" resumes it at clip 5 (the place is kept on the server).
  await expect
    .poll(async () => (await (await page.request.get('/api/app/discover/in-progress')).json()).reel?.index)
    .toBe(4)
  await page.goto('/')
  const resume = page.getByTestId('in-progress-reel')
  await expect(resume).toContainText('Short listen · clip 5 of 11')
  await resume.getByTestId('in-progress-reel-resume').click()
  await expect(page).toHaveURL(/\/listen\/topic\/topic(:|%3A)systems-thinking\?minutes=15&at=5/)
  await expect(current).toContainText('Clip 5 of 11')

  await page.getByTestId('topic-reel-keep').click()
  await expect(page).toHaveURL(/\/episode\/p07-769d04686d/)
})

test('a short listen from a storyline and a theme: clips about any member topic; chips wear their kind', async ({
  page,
}, testInfo) => {
  // Operator 2026-10-11: reels for themes and storylines, a door on their pages, chips coloured
  // like topics, themes and storylines on episode notes.
  await signInIsolated(page, 'discover-storyline-reel', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  // Whichever storyline and theme are offered: which rise depends on what earlier specs in the run
  // listened to (everyone's momentum is shared state). Each wears its kind's colours.
  await expect(page.locator('[data-testid="reel-topic-chip"][data-kind="storyline"]').first()).toHaveClass(/text-storyline/)
  await expect(page.locator('[data-testid="reel-topic-chip"][data-kind="theme"]').first()).toHaveClass(/text-theme/)

  await page.goto(`/storyline/${encodeURIComponent('thc:managing-risk')}`)
  // Its other door: seven days of it (operator 2026-10-11).
  await expect(page.getByTestId('learn-door')).toHaveText('Learn this in 7 days')
  await page.getByTestId('reel-door').click()
  await expect(page).toHaveURL(/\/listen\/topic\/thc(:|%3A)managing-risk\?minutes=15/)
  await expect(page.getByTestId('topic-reel-title')).toHaveText('Managing risk across domains')
  // 4:48 · 39 clips · 27 episodes since 2026-10-12: clips stop before an ad break and where their
  // speaker does, and the translated shows' guests' risk-management lines are insights now.
  // 4:42 · 38 clips · 26 episodes since 2026-10-13: the corpus rebuilt on main's D-44 English
  // analysis (test_app_goal_reel pins the same reel).
  await expect(page.getByTestId('topic-reel-length')).toHaveText('4:42 · 38 clips · 26 episodes')

  await page.goto(`/theme/${encodeURIComponent('tc:reliability')}`)
  await page.getByTestId('reel-door').click()
  await expect(page.getByTestId('topic-reel-title')).toHaveText('reliability')
})

test('Learn X this week: start on Discover, do today, go ahead, find it on Home, stop', async ({ page }, testInfo) => {
  // Operator 2026-10-11: a seven-day path, one day a day or faster, up to three at once; Home's
  // "In progress" is the door back into it.
  await signInIsolated(page, 'discover-learn', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  const section = page.getByTestId('learn-this-week')
  // Whichever storyline is offered: which one rises depends on what earlier specs in the run
  // listened to (everyone's momentum is shared state). The path's contents are pinned server-side
  // (test_app_learning_path.py).
  const subject = section.locator('[data-testid="learn-subject"][data-kind="storyline"]').first()
  // The pill is the subject's name alone: every path is seven days, so it does not repeat that.
  await expect(subject).not.toContainText('days')
  // It starts with its kind word (a mixed row, 2026-10-12); the path's title is the name alone.
  await expect(subject.getByTestId('kind-word')).toHaveText(/^Storyline\s*$/)
  const label = ((await subject.textContent()) ?? '').replace(/^\s*Storyline\s*/, '').trim()
  await subject.click()

  // Starting opens its week, on day 1.
  await expect(page).toHaveURL(/\/learn\?subject=thc/)
  await expect(page.getByTestId('learn-view-title')).toHaveText(label)
  const today = page.locator('[data-testid="learn-step"][data-open="true"]')
  await expect(today).toHaveAttribute('data-day', '1')
  await expect(today.getByTestId('learn-step-play')).toHaveAttribute('href', /\/episode\/[^/?]+\?t=\d+&play=1&from=learn$/)
  await today.getByTestId('learn-step-mark-done').click()
  await expect(today.getByTestId('learn-step-done-mark')).toBeVisible()
  // A day a day is the rhythm, not a wall: once today's is done, the next can open now.
  await today.getByTestId('learn-step-next').click()
  await expect(page.locator('[data-testid="learn-step"][data-open="true"]')).toHaveAttribute('data-day', '2')
  await expect(page.getByTestId('learn-step-locked')).toHaveCount(5) // never early unless opened

  // Home's "In progress" is the door back: day 2, playable from its clip.
  await page.goto('/')
  const card = page.getByTestId('in-progress-learn').filter({ hasText: label })
  await expect(card).toContainText('Learn · day 2 of 7')
  await expect(card.getByTestId('in-progress-learn-play')).toHaveAttribute('href', /\/episode\/[^/?]+\?t=\d+&play=1$/)
  await card.getByTestId('in-progress-learn-open').click()
  await expect(page).toHaveURL(/\/learn\?subject=thc/)

  await page.getByTestId('learn-stop').click()
  await page.getByTestId('learn-stop-yes').click()
  await expect(page).toHaveURL(/\/browse/)
  await expect(page.getByTestId('learn-subject').first()).toBeVisible()
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
  // The whole card opens the episode, paused; the show is named, not linked (operator 2026-10-11).
  await expect(rail.locator('a[href^="/podcast/"]')).toHaveCount(0)
  const card = rail.getByTestId('thread-card').first()
  const href = await card.getByTestId('thread-card-open').getAttribute('href')
  await card.scrollIntoViewIfNeeded()
  const box = await card.boundingBox()
  await page.mouse.click(box!.x + box!.width - 12, box!.y + 12) // a corner, not the title
  await expect(page).toHaveURL(new RegExp(`${href}$`))
  await expect.poll(() => page.evaluate(() => document.querySelector('audio')?.paused ?? true)).toBe(true)
})
