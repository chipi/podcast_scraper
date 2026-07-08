import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The recommender responds to the influence levers (PRD-043 personalized discovery, #1098). Over the
 * SAME committed corpus, the /discover feed is recency-ordered with NO influence, and re-ranks toward
 * a followed topic WITH influence — asserting *what* is recommended changes because of the lever.
 *
 * The fixture discriminates at the topic level: `topic:safety-practices` belongs only to the p03
 * dive episodes, while the newest episode (which tops recency) is the p05 finance one — so following
 * that topic visibly flips the top of the feed. Personalization needs APP_PERSONALIZED_RANKING (set
 * in the e2e server env); with no interests the feed stays recency, so every other spec is unaffected.
 *
 * Lever scope: only EXPLICIT interests (followed topic/cluster/person tokens) feed discover ranking.
 * Saved highlights produce derived-interest *suggestions* but don't auto-re-rank the feed (the user
 * must follow a suggestion) — so the lever asserted end-to-end here is an explicit topic follow.
 */
const NEWEST = 'Risk Is a Systems Property' // p09_e04 — newest on the #1148 2024→now schedule, leads recency
const FOLLOW_TOPIC = 'topic:safety-practices' // p03-only → the dive episodes
const FOLLOWED_EPISODES = ['Wreck Diving Fundamentals', 'Marine Biology for Divers', 'Calm Under Pressure', 'Plan the Dive, Manage the Risk'] // all p03 dive eps (#1148: p03 now has 4)

async function discoverTitles(page: import('@playwright/test').Page): Promise<string[]> {
  const res = await page.request.get('/api/app/discover?limit=8')
  expect(res.ok()).toBeTruthy()
  return (await res.json()).items.map((i: { title: string }) => i.title)
}

test('discover re-ranks toward a followed topic (with vs without influence)', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'recommend', testInfo)

  // Reset to NO influence so the baseline is deterministic across retries.
  await page.request.put('/api/app/interests', { data: { items: [] } })

  // WITHOUT influence → recency: the newest episode leads the feed.
  const baseline = await discoverTitles(page)
  expect(baseline[0]).toContain(NEWEST)
  // none of the (older) followed-topic episodes lead by recency
  expect(FOLLOWED_EPISODES.some((t) => baseline[0].includes(t))).toBe(false)

  // Pull the lever: follow a discriminating topic (real API, real ranking).
  const follow = await page.request.post(`/api/app/interests/${encodeURIComponent(FOLLOW_TOPIC)}`)
  expect(follow.ok()).toBeTruthy()

  // WITH influence → the feed re-ranks: a followed-topic episode now leads, and the order changed.
  const influenced = await discoverTitles(page)
  expect(influenced[0]).not.toBe(baseline[0]) // the recommendation actually changed
  expect(FOLLOWED_EPISODES.some((t) => influenced[0].includes(t))).toBe(true)

  // The UI reflects the personalized order: Home's "What's new" hero (fed by /discover) leads with it.
  await page.goto('/')
  await expect(page.getByText(influenced[0]).first()).toBeVisible()
})
