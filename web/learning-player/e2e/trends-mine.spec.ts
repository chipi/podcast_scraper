import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Trends "Mine" is ON by default and strictly the listener's own world (operator 2026-10-07).
 *
 * A fresh account has heard, saved and followed nothing, so Mine has nothing to rank: it says so
 * and offers everyone's trends. Following a topic puts that topic — and only the listener's world —
 * in Mine. REAL API over the committed corpus, NO mocks.
 */
test('a fresh listener sees Mine empty, explained, with a way to everyone', async ({ page }, testInfo) => {
  await signInIsolated(page, 'trends-mine-fresh', testInfo)
  await page.goto('/browse')
  const toggle = page.getByTestId('home-trending-scope')
  await expect(toggle).toHaveAttribute('aria-pressed', 'true') // Mine by default
  await expect(page.getByTestId('discovery-mine-empty')).toBeVisible()
  await expect(page.getByTestId('discovery-row')).toHaveCount(0)

  await page.getByTestId('discovery-show-everyone').click()
  await expect(toggle).toHaveAttribute('aria-pressed', 'false')
  await expect(page.getByTestId('discovery-row').first()).toBeVisible()

  // The choice sticks: a reload keeps everyone's trends.
  await page.reload()
  await expect(page.getByTestId('home-trending-scope')).toHaveAttribute('aria-pressed', 'false')
})

test("following a topic puts it in Mine, and Mine holds only the listener's world", async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'trends-mine-follow', testInfo)
  const everyone = await page.request.get('/api/app/trending?kind=topic&scope=corpus&limit=50')
  const all = ((await everyone.json()).items as Array<{ entity_id: string }>).map((r) => r.entity_id)
  expect(all.length, 'the corpus must trend more than one topic for this to mean anything').toBeGreaterThan(1)
  const followed = all[1]
  expect((await page.request.post(`/api/app/interests/${encodeURIComponent(followed)}`)).ok()).toBeTruthy()

  const mine = await page.request.get('/api/app/trending?kind=topic&scope=mine&limit=50')
  const ids = ((await mine.json()).items as Array<{ entity_id: string }>).map((r) => r.entity_id)
  expect(ids).toContain(followed)
  expect(ids.length, 'Mine must be a subset, not everyone again').toBeLessThan(all.length)

  await page.goto('/browse')
  await expect(page.getByTestId('home-trending-scope')).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByTestId('discovery-row').first()).toBeVisible()
  await expect(page.getByTestId('discovery-mine-empty')).toHaveCount(0)
})
