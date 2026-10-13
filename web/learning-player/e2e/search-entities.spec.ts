import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Search lists topics and people ONCE, at the top — never under an episode (operator 2026-10-11).
 *
 * A topic connects episodes. Shown as a "TOPIC · systems thinking integration" row under the
 * episode it was indexed from, it read as a fact about that episode. Real API over the committed
 * corpus, no mocks.
 */
test('topics the query reaches are a list at the top, and no episode carries a Topic row', async ({ page }, testInfo) => {
  await signInIsolated(page, 'search-entities', testInfo)
  await page.goto('/search?q=systems%20thinking&scope=all')

  const topics = page.getByTestId('search-topic-result')
  await expect(topics.first()).toBeVisible({ timeout: 30_000 })
  const labels = (await topics.allInnerTexts()).map((t) => t.trim())
  expect(labels).toContain('systems thinking')
  expect(new Set(labels).size).toBe(labels.length) // one row per topic, however many episodes

  // Above the episodes.
  const top = (await page.getByTestId('search-section-topics').boundingBox())!
  const firstEpisode = page.getByTestId('episode-group').first()
  await expect(firstEpisode).toBeVisible()
  expect(top.y).toBeLessThan((await firstEpisode.boundingBox())!.y)

  // No episode group lists a topic as one of its matches any more.
  for (const group of await page.getByTestId('episode-group').all()) {
    await expect(group.getByText('Topic', { exact: true })).toHaveCount(0)
  }

  // A pill opens the topic.
  await topics.filter({ hasText: /^systems thinking$/ }).click()
  await expect(page.getByRole('dialog').filter({ hasText: /systems thinking/i }).first()).toBeVisible()
})

test('a person the query names is listed under People at the top', async ({ page }, testInfo) => {
  await signInIsolated(page, 'search-entities-person', testInfo)
  await page.goto('/search?q=Elena%20Fischer&scope=all')
  const people = page.getByTestId('search-person-result')
  await expect(people.first()).toBeVisible({ timeout: 30_000 })
  await expect(people.first()).toHaveText('Dr. Elena Fischer')
  for (const group of await page.getByTestId('episode-group').all()) {
    await expect(group.getByText('Dr. Elena Fischer', { exact: true })).toHaveCount(0)
  }
})
