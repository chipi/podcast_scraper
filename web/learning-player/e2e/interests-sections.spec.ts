import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Profile › Interests — one section per kind, each editable in place (beta feedback 2026-10-04).
 *
 * Against the REAL server and the committed corpus: the suggestions come from `/trending`, the
 * search box from `/interests/search`, and every tap is a real follow written through the store. No
 * search index is involved — the interest search matches labels in the knowledge-graph index — so
 * this runs on a machine without the search extras.
 */

async function openInterests(page: Page): Promise<void> {
  await page.goto('/profile')
  await page.getByRole('tab', { name: 'Interests' }).click()
  await expect(page.getByTestId('interests-section-topic')).toBeVisible()
}

test('the tab is Interests, with the note above four sections', async ({ page }, testInfo) => {
  await signInIsolated(page, 'interests-tab', testInfo)
  await openInterests(page)
  await expect(page.getByTestId('interests-help')).toHaveText(
    'These shape what surfaces on your Home when personalization is on.'
  )
  const headings = page.locator('[data-testid^="interests-section-"] h2')
  await expect(headings).toHaveText(['Topics', 'People', 'Themes', 'Storylines'])
})

test('a suggestion follows on tap and survives a reload', async ({ page }, testInfo) => {
  await signInIsolated(page, 'interests-suggest', testInfo)
  await openInterests(page)
  const topics = page.getByTestId('interests-section-topic')
  const first = topics.getByTestId('interest-suggestion').first()
  await expect(first).toBeVisible()
  const name = ((await first.textContent()) ?? '').replace('+', '').trim()
  await first.click()
  await expect(topics.getByTestId('interest-following-topic')).toContainText([name])

  // Written to the server, not just flipped on screen.
  await page.reload()
  await page.getByRole('tab', { name: 'Interests' }).click()
  await expect(
    page.getByTestId('interests-section-topic').getByTestId('interest-following-topic')
  ).toContainText([name])
})

test('search finds a person the suggestions do not show, and × unfollows', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'interests-search', testInfo)
  await openInterests(page)
  const people = page.getByTestId('interests-section-person')
  // Pick a person from the corpus who is NOT among the suggestions, so the search is what finds
  // them. The API answers the same question the box asks.
  const suggested = (await people.getByTestId('interest-suggestion').allTextContents()).map((s) =>
    s.replace('+', '').trim()
  )
  const res = await page.request.get('/api/app/interests/search?kind=person&q=a&limit=50')
  expect(res.ok()).toBe(true)
  const all = ((await res.json()) as { items: { label: string }[] }).items.map((i) => i.label)
  const target = all.find((l) => !suggested.includes(l) && l.length >= 3)
  expect(target, `every person in the corpus is already suggested: ${all}`).toBeTruthy()

  await people.getByTestId('interest-search-person').fill(target!.slice(0, 4))
  const hit = people.getByTestId('interest-result').filter({ hasText: target! })
  await expect(hit).toBeVisible()
  await hit.click()
  await expect(hit).toHaveAttribute('aria-pressed', 'true')

  await people.getByTestId('interest-search-person').fill('')
  const chip = people.getByTestId('interest-following-person').filter({ hasText: target! })
  await expect(chip).toBeVisible()
  await chip.getByTestId('interest-remove').click()
  await expect(chip).toHaveCount(0)
  await expect(people.getByTestId('interest-none')).toBeVisible()
})
