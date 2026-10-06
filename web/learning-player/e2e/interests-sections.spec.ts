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

test('+ Add opens one search at a time; a closed section shows only its pills', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'interests-add', testInfo)
  await openInterests(page)
  await expect(page.locator('input[type="search"]')).toHaveCount(0)
  await page.getByTestId('interest-add-topic').click()
  await expect(page.getByTestId('interest-search-topic')).toBeFocused()
  await page.getByTestId('interest-add-person').click()
  await expect(page.locator('input[type="search"]')).toHaveCount(1)
  await expect(page.getByTestId('interest-search-person')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.locator('input[type="search"]')).toHaveCount(0)
})

test('a suggestion follows on tap and survives a reload', async ({ page }, testInfo) => {
  await signInIsolated(page, 'interests-suggest', testInfo)
  await openInterests(page)
  const topics = page.getByTestId('interests-section-topic')
  await topics.getByTestId('interest-add-topic').click()
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
  await people.getByTestId('interest-add-person').click()
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

  await people.getByTestId('interest-add-done-person').click()
  const chip = people.getByTestId('interest-following-person').filter({ hasText: target! })
  await expect(chip).toBeVisible()
  await chip.getByTestId('interest-remove').click()
  await expect(chip).toHaveCount(0)
  await expect(people.getByTestId('interest-none')).toBeVisible()
})

test('an old ?tab=topics link lands on Interests; each heading carries its one-line hint', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'interests-alias', testInfo)
  await page.goto('/profile?tab=topics')
  await expect(page.getByRole('tab', { name: 'Interests' })).toHaveAttribute('aria-selected', 'true')
  const hint = page.getByTestId('interest-hint-storyline')
  await expect(hint).toHaveText('Topics that come up together')
  // On the heading's row, not under it.
  const [h, t] = await page.evaluate(() => {
    const s = document.querySelector('[data-testid="interests-section-storyline"]')!
    return [s.querySelector('h2, h3')!.getBoundingClientRect(), s.querySelector('[data-testid="interest-hint-storyline"]')!.getBoundingClientRect()].map(
      (r) => r.top + r.height / 2,
    )
  })
  expect(Math.abs(h - t)).toBeLessThanOrEqual(6)
})

test('a followed interest opens its card; a search with no hits says so', async ({ page }, testInfo) => {
  await signInIsolated(page, 'interests-open', testInfo)
  const res = await page.request.put('/api/app/interests', { data: { items: ['topic:risk-management'] } })
  expect(res.ok()).toBeTruthy()
  await page.goto('/profile?tab=interests')
  const section = page.getByTestId('interests-section-topic')
  await section.getByTestId('interest-open').first().click()
  await expect(page.getByRole('dialog').last()).toContainText('risk management')
  await page.keyboard.press('Escape')
  await section.getByTestId('interest-add-topic').click()
  await section.getByTestId('interest-search-topic').fill('zzqq-nothing-matches')
  await expect(section.getByTestId('interest-no-match')).toHaveText('Nothing matches “zzqq-nothing-matches”.')
})
