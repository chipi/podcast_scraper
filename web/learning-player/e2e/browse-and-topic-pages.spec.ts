import { expect, test } from '@playwright/test'

/**
 * Browse pages + standalone Topic/Person deep-links (#1261-6, #1261-9). Real API +
 * committed corpus. Home surfaces "Browse topics" / "Browse people" chip strip,
 * each opens the index page, tapping a chip lands on the standalone Topic /
 * Person page.
 *
 * These pages replace the mobile-hostile Cmd-K palette that was explicitly
 * ruled out of the listener player.
 */

test('Home surfaces "Browse topics" / "Browse people" and both index pages render', async ({
  page,
}) => {
  await page.goto('/')
  const nav = page.getByTestId('home-browse-nav')
  await expect(nav).toBeVisible()

  const topicsLink = nav.getByRole('link', { name: /Browse topics/ })
  const peopleLink = nav.getByRole('link', { name: /Browse people/ })
  await expect(topicsLink).toBeVisible()
  await expect(peopleLink).toBeVisible()

  await topicsLink.click()
  await expect(page).toHaveURL(/\/browse\/topics$/)
  await expect(page.getByRole('heading', { name: 'Browse topics' })).toBeVisible()

  await page.goto('/')
  await page.getByTestId('home-browse-nav').getByRole('link', { name: /Browse people/ }).click()
  await expect(page).toHaveURL(/\/browse\/people$/)
  await expect(page.getByRole('heading', { name: 'Browse people' })).toBeVisible()
})

test('a standalone /topic/:id page loads the topic card body via EntityCardBody inline mode', async ({
  page,
}) => {
  // Committed corpus (v3): topic:index-investing is one of the shipped topics —
  // seed from /browse/topics to avoid pinning a specific id here.
  await page.goto('/browse/topics')
  const anyTopicLink = page.locator('a[href^="/topic/"]').first()
  await expect(anyTopicLink).toBeVisible()
  const href = await anyTopicLink.getAttribute('href')
  await anyTopicLink.click()
  // toHaveURL compares against the FULL URL — regex-escape the id (contains
  // ':' which is not a regex meta but stays literal via escape anyway) and
  // match the tail rather than the full string.
  const escaped = href!.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  await expect(page).toHaveURL(new RegExp(`${escaped}$`))
  // EntityCardBody renders in variant='inline' — the "Topic" kicker plus the topic label.
  await expect(page.getByTestId('topic-view')).toBeVisible()
  await expect(page.getByText('Topic', { exact: true })).toBeVisible()
})
