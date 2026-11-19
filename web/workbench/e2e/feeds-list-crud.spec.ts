import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  requireSerialCorpusAccess,
  SHELL_HEADING_RE,
  signInAsAdmin,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * Configuration › Feeds list rows: **Edit** (inline URL, Save / Cancel, duplicate refused) and
 * **Delete**, each persisted through `PUT /api/feeds` and read BACK off the server — so "persists"
 * means the server kept it, not that the UI sent a body. Same seeding as
 * `feed-overrides-mocks.spec.ts` (disposable corpus copy; serial).
 */
const FEED_A = 'https://a.example/rss'
const FEED_B = 'https://b.example/rss'
const FEED_C = 'https://c.example/rss'

async function seedFeeds(page: Page, corpusPath: string, feeds: string[]): Promise<void> {
  const resp = await page.request.put(`/api/feeds?path=${encodeURIComponent(corpusPath)}`, {
    data: { feeds },
  })
  if (!resp.ok()) throw new Error(`seedFeeds: PUT /api/feeds returned ${resp.status()}`)
}

async function readFeeds(page: Page, corpusPath: string): Promise<unknown[]> {
  const resp = await page.request.get(`/api/feeds?path=${encodeURIComponent(corpusPath)}`)
  return ((await resp.json()) as { feeds: unknown[] }).feeds
}

async function openFeedsList(page: Page, feeds: string[]): Promise<string> {
  await signInAsAdmin(page)
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
  const corpusPath = await liveCorpusRoot(page)
  await seedFeeds(page, corpusPath, feeds)
  await statusBarCorpusPathInput(page).fill(corpusPath)
  await statusBarCorpusPathInput(page).press('Enter')
  await page.getByTestId('status-bar-sources-trigger').click()
  await expect(page.getByTestId('sources-dialog-feeds-row-0')).toContainText(feeds[0]!)
  return corpusPath
}

test.describe.configure({ mode: 'serial' })

test.describe('Configuration › Feeds list: edit and delete rows', () => {
  test('Edit rewrites one URL in place; Cancel leaves it; a duplicate is refused', async ({
    page,
  }, testInfo) => {
    requireSerialCorpusAccess(testInfo)
    const corpusPath = await openFeedsList(page, [FEED_A, FEED_B])

    // Cancel: the row returns unchanged, nothing is written.
    await page.getByTestId('sources-dialog-feeds-row-edit-1').click()
    await page.getByTestId('sources-dialog-feeds-row-edit-input-1').fill('https://never.example/rss')
    await page.getByTestId('sources-dialog-feeds-row-cancel-1').click()
    await expect(page.getByTestId('sources-dialog-feeds-row-1')).toContainText(FEED_B)
    expect(await readFeeds(page, corpusPath)).toEqual([FEED_A, FEED_B])

    // A URL another row already uses is refused, and nothing is written.
    await page.getByTestId('sources-dialog-feeds-row-edit-1').click()
    await page.getByTestId('sources-dialog-feeds-row-edit-input-1').fill(FEED_A)
    await page.getByTestId('sources-dialog-feeds-row-save-1').click()
    await expect(page.getByTestId('status-bar-sources-dialog')).toContainText(
      'Another feed already uses this URL.',
    )
    expect(await readFeeds(page, corpusPath)).toEqual([FEED_A, FEED_B])

    // A real edit: row 1 only, order kept, on the server.
    await page.getByTestId('sources-dialog-feeds-row-edit-input-1').fill(FEED_C)
    await page.getByTestId('sources-dialog-feeds-row-save-1').click()
    await expect(page.getByTestId('sources-dialog-feeds-row-1')).toContainText(FEED_C)
    await expect.poll(() => readFeeds(page, corpusPath), { timeout: 15_000 }).toEqual([FEED_A, FEED_C])
  })

  test('Delete removes exactly that row and the server keeps the rest in order', async ({
    page,
  }, testInfo) => {
    requireSerialCorpusAccess(testInfo)
    const corpusPath = await openFeedsList(page, [FEED_A, FEED_B, FEED_C])

    await page.getByTestId('sources-dialog-feeds-row-delete-1').click()
    await expect(page.getByTestId('sources-dialog-feeds-row-1')).toContainText(FEED_C)
    await expect(page.getByTestId('sources-dialog-feeds-row-2')).toHaveCount(0)
    await expect.poll(() => readFeeds(page, corpusPath), { timeout: 15_000 }).toEqual([FEED_A, FEED_C])
  })
})
