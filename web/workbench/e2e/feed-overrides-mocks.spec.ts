import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  requireSerialCorpusAccess,
  SHELL_HEADING_RE,
  signInAsAdmin,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * #694 — per-feed override drill-in. Configuring a feed sets structured override fields (e.g.
 * `max_episodes`) on that entry only; other feeds round-trip unchanged, and the result persists
 * via `PUT /api/feeds`.
 *
 * #1619 — migrated to the live API, including the write.
 *
 * Previously blocked because the assertion IS the write, and writing dirtied the tracked fixture
 * corpus. `e2e/run-local-stack.sh` now serves from a disposable copy, so the spec seeds its own
 * two-feed starting state through the real API and reads the result back off the server.
 *
 * That matters here more than most: the old version captured the request body in a closure and
 * asserted against it, so "persists via PUT" only ever meant "the UI sent this JSON". The server
 * could have rejected, reordered, or dropped the sibling entry and the test would still pass.
 */

const FEED_A = 'https://a.example/rss'
const FEED_B = 'https://b.example/rss'

/** Write the starting feed list through the real API (safe: disposable corpus copy). */
async function seedFeeds(page: Page, corpusPath: string): Promise<void> {
  const resp = await page.request.put(`/api/feeds?path=${encodeURIComponent(corpusPath)}`, {
    data: { feeds: [FEED_A, FEED_B] },
  })
  if (!resp.ok()) throw new Error(`seedFeeds: PUT /api/feeds returned ${resp.status()}`)
}

/** Read the feed list back off the server. */
async function readFeeds(page: Page, corpusPath: string): Promise<unknown[]> {
  const resp = await page.request.get(`/api/feeds?path=${encodeURIComponent(corpusPath)}`)
  const body = (await resp.json()) as { feeds: unknown[] }
  return body.feeds
}

/** SERIAL: this test rewrites the served corpus's `feeds.spec.yaml`. See e2e/README.md. */
test.describe.configure({ mode: 'serial' })

test.describe('Per-feed override editor (#694)', () => {
  test('Configure sets max_episodes on one feed and persists via PUT', async ({
    page,
  }, testInfo) => {
    requireSerialCorpusAccess(testInfo)
    await signInAsAdmin(page)

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    const corpusPath = await liveCorpusRoot(page)
    await seedFeeds(page, corpusPath)

    // Enter commits the path so the operator fetches fire.
    await statusBarCorpusPathInput(page).fill(corpusPath)
    await statusBarCorpusPathInput(page).press('Enter')

    await page.getByTestId('status-bar-sources-trigger').click()
    await expect(page.getByTestId('status-bar-sources-dialog')).toBeVisible()
    await expect(page.getByTestId('sources-dialog-feeds-row-0')).toContainText(FEED_A)

    // Drill into feed 0 and set a per-feed override.
    await page.getByTestId('sources-dialog-feeds-row-configure-0').click()
    await expect(page.getByTestId('feed-override-editor')).toBeVisible()
    await expect(page.getByTestId('feed-override-url')).toHaveText(FEED_A)
    await page.getByTestId('feed-override-max-episodes').fill('2')
    await page.getByTestId('feed-override-save').click()

    // Back to the list.
    await expect(page.getByTestId('feed-override-editor')).toBeHidden()

    /* Read the persisted spec back from the server: the override must be on feed 0 only, and the
     * sibling must survive the rewrite untouched. */
    await expect
      .poll(async () => readFeeds(page, corpusPath), { timeout: 15_000 })
      .toEqual([{ url: FEED_A, max_episodes: 2 }, FEED_B])
  })

  /** Seed [A, B], open Configuration and drill into feed 0. Returns the corpus path. */
  async function openFeedZeroEditor(page: Page): Promise<string> {
    await signInAsAdmin(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    const corpusPath = await liveCorpusRoot(page)
    await seedFeeds(page, corpusPath)
    await statusBarCorpusPathInput(page).fill(corpusPath)
    await statusBarCorpusPathInput(page).press('Enter')
    await page.getByTestId('status-bar-sources-trigger').click()
    await expect(page.getByTestId('sources-dialog-feeds-row-0')).toContainText(FEED_A)
    await page.getByTestId('sources-dialog-feeds-row-configure-0').click()
    await expect(page.getByTestId('feed-override-editor')).toBeVisible()
    return corpusPath
  }

  test('max-episodes hint, Advanced and Other fields round-trip; clearing them reverts to a bare URL', async ({
    page,
  }, testInfo) => {
    requireSerialCorpusAccess(testInfo)
    const corpusPath = await openFeedZeroEditor(page)
    const editor = page.getByTestId('feed-override-editor')

    await expect(page.getByTestId('feed-override-maxep-hint')).toHaveCount(0)
    await page.getByTestId('feed-override-max-episodes').fill('3')
    await expect(page.getByTestId('feed-override-maxep-hint')).toContainText(
      'viewer_operator default',
    )

    await editor.getByRole('button', { name: /^Advanced/ }).click()
    await page.getByTestId('feed-override-adv-delay_ms').fill('500')
    await expect(editor.getByRole('button', { name: 'Advanced (1 set)' })).toBeVisible()
    // `show_centric` is a key the server's RssFeedEntry accepts but the editor does not model —
    // exactly what "Other fields" exists to carry.
    await page.getByTestId('feed-override-extras').fill('{"show_centric": true}')
    await page.getByTestId('feed-override-save').click()
    await expect(editor).toBeHidden()
    await expect
      .poll(async () => readFeeds(page, corpusPath), { timeout: 15_000 })
      .toEqual([{ url: FEED_A, max_episodes: 3, delay_ms: 500, show_centric: true }, FEED_B])

    // Re-open: the editor reads the saved entry back into its fields.
    await page.getByTestId('sources-dialog-feeds-row-configure-0').click()
    await expect(page.getByTestId('feed-override-max-episodes')).toHaveValue('3')
    await editor.getByRole('button', { name: /^Advanced/ }).click()
    await expect(page.getByTestId('feed-override-adv-delay_ms')).toHaveValue('500')
    await expect(page.getByTestId('feed-override-extras')).toHaveValue(/"show_centric": true/)

    // Empty = inherit: clear everything and the entry collapses back to the bare URL string.
    await page.getByTestId('feed-override-max-episodes').fill('')
    await page.getByTestId('feed-override-adv-delay_ms').fill('')
    await page.getByTestId('feed-override-extras').fill('')
    await page.getByTestId('feed-override-save').click()
    await expect(editor).toBeHidden()
    await expect
      .poll(async () => readFeeds(page, corpusPath), { timeout: 15_000 })
      .toEqual([FEED_A, FEED_B])
  })

  test('invalid input is refused in the editor and nothing is written', async ({
    page,
  }, testInfo) => {
    requireSerialCorpusAccess(testInfo)
    const corpusPath = await openFeedZeroEditor(page)
    let puts = 0
    page.on('request', (r) => {
      if (r.method() === 'PUT' && new URL(r.url()).pathname === '/api/feeds') puts += 1
    })

    await page.getByTestId('feed-override-max-episodes').fill('0')
    await page.getByTestId('feed-override-save').click()
    await expect(page.getByTestId('feed-override-error')).toHaveText(
      'Max episodes must be an integer ≥ 1.',
    )

    await page.getByTestId('feed-override-max-episodes').fill('')
    // "Other fields" sits in the collapsed Advanced section.
    await page.getByTestId('feed-override-editor').getByRole('button', { name: /^Advanced/ }).click()
    await page.getByTestId('feed-override-extras').fill('not json')
    await page.getByTestId('feed-override-save').click()
    await expect(page.getByTestId('feed-override-error')).toHaveText(
      'Other fields must be valid JSON.',
    )
    await expect(page.getByTestId('feed-override-editor')).toBeVisible()
    expect(puts).toBe(0)
    expect(await readFeeds(page, corpusPath)).toEqual([FEED_A, FEED_B])
  })

  test('a save the server rejects is shown, the editor stays open, nothing is written', async ({
    page,
  }, testInfo) => {
    requireSerialCorpusAccess(testInfo)
    const corpusPath = await openFeedZeroEditor(page)
    // Valid JSON the client accepts, but a key the server's RssFeedEntry forbids (extra="forbid")
    // — a real rejection, not an injected one.
    await page.getByTestId('feed-override-editor').getByRole('button', { name: /^Advanced/ }).click()
    await page.getByTestId('feed-override-extras').fill('{"not_a_feed_key": 1}')
    const put = page.waitForResponse(
      (r) => r.request().method() === 'PUT' && new URL(r.url()).pathname === '/api/feeds',
    )
    await page.getByTestId('feed-override-save').click()
    expect((await put).status()).toBeGreaterThanOrEqual(400)

    await expect(page.getByTestId('status-bar-sources-dialog')).toContainText('not_a_feed_key')
    await expect(page.getByTestId('feed-override-editor')).toBeVisible()
    expect(await readFeeds(page, corpusPath)).toEqual([FEED_A, FEED_B])
  })
})
