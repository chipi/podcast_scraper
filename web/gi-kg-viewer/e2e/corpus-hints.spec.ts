import { expect, test } from '@playwright/test'

test.describe('Corpus path hints (mocked API)', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
        }),
      })
    })

    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus/feeds/rss_example/metadata',
          hints: [
            'Unified semantic index is under /mock/corpus. Set corpus root to that directory for search and index stats (multi-feed layout).',
          ],
          artifacts: [],
        }),
      })
    })
  })

  test('List shows corpus path hint when API returns hints', async ({ page }) => {
    await page.goto('/')

    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus/feeds/rss_example/metadata')
    await page.getByTestId('status-bar-list-artifacts').click()

    await expect(page.getByText('Corpus path hint')).toBeVisible()
    await expect(page.getByText(/Unified semantic index/i)).toBeVisible()
  })
})
