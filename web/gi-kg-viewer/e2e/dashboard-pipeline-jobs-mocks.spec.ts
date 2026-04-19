import { expect, test } from '@playwright/test'
import { setupDashboardApiMocks } from './dashboardApiMocks'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

function matchExactApiPath(path: string): (url: URL) => boolean {
  return (url: URL) => url.pathname.replace(/\/$/, '') === path
}

test.describe('Dashboard — Pipeline jobs card (mocked API)', () => {
  test('Pipeline tab shows jobs card and empty list when jobs_api is true', async ({ page }) => {
    await setupDashboardApiMocks(page)
    await page.route(matchExactApiPath('/api/health'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          jobs_api: true,
        }),
      })
    })
    await page.route(
      (url) => {
        const p = url.pathname.replace(/\/$/, '')
        return p === '/api/jobs' && url.searchParams.has('path')
      },
      async (route) => {
        if (route.request().method() !== 'GET') {
          await route.fulfill({ status: 405, body: 'expected GET' })
          return
        }
        const url = new URL(route.request().url())
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            path: url.searchParams.get('path') ?? '',
            jobs: [],
          }),
        })
      },
    )

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()

    const tablist = page.getByRole('tablist', { name: 'Dashboard tabs' })
    await expect(tablist).toBeVisible({ timeout: 15_000 })
    await tablist.getByRole('tab', { name: 'Pipeline' }).click()

    await expect(page.getByTestId('pipeline-jobs-card')).toBeVisible({ timeout: 15_000 })
    await expect(
      page.getByText('No jobs yet. Run queues a CLI pipeline for this corpus.'),
    ).toBeVisible()
  })
})
