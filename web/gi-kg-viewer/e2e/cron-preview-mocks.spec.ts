import { expect, test, type Page } from '@playwright/test'
import { SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * #709 — live cron preview + validation under the Job Configuration editor.
 * A valid schedule shows a next-run preview; a bad cron is flagged before save.
 */
function matchExactApiPath(path: string): (url: URL) => boolean {
  return (url: URL) => url.pathname.replace(/\/$/, '') === path
}

const OPERATOR_YAML = `scheduled_jobs:
  - name: nightly
    cron: "0 2 * * *"
    enabled: true
  - name: broken
    cron: nope
    enabled: true
`

async function stubCompanionApis(page: Page): Promise<void> {
  await page.route(
    (url) => {
      const p = url.pathname.replace(/\/$/, '')
      return p === '/api/artifacts' || p.startsWith('/api/artifacts/')
    },
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [], hints: [] }),
      })
    },
  )
  await page.route(matchExactApiPath('/api/index/stats'), async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ available: false, stats: null, reindex_recommended: false, rebuild_in_progress: false, rebuild_last_error: null }),
    })
  })
}

test.describe('Cron schedule preview (#709)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin')
  })

  test('previews valid schedules and flags an invalid cron in the editor', async ({ page }) => {
    await page.route(matchExactApiPath('/api/health'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        // feeds API off → Configuration opens on Job Configuration by default.
        body: JSON.stringify({ status: 'ok', corpus_library_api: true, corpus_digest_api: true, operator_config_api: true }),
      })
    })
    await stubCompanionApis(page)
    await page.route(matchExactApiPath('/api/operator-config'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          corpus_path: '/mock/corpus',
          operator_config_path: '/mock/viewer_operator.yaml',
          content: OPERATOR_YAML,
          available_profiles: [],
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await page.getByTestId('status-bar-sources-trigger').click()
    await page.getByTestId('sources-dialog-tab-operator').click()
    await page.getByTestId('sources-dialog-operator-subtab-config').click()

    const preview = page.getByTestId('cron-schedule-preview')
    await expect(preview).toBeVisible()
    await expect(preview.getByTestId('cron-schedule-preview-row-0')).toContainText('next:')
    await expect(preview.getByTestId('cron-schedule-preview-invalid-1')).toBeVisible()
    await expect(preview.getByTestId('cron-schedule-preview-invalid-summary')).toContainText('1 invalid')
  })
})
