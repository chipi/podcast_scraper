import { expect, test, type Page } from '@playwright/test'
import { SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * #709 — Scheduled section under Configuration. Renders `GET /api/scheduled-jobs`
 * and disables a job by rewriting `enabled:` in the operator YAML via
 * `PUT /api/operator-config`; the row's next-run then shows "—".
 */
function matchExactApiPath(path: string): (url: URL) => boolean {
  return (url: URL) => url.pathname.replace(/\/$/, '') === path
}

const OPERATOR_YAML = `scheduled_jobs:
  - name: nightly
    cron: "0 2 * * *"
    enabled: true
  - name: weekly
    cron: "0 3 * * 0"
    enabled: false
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
      body: JSON.stringify({
        available: false,
        stats: null,
        reindex_recommended: false,
        rebuild_in_progress: false,
        rebuild_last_error: null,
      }),
    })
  })
}

test.describe('Scheduled jobs section (#709)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin')
  })

  test('lists schedules and disabling one persists via operator-config PUT', async ({ page }) => {
    let nightlyEnabled = true
    let lastPutContent: string | null = null

    await page.route(matchExactApiPath('/api/health'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          feeds_api: true,
          operator_config_api: true,
        }),
      })
    })
    await stubCompanionApis(page)
    await page.route(matchExactApiPath('/api/scheduled-jobs'), async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          scheduler_running: true,
          timezone: 'UTC',
          jobs: [
            {
              name: 'nightly',
              cron: '0 2 * * *',
              enabled: nightlyEnabled,
              kind: 'pipeline',
              next_run_at: nightlyEnabled ? '2099-01-01T02:00:00Z' : null,
            },
            { name: 'weekly', cron: '0 3 * * 0', enabled: false, kind: 'enrichment', next_run_at: null },
          ],
        }),
      })
    })
    await page.route(matchExactApiPath('/api/operator-config'), async (route) => {
      if (route.request().method() === 'PUT') {
        lastPutContent = (route.request().postDataJSON() as { content: string }).content
        if (/name:\s*nightly[\s\S]*?enabled:\s*false/.test(lastPutContent)) {
          nightlyEnabled = false
        }
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            corpus_path: '/mock/corpus',
            operator_config_path: '/mock/viewer_operator.yaml',
            content: lastPutContent,
          }),
        })
        return
      }
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
    await page.getByTestId('sources-dialog-tab-scheduled').click()

    await expect(page.getByTestId('scheduled-jobs-section')).toBeVisible()
    await expect(page.getByTestId('scheduled-jobs-row-0')).toContainText('nightly')
    await expect(page.getByTestId('scheduled-jobs-next-0')).toContainText('in ')

    // Disable nightly → operator-config PUT carries enabled: false; row shows —.
    await page.getByTestId('scheduled-jobs-toggle-0').click()
    await expect.poll(() => lastPutContent).not.toBeNull()
    expect(/name:\s*nightly[\s\S]*?enabled:\s*false/.test(lastPutContent ?? '')).toBe(true)
    // weekly stays disabled (unchanged).
    expect(lastPutContent).toContain('- name: weekly')
    await expect(page.getByTestId('scheduled-jobs-next-0')).toHaveText('—')
  })
})
