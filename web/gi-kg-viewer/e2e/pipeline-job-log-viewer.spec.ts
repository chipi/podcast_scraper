import { expect, test } from '@playwright/test'
import { setupDashboardApiMocks } from './dashboardApiMocks'
import {
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

const LOG_TEXT = 'pipeline step 1 starting\npipeline step 2 running\nlast line of the tail'

/**
 * #695 — in-app log viewer modal. Replaces the per-row "open log in new tab"
 * download links with a modal that tails the log, refreshes, and copies.
 */
test.describe('Pipeline job log viewer (#695)', () => {
  test('opens a modal, refreshes, copies, and closes', async ({ page }) => {
    let tailCalls = 0

    await setupDashboardApiMocks(page)
    // Enable the jobs API capability (default health mock leaves it off).
    await page.route('**/api/health', async (route) => {
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
    await page.route('**/api/jobs?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          jobs: [
            {
              job_id: 'job-running-001',
              command_type: 'pipeline',
              status: 'running',
              created_at: '2026-06-18T12:00:00Z',
              started_at: '2026-06-18T12:00:01Z',
              ended_at: null,
              pid: 4242,
              argv_summary: '[]',
              exit_code: null,
              log_relpath: 'logs/job-running-001.log',
              error_reason: null,
            },
          ],
        }),
      })
    })
    await page.route('**/api/jobs/subprocess-log-tail?**', async (route) => {
      tailCalls += 1
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ text: LOG_TEXT, truncated: false }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')

    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await page.getByTestId('briefing-card').waitFor({ state: 'visible' })

    await page.getByTestId('dashboard-tab-pipeline').click()
    await page.getByTestId('dashboard-pipeline-subtab-jobs').click()

    // The row's "View log" affordance is the new modal trigger (was a download link).
    const viewLog = page.getByTestId('pipeline-job-log-link').first()
    await viewLog.click()

    const dialog = page.getByTestId('pipeline-job-log-dialog')
    await expect(dialog).toBeVisible()
    await expect(page.getByTestId('pipeline-job-log-body')).toContainText('last line of the tail')

    // A "Download full log" escape hatch is preserved inside the modal.
    await expect(page.getByTestId('pipeline-job-log-download')).toHaveAttribute(
      'href',
      /\/api\/jobs\/subprocess-log\?/,
    )

    // Manual Refresh issues another tail request.
    const before = tailCalls
    await page.getByTestId('pipeline-job-log-refresh').click()
    await expect.poll(() => tailCalls).toBeGreaterThan(before)

    // Copy reacts (Firefox may block the async clipboard; the util falls back and
    // the button reflects either outcome).
    await page.getByTestId('pipeline-job-log-copy').click()
    await expect(page.getByTestId('pipeline-job-log-copy')).toHaveText(/Copied|Copy failed/)

    // Close via the explicit Close button.
    await page.getByTestId('pipeline-job-log-close').click()
    await expect(dialog).toBeHidden()
  })
})
