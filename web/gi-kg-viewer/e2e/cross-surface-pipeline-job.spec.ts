import { expect, test } from '@playwright/test'
import { liveCorpusRoot, mainViewsNav, SHELL_HEADING_RE, signInAsAdmin, statusBarCorpusPathInput } from './helpers'

/**
 * A pipeline job finishing reloads the KEPT-ALIVE Library tab (2026-10-09).
 *
 * Library and Digest are kept alive and reloaded only on a corpus-path or health change, so a job
 * run from the Dashboard left them showing the corpus as it was before the job — and once the fix
 * existed, it only fired while the Dashboard was open. Same shape as the player's
 * `e2e/cross-surface.spec.ts`: open the READER first, make the change elsewhere, come back IN-APP.
 *
 * Only `/api/jobs` is stubbed (a job seen running, then succeeded) — the corpus reads are live, so
 * what is asserted is that Library ASKED the server again, which is the whole contract.
 */
test('Library re-reads the corpus when a job it did not watch finishes', async ({ page }) => {
  await signInAsAdmin(page)
  let jobState: 'running' | 'succeeded' = 'running'
  await page.route('**/api/jobs?**', async (route) => {
    if (route.request().method() !== 'GET') return route.continue()
    await route.fulfill({
      json: {
        path: '/corpus',
        jobs: [
          {
            job_id: 'e2e-job-1',
            command_type: 'pipeline',
            status: jobState,
            created_at: '2026-10-09T00:00:00Z',
            started_at: '2026-10-09T00:00:01Z',
            ended_at: jobState === 'succeeded' ? '2026-10-09T00:01:00Z' : null,
            pid: null,
            argv_summary: 'pipeline',
            exit_code: jobState === 'succeeded' ? 0 : null,
            log_relpath: '',
            error_reason: null,
          },
        ],
      },
    })
  })

  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await statusBarCorpusPathInput(page).press('Enter')

  // 1. the reader, mounted and kept alive
  await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
  await expect(page.getByTestId('library-root')).toBeVisible()

  // 2. elsewhere: the Dashboard sees the job running …
  await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
  const tablist = page.getByRole('tablist', { name: 'Dashboard tabs' })
  await expect(tablist).toBeVisible({ timeout: 15_000 })
  await tablist.getByRole('tab', { name: 'Pipeline' }).click()
  await expect(page.getByTestId('pipeline-jobs-card')).toContainText('running', { timeout: 30_000 })

  // … then the operator LEAVES the Dashboard, and the job finishes while nothing on screen lists jobs.
  await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
  const reread = page.waitForRequest((r) => r.url().includes('/api/corpus/feeds'), { timeout: 30_000 })
  jobState = 'succeeded'

  // 3. Library — already open — asks the corpus again once the session-wide watcher sees it finish.
  await reread
})
