import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  requireSerialCorpusAccess,
  SHELL_HEADING_RE,
  signInAsAdmin,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * #709 — Scheduled section under Configuration. Renders `GET /api/scheduled-jobs` and disables a
 * job by rewriting `enabled:` in the operator YAML via `PUT /api/operator-config`; the row's
 * next-run then shows "—".
 *
 * #1619 — migrated to the live API, including the write.
 *
 * This was previously blocked for a reason that has now been removed rather than worked around:
 * the operator plane writes into whatever corpus it is given, and the fixture corpus is
 * force-included by `.gitignore`, so a live run dirtied the tracked tree.
 * `e2e/run-local-stack.sh` now serves from a disposable copy, so writing is safe.
 *
 * What that buys is a real round-trip. The old version kept `nightlyEnabled` in a closure and had
 * its own `PUT` handler flip it — so "disabling a job persists" was a statement about the mock's
 * local variable. Here the spec seeds the YAML through the real API, the server computes
 * `next_run_at` from the cron itself, and after the toggle the assertion reads the operator config
 * **back off the server** to prove `enabled: false` landed on disk.
 */

const SEED_YAML = `scheduled_jobs:
  - name: nightly
    cron: "0 2 * * *"
    enabled: true
  - name: weekly
    cron: "0 3 * * 0"
    enabled: false
`

/** Write the starting operator config through the real API (safe: disposable corpus copy). */
async function seedOperatorConfig(page: Page, corpusPath: string): Promise<void> {
  const resp = await page.request.put(
    `/api/operator-config?path=${encodeURIComponent(corpusPath)}`,
    { data: { content: SEED_YAML } },
  )
  if (!resp.ok()) {
    throw new Error(`seedOperatorConfig: PUT /api/operator-config returned ${resp.status()}`)
  }
}

/** Read the operator YAML back off the server. */
async function readOperatorConfig(page: Page, corpusPath: string): Promise<string> {
  const resp = await page.request.get(
    `/api/operator-config?path=${encodeURIComponent(corpusPath)}`,
  )
  const body = (await resp.json()) as { content: string }
  return body.content
}

/** SERIAL: this test rewrites the served corpus's `viewer_operator.yaml`. See e2e/README.md. */
test.describe.configure({ mode: 'serial' })

test.describe('Scheduled jobs section (#709)', () => {
  test('lists schedules and disabling one persists via operator-config PUT', async ({
    page,
  }, testInfo) => {
    requireSerialCorpusAccess(testInfo)
    await signInAsAdmin(page)

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    const corpusPath = await liveCorpusRoot(page)
    await seedOperatorConfig(page, corpusPath)

    /* Enter commits the path — without it the dependent operator fetches never fire. */
    await statusBarCorpusPathInput(page).fill(corpusPath)
    await statusBarCorpusPathInput(page).press('Enter')

    await page.getByTestId('status-bar-sources-trigger').click()
    await page.getByTestId('sources-dialog-tab-scheduled').click()

    await expect(page.getByTestId('scheduled-jobs-section')).toBeVisible()
    await expect(page.getByTestId('scheduled-jobs-row-0')).toContainText('nightly')
    /* The server computes `next_run_at` from the cron — the enabled job has one, so the cell shows
     * a relative time rather than the em-dash. */
    await expect(page.getByTestId('scheduled-jobs-next-0')).toContainText('in ')

    // Disable nightly → the app PUTs a rewritten YAML; the row's next-run collapses to "—".
    await page.getByTestId('scheduled-jobs-toggle-0').click()
    await expect(page.getByTestId('scheduled-jobs-next-0')).toHaveText('—')

    /* The point of the migration: it really persisted. Read the YAML back off the server rather
     * than trusting the request body the UI sent. */
    await expect
      .poll(async () => readOperatorConfig(page, corpusPath), { timeout: 15_000 })
      .toMatch(/name:\s*nightly[\s\S]*?enabled:\s*false/)

    // weekly is untouched by the toggle — the rewrite must not drop sibling entries.
    expect(await readOperatorConfig(page, corpusPath)).toContain('name: weekly')
  })
})
