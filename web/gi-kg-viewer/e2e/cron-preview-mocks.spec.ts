import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  requireSerialCorpusAccess,
  SHELL_HEADING_RE,
  signInAsAdmin,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * #709 — live cron preview + validation under the Job Configuration editor.
 * A valid schedule shows a next-run preview; a bad cron is flagged before save.
 *
 * #1619 — migrated to the live API.
 *
 * Previously blocked because the test needs the corpus seeded with a **deliberately invalid**
 * cron, and writing to the fixture corpus dirtied a tracked tree. `e2e/run-local-stack.sh` now
 * serves from a disposable copy, so the spec seeds its own starting state through the real
 * `PUT /api/operator-config` — no committed corpus has to ship a broken schedule on its behalf.
 *
 * The preview itself is now genuinely server-informed: the valid row's next-run is computed from
 * the cron by the backend rather than read back out of a stubbed payload.
 */

/** One valid schedule and one intentionally malformed cron, which is the whole point. */
const SEED_YAML = `scheduled_jobs:
  - name: nightly
    cron: "0 2 * * *"
    enabled: true
  - name: broken
    cron: nope
    enabled: true
`

async function seedOperatorConfig(page: Page, corpusPath: string): Promise<void> {
  const resp = await page.request.put(
    `/api/operator-config?path=${encodeURIComponent(corpusPath)}`,
    { data: { content: SEED_YAML } },
  )
  if (!resp.ok()) {
    throw new Error(`seedOperatorConfig: PUT /api/operator-config returned ${resp.status()}`)
  }
}

/** SERIAL: this test rewrites the served corpus's `viewer_operator.yaml`. See e2e/README.md. */
test.describe.configure({ mode: 'serial' })

test.describe('Cron schedule preview (#709)', () => {
  test('previews valid schedules and flags an invalid cron in the editor', async ({
    page,
  }, testInfo) => {
    requireSerialCorpusAccess(testInfo)
    await signInAsAdmin(page)

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
    const corpusPath = await liveCorpusRoot(page)
    await seedOperatorConfig(page, corpusPath)

    // Enter commits the path so the operator fetches fire.
    await statusBarCorpusPathInput(page).fill(corpusPath)
    await statusBarCorpusPathInput(page).press('Enter')

    await page.getByTestId('status-bar-sources-trigger').click()
    await page.getByTestId('sources-dialog-tab-operator').click()
    await page.getByTestId('sources-dialog-operator-subtab-config').click()

    const preview = page.getByTestId('cron-schedule-preview')
    await expect(preview).toBeVisible()
    // Row 0 is the valid schedule → a computed next run.
    await expect(preview.getByTestId('cron-schedule-preview-row-0')).toContainText('next:')
    // Row 1 is `cron: nope` → flagged before save, and counted in the summary.
    await expect(preview.getByTestId('cron-schedule-preview-invalid-1')).toBeVisible()
    await expect(preview.getByTestId('cron-schedule-preview-invalid-summary')).toContainText(
      '1 invalid',
    )
  })
})
