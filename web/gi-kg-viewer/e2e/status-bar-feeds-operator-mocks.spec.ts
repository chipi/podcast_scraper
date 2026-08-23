import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  requireSerialCorpusAccess,
  SHELL_HEADING_RE,
  signInAsAdmin,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * Status-bar Configuration dialog — Feeds + Operator YAML sections, against the real backend.
 *
 * #1619 — migrated. The header this file used to carry described the blocker exactly:
 *
 *   "With a real server, `GET /api/operator-config` may **create** `viewer_operator.yaml` … when
 *    the file is missing"
 *
 * That write is why the file stayed stubbed — the fixture corpus is force-included by
 * `.gitignore`, so a live run left the tracked tree dirty. `e2e/run-local-stack.sh` now serves
 * from a disposable copy, so the writes are safe and the seeding happens through the real API.
 *
 * Two kinds of assertion, kept apart on purpose:
 *
 * * **UI contract** — what the app puts on the wire (e.g. that Apply JSON does NOT trim the URLs
 *   the operator typed). Observed with `page.on('request')`, which reads the real request rather
 *   than intercepting it.
 * * **Persistence** — read back off the server afterwards. The old version asserted only the
 *   captured request body, so the server could have rejected, reordered or dropped an entry and
 *   every test would still have passed.
 */

/** Sign in as admin, land the shell, and return the corpus path the server is serving. */
async function openShell(page: Page): Promise<string> {
  await signInAsAdmin(page)
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 60_000 })
  return liveCorpusRoot(page)
}

/** Commit the corpus path — `fill` alone leaves it uncommitted and the operator fetches never fire. */
async function commitCorpusPath(page: Page, corpusPath: string): Promise<void> {
  await statusBarCorpusPathInput(page).fill(corpusPath)
  await statusBarCorpusPathInput(page).press('Enter')
}

async function seedFeeds(page: Page, corpusPath: string, feeds: unknown[]): Promise<void> {
  const resp = await page.request.put(`/api/feeds?path=${encodeURIComponent(corpusPath)}`, {
    data: { feeds },
  })
  if (!resp.ok()) throw new Error(`seedFeeds: PUT /api/feeds returned ${resp.status()}`)
}

async function readFeeds(page: Page, corpusPath: string): Promise<unknown[]> {
  const resp = await page.request.get(`/api/feeds?path=${encodeURIComponent(corpusPath)}`)
  return ((await resp.json()) as { feeds: unknown[] }).feeds
}

async function seedOperatorConfig(page: Page, corpusPath: string, content: string): Promise<void> {
  const resp = await page.request.put(
    `/api/operator-config?path=${encodeURIComponent(corpusPath)}`,
    { data: { content } },
  )
  if (!resp.ok()) {
    throw new Error(`seedOperatorConfig: PUT /api/operator-config returned ${resp.status()}`)
  }
}

async function readOperatorConfig(page: Page, corpusPath: string): Promise<string> {
  const resp = await page.request.get(
    `/api/operator-config?path=${encodeURIComponent(corpusPath)}`,
  )
  return ((await resp.json()) as { content: string }).content
}

/**
 * SERIAL: every test here rewrites `feeds.spec.yaml` / `viewer_operator.yaml` in the one corpus the
 * stack serves. `playwright.config.ts` sets `fullyParallel: true`, so without this the tests in
 * this file would seed over each other mid-run. See the note in e2e/README.md about specs that
 * mutate shared server state.
 */
test.describe.configure({ mode: 'serial' })

test.describe('Status bar — Feeds + Operator configuration', () => {
  test.beforeEach(async ({}, testInfo) => {
    requireSerialCorpusAccess(testInfo)
  })

  test('opens Feeds tab without calling operator-config when only feeds is opened', async ({
    page,
  }) => {
    const corpusPath = await openShell(page)
    await seedFeeds(page, corpusPath, ['https://seed.example/rss'])

    /* Observe rather than intercept: the claim is that opening Feeds does not *also* fetch the
     * operator YAML (it is a separate, more expensive section). */
    const operatorGets: string[] = []
    page.on('request', (r) => {
      const u = new URL(r.url())
      if (u.pathname.replace(/\/$/, '') === '/api/operator-config' && r.method() === 'GET') {
        operatorGets.push(r.url())
      }
    })

    await commitCorpusPath(page, corpusPath)
    await expect(page.getByTestId('status-bar-sources-trigger')).toBeVisible({ timeout: 15_000 })
    await page.getByTestId('status-bar-sources-trigger').click()
    await expect(page.getByTestId('status-bar-sources-dialog')).toBeVisible()
    // IA: sections live in a left sub-nav rail (restructured from a top tab-strip).
    await expect(
      page.getByTestId('status-bar-sources-dialog').getByRole('navigation', {
        name: 'Configuration sections',
      }),
    ).toBeVisible()
    await expect(page.getByTestId('sources-dialog-feeds-list')).toBeVisible()
    await expect(page.getByTestId('sources-dialog-feeds-row-0')).toContainText(
      'https://seed.example/rss',
    )
    await page.getByTestId('sources-dialog-feeds-panel-json').click()
    await expect(page.getByTestId('sources-dialog-feeds-textarea')).toHaveValue(
      '{\n  "feeds": [\n    "https://seed.example/rss"\n  ]\n}',
    )
    expect(operatorGets).toHaveLength(0)
  })

  test('Apply JSON sends PUT with parsed feeds array', async ({ page }) => {
    const corpusPath = await openShell(page)
    await seedFeeds(page, corpusPath, [])

    /* UI contract: the operator's exact strings go on the wire — surrounding whitespace is NOT
     * trimmed by the client. Observed from the real request. */
    let putFeeds: unknown[] | null = null
    page.on('request', (r) => {
      const u = new URL(r.url())
      if (u.pathname.replace(/\/$/, '') === '/api/feeds' && r.method() === 'PUT') {
        const body = r.postDataJSON() as { feeds?: unknown[] }
        putFeeds = Array.isArray(body.feeds) ? body.feeds : []
      }
    })

    await commitCorpusPath(page, corpusPath)
    await expect(page.getByTestId('status-bar-sources-trigger')).toBeVisible({ timeout: 15_000 })
    await page.getByTestId('status-bar-sources-trigger').click()
    await page.getByTestId('sources-dialog-feeds-panel-json').click()
    await page
      .getByTestId('sources-dialog-feeds-textarea')
      .fill('{\n  "feeds": [\n    "  https://a.example/x  ",\n    "https://b.example/y"\n  ]\n}')
    await page.getByTestId('sources-dialog-feeds-apply-json').click()

    await expect.poll(() => putFeeds, { timeout: 15_000 }).toEqual([
      '  https://a.example/x  ',
      'https://b.example/y',
    ])
    // …and the server accepted it: two entries persisted.
    await expect.poll(async () => (await readFeeds(page, corpusPath)).length, {
      timeout: 15_000,
    }).toBe(2)
  })

  test('Add feed appends URL and saves via PUT', async ({ page }) => {
    const corpusPath = await openShell(page)
    await seedFeeds(page, corpusPath, ['https://existing.example/rss'])

    await commitCorpusPath(page, corpusPath)
    await expect(page.getByTestId('status-bar-sources-trigger')).toBeVisible({ timeout: 15_000 })
    await page.getByTestId('status-bar-sources-trigger').click()
    await page.getByTestId('sources-dialog-feeds-add-url').fill('https://new.example/feed')
    await page.getByTestId('sources-dialog-feeds-add-btn').click()

    await expect(page.getByTestId('sources-dialog-feeds-row-1')).toContainText(
      'https://new.example/feed',
    )
    // Appended, not replaced — and it survived the round trip to disk.
    await expect
      .poll(async () => readFeeds(page, corpusPath), { timeout: 15_000 })
      .toEqual(['https://existing.example/rss', 'https://new.example/feed'])
  })

  test('Operator tab loads YAML and save sends PUT body', async ({ page }) => {
    const corpusPath = await openShell(page)
    await seedFeeds(page, corpusPath, [])
    await seedOperatorConfig(page, corpusPath, 'keep: true\n')

    await commitCorpusPath(page, corpusPath)
    await expect(page.getByTestId('status-bar-sources-trigger')).toBeVisible({ timeout: 15_000 })
    await page.getByTestId('status-bar-sources-trigger').click()
    await page.getByTestId('sources-dialog-tab-operator').click()
    await page.getByTestId('sources-dialog-operator-subtab-config').click()

    const textarea = page.getByTestId('sources-dialog-operator-textarea')
    await expect(textarea).toBeVisible()
    // The editor loads what the server actually holds.
    await expect(textarea).toHaveValue('keep: true')
    await textarea.fill('keep: true\nextra: 2\n')
    await page.getByTestId('sources-dialog-save-overrides').click()

    await expect
      .poll(async () => readOperatorConfig(page, corpusPath), { timeout: 15_000 })
      .toBe('keep: true\nextra: 2\n')
  })

  test('health tab in corpus dialog lists feeds, operator, and pipeline jobs API rows as Yes', async ({
    page,
  }) => {
    const corpusPath = await openShell(page)

    /* These three rows mirror server capabilities, which are only true when the stack was started
     * with the PODCAST_SERVE_ENABLE_* env (see e2e/run-local-stack.sh). Assert the premise so a
     * mis-configured stack fails here, loudly, instead of somewhere downstream. */
    const health = await (await page.request.get('/api/health')).json()
    expect({
      feeds: health.feeds_api,
      operator: health.operator_config_api,
      jobs: health.jobs_api,
    }).toEqual({ feeds: true, operator: true, jobs: true })

    await commitCorpusPath(page, corpusPath)
    await page.getByTestId('status-bar-health-trigger').click()
    const dialog = page.getByTestId('status-bar-sources-dialog')
    const healthPanel = page.getByTestId('sources-dialog-health-panel')
    await expect(dialog).toBeVisible()
    await expect(healthPanel).toBeVisible()
    await expect(healthPanel.getByText('Feeds file API')).toBeVisible()
    await expect(healthPanel.getByText('Operator YAML API')).toBeVisible()
    await expect(healthPanel.getByText('Pipeline jobs API')).toBeVisible()
    const feedsDt = healthPanel.locator('dt').filter({ hasText: /Feeds file API/ })
    await expect(feedsDt.locator('xpath=./following-sibling::dd[1]')).toHaveText('Yes')
    const opDt = healthPanel.locator('dt').filter({ hasText: /Operator YAML API/ })
    await expect(opDt.locator('xpath=./following-sibling::dd[1]')).toHaveText('Yes')
    const jobsDt = healthPanel.locator('dt').filter({ hasText: /Pipeline jobs API/ })
    await expect(jobsDt.locator('xpath=./following-sibling::dd[1]')).toHaveText('Yes')
  })
})
