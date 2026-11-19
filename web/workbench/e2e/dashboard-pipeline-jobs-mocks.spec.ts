import { expect, test } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  SHELL_HEADING_RE,
  signInAsAdmin,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * #1619 — migrated to the live API, and the first operator-plane spec to do so.
 *
 * Two things make the operator plane different from the corpus reads:
 *
 * 1. `/api/jobs` is only *mounted* when the server runs with `PODCAST_SERVE_ENABLE_JOBS_API=1`
 *    (`run-local-stack.sh` does not set it — see e2e/README).
 * 2. It is then gated by `OperatorWriteGuard`, which requires a **real admin session cookie**.
 *    `mockSignIn` only stubs `/api/app/auth/status` in the browser, so it does NOT satisfy the
 *    guard — every operator request would 403. Hence `signInAsAdmin`, which drives the real
 *    mock-OAuth round trip.
 *
 * The empty-jobs state asserted here is what a freshly-served corpus genuinely returns, so this
 * needed no fixture at all — only a backend.
 *
 * **Known side effect, unresolved:** reading `/api/jobs` makes the server create
 * `.viewer/jobs.jsonl.lock` in the corpus directory. `.gitignore:82` force-includes
 * `tests/fixtures/app-validation-corpus/**`, so a live run leaves that one untracked file behind
 * (verified: a full-suite run on 2026-08-14 produced exactly it, and nothing else). Either the
 * lock file stops being tracked or these runs serve from a corpus copy — a fixture-policy
 * decision, not a test change. Until then, `git status` after a live run will show it.
 */
test.describe('Dashboard — Pipeline jobs card', () => {
  test('Pipeline tab shows jobs card and empty list when jobs_api is true', async ({ page }) => {
    await signInAsAdmin(page)

    /* The server must actually be advertising the jobs API, or this test would silently assert
     * the wrong absence — the card hides entirely when `jobs_api` is false. */
    const health = await (await page.request.get('/api/health')).json()
    expect(health.jobs_api).toBe(true)

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    /* Enter COMMITS the corpus path — `fill` alone leaves it uncommitted, and the dashboard's
     * dependent fetches (including `/api/jobs`) never fire, so the card sits on "Loading…"
     * forever with no request on the wire. Same pattern as `pipeline.spec.ts`'s
     * `navigateToPipelineTab`. */
    const corpusPath = await liveCorpusRoot(page)
    await statusBarCorpusPathInput(page).fill(corpusPath)
    await statusBarCorpusPathInput(page).press('Enter')
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()

    const tablist = page.getByRole('tablist', { name: 'Dashboard tabs' })
    await expect(tablist).toBeVisible({ timeout: 15_000 })
    await tablist.getByRole('tab', { name: 'Pipeline' }).click()

    const card = page.getByTestId('pipeline-jobs-card')
    await expect(card).toBeVisible({ timeout: 30_000 })
    /* Scoped to the card, and given a generous budget.
     *
     * Two things are being waited on. The Pipeline tab renders the *runs* empty state ("No
     * pipeline runs found…") immediately while the jobs card fills in once `/api/jobs` resolves —
     * so a page-wide `getByText` raced it. And the API serves this on a SINGLE worker, where a
     * concurrent spec's live search (query embedding + LanceDB) holds it for seconds, so under
     * `--workers=2` this request can sit in the queue well past a 15s budget. Measured alone the
     * whole test takes ~10s; the wait is for contention, not for a defect. */
    await expect(card).toContainText('No jobs yet. Run queues a CLI pipeline for this corpus.', {
      timeout: 60_000,
    })
  })
})
