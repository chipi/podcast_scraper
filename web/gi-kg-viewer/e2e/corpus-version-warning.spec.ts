import { expect, test } from '@playwright/test'
import { mockSignIn } from './helpers'

/**
 * The corpus-version warning banner, against the REAL backend (#1619).
 *
 * This used to fulfil `/api/health` with a hand-written warning string. It does not need to: the
 * committed fixture corpus carries no `produced_by` stamp, so a real server genuinely reports a
 * version warning for it — the same code path, with the payload the server actually produces
 * rather than one a spec author imagined.
 *
 * Asserted on the STRUCTURE (the banner renders, and shows the server's own text) rather than on a
 * fixed sentence, so a reworded warning is not a test failure while a missing banner still is.
 */
test.describe('Corpus version warning banner', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('shows the banner when health reports a corpus_version_warning', async ({ page }) => {
    const health = await page.request.get('/api/health')
    expect(health.ok()).toBe(true)
    const { corpus_version_warning: warning } = (await health.json()) as {
      corpus_version_warning: string | null
    }
    expect(
      warning,
      'the fixture corpus is expected to trigger a version warning (no produced_by stamp)',
    ).toBeTruthy()

    await page.goto('/')
    const banner = page.getByTestId('corpus-version-warning-banner')
    await expect(banner).toBeVisible()
    // The banner must carry the server's own message, not a static string.
    await expect(banner).toContainText((warning as string).slice(0, 40))
  })
})
