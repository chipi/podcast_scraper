/**
 * T4 — HandoffErrorStrip render test (Playwright; the viewer has no
 * `@vue/test-utils` mount infra so component-level Vitest tests aren't an
 * option for `.vue` SFCs without that setup).
 *
 * Decision #15: failed handoffs surface a visible
 * `data-testid="handoff-error-strip"` element with a reason and a dismiss
 * button. Replaces the silent swallow at GraphCanvas.vue:901-903 (now wired
 * through `handoffFailed` in C6 → store sets `lastResult.status = 'failed'`).
 *
 * Regression this catches:
 * - someone refactors `<HandoffErrorStrip />` out of `GraphTabPanel.vue`
 * - someone breaks the `v-if="visible"` condition tied to `lastResult.status`
 * - someone removes the `data-testid` (test goes red)
 *
 * The test triggers the failure via the territory-fetch-404 path (real
 * production code path; not a synthetic store mutation).
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from '../helpers'
import { setupHandoffMatrixMocks } from './_handoff-helpers'

test.describe('HandoffErrorStrip § T4', () => {
  test('renders with reason on handoffFailed; dismiss removes it', async ({
    page,
  }) => {
    // Use the dev-only `__GIKG_HANDOFF_STORE__` hook to fire `handoffFailed`
    // directly without setting up the full territory-fetch-404 pipeline.
    // The hook is exposed only when `import.meta.env.DEV` is true (Vite
    // dev mode); production builds don't ship it.
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    // Wait for GraphTabPanel to mount (where HandoffErrorStrip lives).
    await page.waitForTimeout(500)

    // Trigger handoffFailed via the store dev hook.
    await page.evaluate(() => {
      const store = (
        window as unknown as {
          __GIKG_HANDOFF_STORE__?: { handoffFailed: (reason: string) => void }
        }
      ).__GIKG_HANDOFF_STORE__
      store?.handoffFailed('episode not found (test-injected)')
    })

    // The strip becomes visible when lastResult.status === 'failed'.
    const strip = page.getByTestId('handoff-error-strip')
    await expect(strip).toBeVisible({ timeout: 5000 })
    await expect(strip).toContainText(/could not open episode in graph/i)
    await expect(strip).toContainText(/episode not found \(test-injected\)/i)

    // Dismiss: click the dismiss button; the strip clears lastResult and
    // hides itself.
    await page.getByTestId('handoff-error-strip-dismiss').click()
    await expect(strip).toBeHidden()
  })

  test('strip is NOT visible when no handoff has failed', async ({ page }) => {
    // Sanity test: on a clean graph view with no failure, the strip does
    // NOT render. Catches the regression where a default state would
    // accidentally show the strip on every page load.
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()

    // Wait for the graph panel to mount and the canvas surface to settle.
    await page.waitForTimeout(1000)
    await expect(page.getByTestId('handoff-error-strip')).toHaveCount(0)
  })
})
