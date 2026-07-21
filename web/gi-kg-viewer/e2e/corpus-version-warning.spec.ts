import { expect, test } from '@playwright/test'
import { mockSignIn } from './helpers'

test.describe('Corpus version warning banner (mocked API)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })
  test('shows banner when health reports corpus_version_warning', async ({ page }) => {
    await page.route('**/api/health**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_version_warning:
            'Corpus produced by 2.5.0 is below server minimum 2.6.0. Reprocess from transcripts.',
        }),
      })
    })

    await page.goto('/')
    const banner = page.getByTestId('corpus-version-warning-banner')
    await expect(banner).toBeVisible()
    await expect(banner).toContainText('below server minimum 2.6.0')
  })
})
