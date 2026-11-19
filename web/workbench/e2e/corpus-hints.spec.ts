import { expect, test } from '@playwright/test'
import {
  liveFeedMetadataDir,
  mockSignIn,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * #1619 — migrated to the live API.
 *
 * The hint this asserts is produced by the server, not by the spec: point `/api/artifacts` at a
 * per-feed `metadata/` directory inside a multi-feed corpus and it answers with a `hints` entry
 * naming the corpus root to use instead. The old version hand-wrote that sentence, so it passed
 * whether or not the server still emitted one.
 */
test.describe('Corpus path hints', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('List shows the corpus path hint the API returns for a feed subdirectory', async ({
    page,
  }) => {
    await page.goto('/')

    const metadataDir = await liveFeedMetadataDir(page)
    await statusBarCorpusPathInput(page).fill(metadataDir)
    await page.getByTestId('status-bar-list-artifacts').click()

    /* Scope to the artifact-list dialog: the same hint text also renders in the sources dialog's
     * index banner, so an unscoped locator is a strict-mode violation (found on migration — the
     * hand-written mock asserted a string that only ever existed in one place). */
    const dialog = page.getByTestId('artifact-list-dialog')
    await expect(dialog.getByText('Corpus path hint')).toBeVisible()

    /* The server's own wording, read back from the API — survives a reworded hint, still fails
     * when the hint stops being emitted or stops reaching the status bar. */
    const resp = await page.request.get(
      `/api/artifacts?path=${encodeURIComponent(metadataDir)}`,
    )
    const { hints } = (await resp.json()) as { hints?: string[] }
    expect(hints?.length ?? 0).toBeGreaterThan(0)
    await expect(dialog.getByText(hints![0]!, { exact: false })).toBeVisible()
  })
})
