import { expect, test } from '@playwright/test'

/**
 * Operator first-run UX from an empty corpus subdir (#693 part 2 / A).
 *
 * Differentiation from ``stack-jobs-flow.spec.ts`` (which seeds the
 * corpus before running): this spec exercises what a real codespace
 * operator sees on day one — a corpus path with no ``feeds.spec.yaml``
 * and no ``viewer_operator.yaml`` saved yet.
 *
 * Coverage:
 *
 * 1. ``GET /api/operator-config`` on an empty corpus subdir auto-seeds
 *    ``viewer_operator.yaml`` from the packaged example. Returns 200
 *    + non-empty content; ``available_profiles`` is populated from
 *    on-disk ``config/profiles/`` (intersected with the env allowlist
 *    if any).
 *
 * 2. ``GET /api/feeds`` on the same path returns an empty feeds list
 *    (api treats missing ``feeds.spec.yaml`` as "no feeds yet").
 *
 * 3. Adding a feed via the Configuration → Feeds dialog persists to
 *    disk; subsequent GET reflects the new feed.
 *
 * 4. Picking a profile in the Job Profile dropdown + Save merges the
 *    packaged preset overrides under ``profile:`` on disk; subsequent
 *    GET shows the new content.
 *
 * Out of scope (covered by ``stack-jobs-flow.spec.ts``):
 *   - Running the actual pipeline + validating artifacts. That's a
 *     long-running flow; running it again from an empty corpus
 *     largely duplicates the existing spec's assertions. The
 *     differentiated value here is the empty-state UX itself.
 *
 * Uses ``/app/output/firstrun-empty`` as the corpus path — a fresh
 * subdir of the bind-mounted corpus volume that the api will create
 * on first GET.
 */

const CORPUS = '/app/output/firstrun-empty'
const MOCK_FEED = 'http://mock-feeds/p01_fast_with_transcript.xml'

function stackTestProgress(msg: string): void {
  // eslint-disable-next-line no-console
  console.log(`[operator-first-run] ${msg}`)
}

async function waitForHealthFlag(
  request: import('@playwright/test').APIRequestContext,
  key: 'feeds_api' | 'operator_config_api' | 'jobs_api',
  timeoutMs: number,
): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const res = await request.get('/api/health')
    if (res.ok()) {
      const body = (await res.json()) as Record<string, unknown>
      if (body[key] === true) {
        return
      }
    }
    await new Promise((r) => setTimeout(r, 1500))
  }
  throw new Error(`timeout waiting for /api/health ${key}`)
}

test.describe('Operator first-run UX from an empty corpus subdir (#693)', () => {
  test.describe.configure({ mode: 'serial' })

  test.beforeEach(async ({ request }) => {
    // The stack must already be seeded (via ``make stack-test-seed``)
    // for the parent ``/app/output`` corpus root, but our subdir
    // ``firstrun-empty/`` is fresh. We just need the api up.
    await waitForHealthFlag(request, 'operator_config_api', 60_000)
  })

  test('GET /api/operator-config on empty subdir auto-seeds viewer_operator.yaml', async ({
    request,
  }) => {
    stackTestProgress(`GET /api/operator-config?path=${CORPUS}`)
    const res = await request.get('/api/operator-config', { params: { path: CORPUS } })
    expect(res.ok(), `status=${res.status()}`).toBeTruthy()

    const body = (await res.json()) as {
      corpus_path: string
      operator_config_path: string
      content: string
      available_profiles?: string[]
      default_profile?: string | null
    }
    expect(body.corpus_path).toContain('/firstrun-empty')
    expect(body.operator_config_path).toMatch(/viewer_operator\.yaml$/)
    expect(
      body.content.length,
      'first GET should auto-seed viewer_operator.yaml from packaged example',
    ).toBeGreaterThan(0)
    // Packaged example has ``max_episodes`` + ``transcribe_missing`` etc.
    expect(body.content).toMatch(/max_episodes|transcribe_missing|reuse_media/)

    // available_profiles always populated from on-disk discovery.
    const presets = body.available_profiles ?? []
    expect(presets.length, 'available_profiles must be non-empty').toBeGreaterThan(0)
  })

  test('GET /api/feeds on empty subdir returns empty feeds list', async ({ request }) => {
    stackTestProgress(`GET /api/feeds?path=${CORPUS}`)
    const res = await request.get('/api/feeds', { params: { path: CORPUS } })
    expect(res.ok(), `status=${res.status()}`).toBeTruthy()

    const body = (await res.json()) as { file_relpath?: string; feeds?: unknown[] }
    // API reports the canonical filename even when missing.
    expect(body.file_relpath).toBe('feeds.spec.yaml')
    const feeds = Array.isArray(body.feeds) ? body.feeds : []
    expect(feeds, 'fresh corpus should have no feeds yet').toEqual([])
  })

  test('Adding feed through Configuration dialog persists + GET reflects it', async ({
    page,
    request,
  }) => {
    await waitForHealthFlag(request, 'feeds_api', 60_000)
    await page.goto('/')
    await page.getByRole('heading', { name: /Podcast Intelligence Platform/i }).waitFor()

    // Set corpus path to the empty subdir.
    await page.getByTestId('status-bar-corpus-path').fill(CORPUS)
    await page.getByTestId('status-bar-corpus-path').press('Tab')
    await page.waitForTimeout(500)

    // Open Configuration → Feeds.
    await page.getByTestId('status-bar-sources-trigger').click()
    await expect(page.getByTestId('status-bar-sources-dialog')).toBeVisible()
    await page.getByTestId('sources-dialog-tab-feeds').click()

    // Add the mock feed.
    await page.getByTestId('sources-dialog-feeds-add-url').fill(MOCK_FEED)
    await page.getByTestId('sources-dialog-feeds-add-btn').click()

    // Wait for the PUT to complete (Add button re-enables once persist returns).
    await expect(page.getByTestId('sources-dialog-feeds-add-btn')).toBeEnabled({
      timeout: 60_000,
    })

    // Validate via direct API: feed shows up on disk.
    const res = await request.get('/api/feeds', { params: { path: CORPUS } })
    expect(res.ok()).toBeTruthy()
    const body = (await res.json()) as { feeds?: unknown[] }
    const urls = (body.feeds ?? []).map((e: unknown) => {
      if (typeof e === 'string') return e
      if (e && typeof e === 'object') {
        const o = e as Record<string, unknown>
        const u = o.url ?? o.rss
        return typeof u === 'string' ? u : ''
      }
      return ''
    })
    expect(urls.some((u) => u.includes('p01_fast_with_transcript'))).toBeTruthy()
  })

  test('Picking profile + Save persists profile: line on disk', async ({ page, request }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: /Podcast Intelligence Platform/i }).waitFor()

    await page.getByTestId('status-bar-corpus-path').fill(CORPUS)
    await page.getByTestId('status-bar-corpus-path').press('Tab')
    await page.waitForTimeout(500)

    // Open Configuration → Job Profile.
    await page.getByTestId('status-bar-sources-trigger').click()
    await expect(page.getByTestId('status-bar-sources-dialog')).toBeVisible()
    await page.getByTestId('sources-dialog-tab-profile').click()

    const profileSelect = page.getByTestId('sources-dialog-profile-select')
    await expect(profileSelect).toBeVisible({ timeout: 30_000 })

    // Pick airgapped_thin (default for stack-test). Wait for option to load.
    await profileSelect
      .locator('option[value="airgapped_thin"]')
      .first()
      .waitFor({ state: 'attached', timeout: 60_000 })
    await profileSelect.selectOption('airgapped_thin')

    // Save and wait for the PUT.
    const putPromise = page.waitForResponse(
      (r) =>
        r.request().method() === 'PUT' &&
        new URL(r.url()).pathname.endsWith('/api/operator-config'),
      { timeout: 60_000 },
    )
    await page.getByTestId('sources-dialog-save-profile').click()
    const put = await putPromise
    expect(put.ok(), `PUT status=${put.status()}`).toBeTruthy()

    // Read back via API: file now has profile: airgapped_thin.
    const res = await request.get('/api/operator-config', { params: { path: CORPUS } })
    expect(res.ok()).toBeTruthy()
    const body = (await res.json()) as { content: string }
    expect(body.content).toMatch(/^profile:\s*airgapped_thin/m)
  })
})
