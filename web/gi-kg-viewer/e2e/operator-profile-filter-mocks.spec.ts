import { expect, test, type Page } from '@playwright/test'
import { SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

/**
 * Operator profile dropdown filtering + default-profile preselection
 * (#692, RFC-081 §Layer 1).
 *
 * The api filters ``available_profiles`` (intersected with
 * ``PODCAST_AVAILABLE_PROFILES`` env) and may surface a
 * ``default_profile`` from ``PODCAST_DEFAULT_PROFILE``. The viewer's
 * StatusBar.vue:
 *
 *   - Renders ``<option>`` per entry in ``available_profiles``
 *     (plus a static ``<option value="">None</option>``).
 *   - Preselects ``default_profile`` when the corpus has no saved
 *     ``profile:`` line yet.
 *   - Renders an extra ``<option>`` showing the saved profile name
 *     followed by ``(custom)`` when the operator's saved choice isn't
 *     in ``available_profiles`` (legacy yaml from before the env
 *     filter was introduced).
 *
 * These specs stub ``/api/*`` so the assertions stay deterministic and
 * don't depend on what's on the operator's disk. They cover viewer-side
 * behavior; the api-side filtering + validator have their own Python
 * unit tests in ``tests/unit/server/test_profile_presets.py``.
 */

const SOURCES_DIALOG = 'status-bar-sources-dialog'
const PROFILE_SELECT = 'sources-dialog-profile-select'

function matchExactApiPath(path: string): (url: URL) => boolean {
  return (url: URL) => url.pathname.replace(/\/$/, '') === path
}

function healthBodyWithSourcesApis(): string {
  return JSON.stringify({
    status: 'ok',
    corpus_library_api: true,
    corpus_digest_api: true,
    feeds_api: true,
    operator_config_api: true,
    jobs_api: true,
  })
}

async function stubMinimalCorpusApis(page: Page): Promise<void> {
  await page.route(matchExactApiPath('/api/health'), async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: healthBodyWithSourcesApis(),
    })
  })
  await page.route(
    (url) => {
      const p = url.pathname.replace(/\/$/, '')
      return p === '/api/artifacts' || p.startsWith('/api/artifacts/')
    },
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [], hints: [] }),
      })
    },
  )
  await page.route(matchExactApiPath('/api/corpus/feeds'), async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        feeds: [],
      }),
    })
  })
  await page.route(matchExactApiPath('/api/index/stats'), async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        available: false,
        reason: null,
        stats: null,
        reindex_recommended: false,
        rebuild_in_progress: false,
        rebuild_last_error: null,
      }),
    })
  })
  await page.route(matchExactApiPath('/api/corpus/digest'), async (route) => {
    const url = new URL(route.request().url())
    const win = url.searchParams.get('window') || 'all'
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/corpus',
        window: win,
        window_start_utc: '1970-01-01T00:00:00Z',
        window_end_utc: '2026-01-01T00:00:00Z',
        compact: false,
        rows: [],
        topics: [],
        topics_unavailable_reason: null,
      }),
    })
  })
}

async function stubOperatorConfig(
  page: Page,
  payload: {
    content?: string
    available_profiles: string[]
    default_profile?: string | null
  },
): Promise<void> {
  await page.route(matchExactApiPath('/api/operator-config'), async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        corpus_path: '/mock/corpus',
        operator_config_path: '/mock/corpus/viewer_operator.yaml',
        content: payload.content ?? '',
        available_profiles: payload.available_profiles,
        default_profile: payload.default_profile ?? null,
      }),
    })
  })
}

async function openProfileTab(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

  await statusBarCorpusPathInput(page).fill('/mock/corpus')

  // Status bar's "Configuration" button opens the sources dialog;
  // ``status-bar-sources-trigger`` is the actual testid (the testid
  // ``status-bar-sources-button`` doesn't exist — verified against
  // StatusBar.vue:547). Click the Profile tab so the dropdown renders.
  await page.getByTestId('status-bar-sources-trigger').click()
  await page.getByTestId(SOURCES_DIALOG).waitFor({ state: 'visible' })
  await page.getByTestId('sources-dialog-tab-profile').click()
}

test.describe('Operator profile dropdown — filter + default preselect (#692)', () => {
  test.describe.configure({ mode: 'serial' })

  test('preprod: dropdown shows only cloud_thin + preselected on fresh corpus', async ({
    page,
  }) => {
    await stubMinimalCorpusApis(page)
    await stubOperatorConfig(page, {
      content: '', // fresh corpus, no profile saved yet
      available_profiles: ['cloud_thin'],
      default_profile: 'cloud_thin',
    })

    await openProfileTab(page)

    const select = page.getByTestId(PROFILE_SELECT)
    await expect(select).toBeVisible()

    // Default-profile preselection — value should equal cloud_thin
    // even though no profile is saved on disk.
    await expect(select).toHaveValue('cloud_thin')

    // Dropdown options: "None" + "cloud_thin" only. No legacy / custom.
    const optionTexts = (await select.locator('option').allTextContents()).map((t) => t.trim())
    expect(optionTexts).toEqual(['None', 'cloud_thin'])
  })

  test('preprod with saved profile in allowlist: saved wins over default', async ({
    page,
  }) => {
    await stubMinimalCorpusApis(page)
    // Allowlist has both; default is cloud_thin; on-disk is cloud_balanced.
    // On-disk MUST win.
    await stubOperatorConfig(page, {
      content: 'profile: cloud_balanced\nmax_episodes: 1\n',
      available_profiles: ['cloud_balanced', 'cloud_thin'],
      default_profile: 'cloud_thin',
    })

    await openProfileTab(page)

    const select = page.getByTestId(PROFILE_SELECT)
    await expect(select).toHaveValue('cloud_balanced')
    const optionTexts = (await select.locator('option').allTextContents()).map((t) => t.trim())
    expect(optionTexts).toEqual(['None', 'cloud_balanced', 'cloud_thin'])
  })

  test('legacy yaml: saved profile outside allowlist renders as "(custom)"', async ({
    page,
  }) => {
    await stubMinimalCorpusApis(page)
    // Saved profile (``hybrid``) is NOT in available_profiles. The viewer
    // must still render it as a "(custom)" option so the operator sees
    // what's currently in their yaml — they can keep it, or pick a
    // valid one. Defense-in-depth on the api side rejects job submission
    // until they pick a valid profile.
    await stubOperatorConfig(page, {
      content: 'profile: hybrid\nmax_episodes: 1\n',
      available_profiles: ['cloud_thin'],
      default_profile: 'cloud_thin',
    })

    await openProfileTab(page)

    const select = page.getByTestId(PROFILE_SELECT)
    await expect(select).toHaveValue('hybrid')
    const optionTexts = (await select.locator('option').allTextContents()).map((t) => t.trim())
    expect(optionTexts).toEqual(['None', 'hybrid (custom)', 'cloud_thin'])
  })

  test('dev / CI default: no allowlist + no default → all profiles, "None" preselect', async ({
    page,
  }) => {
    await stubMinimalCorpusApis(page)
    await stubOperatorConfig(page, {
      content: '',
      available_profiles: ['airgapped_thin', 'cloud_balanced', 'cloud_thin'],
      default_profile: null,
    })

    await openProfileTab(page)

    const select = page.getByTestId(PROFILE_SELECT)
    // No default_profile + no on-disk profile → empty value = "None".
    await expect(select).toHaveValue('')
    const optionTexts = (await select.locator('option').allTextContents()).map((t) => t.trim())
    expect(optionTexts).toEqual(['None', 'airgapped_thin', 'cloud_balanced', 'cloud_thin'])
  })

  test('preprod with empty allowlist: dropdown shows only "None"', async ({ page }) => {
    await stubMinimalCorpusApis(page)
    // Misconfig: env says allowlist=[some-typo], no on-disk match.
    // Server returns available_profiles=[] + default_profile=null.
    // Dropdown should still render the "None" sentinel — the operator
    // can save with no profile (CLI's _resolve_profile fallback path).
    await stubOperatorConfig(page, {
      content: '',
      available_profiles: [],
      default_profile: null,
    })

    await openProfileTab(page)

    const select = page.getByTestId(PROFILE_SELECT)
    await expect(select).toHaveValue('')
    const optionTexts = (await select.locator('option').allTextContents()).map((t) => t.trim())
    expect(optionTexts).toEqual(['None'])
  })
})
