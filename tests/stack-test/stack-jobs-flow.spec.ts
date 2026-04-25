import { expect, test } from '@playwright/test'

const CORPUS = '/app/output'

/** Must match the sole URL in ``config/ci/stack-test-seed/feeds.spec.yaml`` (keep in sync). */
const STACK_TEST_SEED_FIRST_FEED_URL = 'http://mock-feeds/p01_fast_with_transcript.xml'

/**
 * Feed added in the UI — must be a **different** URL from
 * ``STACK_TEST_SEED_FIRST_FEED_URL`` so we truly append a second source
 * (``p01_episode_selection``: transcript-only items, no MP3).
 */
const MOCK_FEED_EXTRA = 'http://mock-feeds/p01_episode_selection.xml'

expect(
  MOCK_FEED_EXTRA,
  'UI feed must differ from seeded feed (feeds.spec.yaml)',
).not.toBe(STACK_TEST_SEED_FIRST_FEED_URL)

/** ``p01_fast_with_transcript`` (1 item) + ``p01_episode_selection`` (3 items) — see fixtures under ``tests/fixtures/rss/``. */
const STACK_TEST_EXPECTED_EPISODE_TOTAL = 4

/** Substrings that must appear somewhere in the Library episode list after a successful job (fixture titles). */
const STACK_TEST_LIBRARY_TITLE_MARKERS = ['Building Trails', 'E2E Selection']

/**
 * Where feeds are saved (stack test):
 *
 * 1. **UI** — Footer **Configuration** opens ``<dialog>``
 *    (``status-bar-sources-dialog``). **Feeds** tab → URL field
 *    ``sources-dialog-feeds-add-url`` + **Add feed**
 *    ``sources-dialog-feeds-add-btn`` calls ``addFeedFromInput()`` in
 *    ``web/gi-kg-viewer/src/components/shell/StatusBar.vue``, which
 *    appends to the in-memory list then ``await persistFeedsFromCrud()``
 *    → ``putFeeds()`` in ``web/gi-kg-viewer/src/api/feedsApi.ts``.
 *
 * 2. **API** — ``PUT /api/feeds?path=<corpus>`` with body
 *    ``{ feeds: [..] }`` (see
 *    ``src/podcast_scraper/server/routes/feeds.py`` → ``put_feeds``).
 *
 * 3. **Disk** — Same handler writes
 *    ``<corpus_root>/feeds.spec.yaml`` (basename from
 *    ``FEEDS_SPEC_DEFAULT_BASENAME``) via ``atomic_write_text``. For
 *    stack test, corpus root is ``/app/output`` inside containers
 *    (named volume). Seed lists ``STACK_TEST_SEED_FIRST_FEED_URL``;
 *    Playwright adds ``MOCK_FEED_EXTRA`` (asserted distinct).
 *
 * **Job profile (same dialog)** — **Job Profile** tab +
 * ``sources-dialog-profile-select`` + **Save (applies preset +
 * overrides on disk)** (``sources-dialog-save-profile``) →
 * ``PUT /api/operator-config`` so ``<corpus>/viewer_operator.yaml``
 * gains merged ``profile:`` + overrides before **Run pipeline job**.
 * Preset: ``STACK_TEST_OPERATOR_PROFILE`` or default
 * ``airgapped_thin`` when the env var is unset.
 *
 * **Configuration** (feeds + optional preset list + profile UI)
 * before job; **corpus output** only after the job reaches
 * ``succeeded``. Post-job checks (see ``validateCorpusDataLoadedAfterJob``)
 * run in order: API invariants → Digest → Library → Search → Graph.
 * Mock feeds are fixed (1 + 3 episodes) so totals and title substrings
 * are asserted against known fixture copy.
 *
 * **Operator profile:** ``STACK_TEST_OPERATOR_PROFILE`` overrides the
 * packaged preset name (e.g. ``cloud_thin`` or ``airgapped``). When
 * unset, the spec selects ``airgapped_thin`` and saves so Docker jobs
 * have a known ``profile:`` on disk (see
 * ``effective_pipeline_install_extras_for_docker``).
 */
function stackTestOperatorProfile(): string {
  return (process.env.STACK_TEST_OPERATOR_PROFILE ?? '').trim() || 'airgapped_thin'
}

/**
 * Whether the active profile builds a local FAISS / sentence-transformers
 * vector index during the pipeline run. ``cloud_thin`` opts out via
 * ``vector_search: false`` — the API correctly returns
 * ``error: "no_index"`` and the spec asserts that gracefully instead of
 * the populated-index path. Profiles known to keep ``vector_search: true``
 * (the default for the airgapped + cloud_balanced/quality presets) take
 * the populated-index path.
 */
const STACK_TEST_NO_VECTOR_PROFILES: ReadonlySet<string> = new Set(['cloud_thin'])

function stackTestProfileBuildsVectorIndex(): boolean {
  return !STACK_TEST_NO_VECTOR_PROFILES.has(stackTestOperatorProfile())
}

/** Heartbeat so CI / local logs are not silent during long polls. */
function stackTestProgress(msg: string): void {
  // eslint-disable-next-line no-console
  console.log(`[stack-test] ${new Date().toISOString()} ${msg}`)
}

/** First-run gesture card can sit above ``.graph-canvas``; mirror ``gi-kg-viewer/e2e/helpers.ts``. */
async function dismissGraphGestureOverlayIfPresent(
  page: import('@playwright/test').Page,
): Promise<void> {
  const btn = page.getByTestId('graph-gesture-overlay-dismiss')
  try {
    await btn.waitFor({ state: 'visible', timeout: 3000 })
  } catch {
    return
  }
  await btn.click()
}

function feedUrlsFromApiList(feeds: unknown[] | undefined): string[] {
  if (!Array.isArray(feeds)) {
    return []
  }
  return feeds.map((e) => {
    if (typeof e === 'string') {
      return e.trim()
    }
    if (e && typeof e === 'object') {
      const o = e as Record<string, unknown>
      const u = o.url ?? o.rss
      if (typeof u === 'string') {
        return u.trim()
      }
    }
    return ''
  })
}

async function assertFeedsPersistedOnCorpus(
  request: import('@playwright/test').APIRequestContext,
  corpusPath: string,
  mustInclude: string[],
): Promise<void> {
  const res = await request.get('/api/feeds', { params: { path: corpusPath } })
  if (!res.ok()) {
    const t = await res.text()
    throw new Error(`GET /api/feeds failed ${res.status()}: ${t.slice(0, 500)}`)
  }
  const body = (await res.json()) as { file_relpath?: string; feeds?: unknown[] }
  expect(body.file_relpath, 'API should report canonical feeds spec basename').toBe(
    'feeds.spec.yaml',
  )
  const urls = feedUrlsFromApiList(body.feeds)
  for (const fragment of mustInclude) {
    expect(
      urls.some((u) => u === fragment || u.includes(fragment)),
      `expected a feed URL matching "${fragment}"; got: ${urls.join(' | ')}`,
    ).toBeTruthy()
  }
}

/** Packaged presets are discoverable (API image includes ``config/profiles``). */
async function assertOperatorPresetsInclude(
  request: import('@playwright/test').APIRequestContext,
  corpusPath: string,
  profileStem: string,
): Promise<void> {
  const res = await request.get('/api/operator-config', { params: { path: corpusPath } })
  if (!res.ok()) {
    const t = await res.text()
    throw new Error(`GET /api/operator-config failed ${res.status()}: ${t.slice(0, 400)}`)
  }
  const body = (await res.json()) as { available_profiles?: string[] }
  const avail = Array.isArray(body.available_profiles) ? body.available_profiles : []
  expect(
    avail.includes(profileStem),
    `preset "${profileStem}" must appear in available_profiles (${avail.join(', ')})`,
  ).toBeTruthy()
}

async function waitForHealthFlag(
  request: import('@playwright/test').APIRequestContext,
  key: 'jobs_api' | 'feeds_api',
  timeoutMs: number,
): Promise<void> {
  const deadline = Date.now() + timeoutMs
  let lastBeat = Date.now()
  stackTestProgress(`waiting for /api/health ${key} (timeout ${timeoutMs}ms)`)
  while (Date.now() < deadline) {
    const res = await request.get('/api/health')
    if (res.ok()) {
      const body = (await res.json()) as Record<string, unknown>
      if (body[key] === true) {
        stackTestProgress(`/api/health ${key}=true`)
        return
      }
    }
    if (Date.now() - lastBeat >= 15_000) {
      const left = Math.max(0, Math.round((deadline - Date.now()) / 1000))
      stackTestProgress(`still waiting for ${key} (~${left}s left)`)
      lastBeat = Date.now()
    }
    await new Promise((r) => setTimeout(r, 2000))
  }
  throw new Error(`timeout waiting for /api/health ${key}`)
}

type CorpusRunSummary = {
  feeds?: Array<{ episodes_processed?: number; feed_url?: string; ok?: boolean }>
}

async function fetchRunSummaryProcessedTotal(
  request: import('@playwright/test').APIRequestContext,
  corpusPath: string,
): Promise<number> {
  const res = await request.get('/api/corpus/documents/run-summary', {
    params: { path: corpusPath },
  })
  if (!res.ok()) {
    const t = await res.text()
    throw new Error(`GET run-summary failed ${res.status()}: ${t.slice(0, 400)}`)
  }
  const summary = (await res.json()) as CorpusRunSummary
  const feeds = Array.isArray(summary.feeds) ? summary.feeds : []
  return feeds.reduce(
    (acc, f) => acc + (typeof f.episodes_processed === 'number' ? f.episodes_processed : 0),
    0,
  )
}

async function countCorpusEpisodesCatalog(
  request: import('@playwright/test').APIRequestContext,
  corpusPath: string,
): Promise<number> {
  let total = 0
  let cursor: string | null = null
  for (let pageIdx = 0; pageIdx < 20; pageIdx += 1) {
    const q = new URLSearchParams({ path: corpusPath, limit: '200' })
    if (cursor) {
      q.set('cursor', cursor)
    }
    const res = await request.get(`/api/corpus/episodes?${q.toString()}`)
    if (!res.ok()) {
      const t = await res.text()
      throw new Error(`GET /api/corpus/episodes failed ${res.status()}: ${t.slice(0, 400)}`)
    }
    const body = (await res.json()) as { items?: unknown[]; next_cursor?: string | null }
    const items = Array.isArray(body.items) ? body.items : []
    total += items.length
    const next = body.next_cursor
    if (typeof next === 'string' && next.trim() !== '') {
      cursor = next
      continue
    }
    break
  }
  return total
}

async function fetchDigestRecentRowCount(
  request: import('@playwright/test').APIRequestContext,
  corpusPath: string,
): Promise<number> {
  const q = new URLSearchParams({ path: corpusPath, window: 'all' })
  const res = await request.get(`/api/corpus/digest?${q.toString()}`)
  if (!res.ok()) {
    const t = await res.text()
    throw new Error(`GET /api/corpus/digest failed ${res.status()}: ${t.slice(0, 400)}`)
  }
  const body = (await res.json()) as { rows?: unknown[] }
  return Array.isArray(body.rows) ? body.rows.length : 0
}

async function waitForSearchHits(
  request: import('@playwright/test').APIRequestContext,
  timeoutMs: number,
): Promise<void> {
  const queries = ['berm', 'trails', 'mountain']
  const deadline = Date.now() + timeoutMs
  let lastBeat = Date.now()
  stackTestProgress(`waiting for /api/search hits (${queries.join(', ')})`)
  while (Date.now() < deadline) {
    for (const q of queries) {
      const params = new URLSearchParams({ q, path: CORPUS, top_k: '10' })
      const res = await request.get(`/api/search?${params.toString()}`)
      if (res.ok()) {
        const body = (await res.json()) as { error?: string | null; results?: unknown[] }
        if (!body.error && Array.isArray(body.results) && body.results.length > 0) {
          stackTestProgress(`/api/search returned results for query "${q}"`)
          return
        }
      }
    }
    if (Date.now() - lastBeat >= 20_000) {
      const left = Math.max(0, Math.round((deadline - Date.now()) / 1000))
      stackTestProgress(`still polling /api/search (~${left}s left)`)
      lastBeat = Date.now()
    }
    await new Promise((r) => setTimeout(r, 4000))
  }
  throw new Error(`timeout waiting for /api/search results (tried ${queries.join(', ')})`)
}

/**
 * Left rail: leave **Explore** if active, then ensure **Semantic
 * search** is visible.
 *
 * **Do not** click ``left-rail-edge-toggle`` when the rail is already
 * expanded: that **collapses** the panel. After Explore→Search,
 * ``#search-q`` can briefly sit in the translated slide — wait first;
 * only expand when ``aria-expanded`` is not ``true`` (collapsed rail).
 */
async function prepareSemanticSearchUi(
  page: import('@playwright/test').Page,
): Promise<void> {
  const backSearch = page.getByTestId('left-panel-back-search')
  if (await backSearch.isVisible().catch(() => false)) {
    stackTestProgress('post-job: left panel was on Explore — Back to Semantic search')
    await backSearch.click()
  }
  const q = page.locator('#search-q')
  const railToggle = page.getByTestId('left-rail-edge-toggle')
  try {
    await expect(q).toBeVisible({ timeout: 15_000 })
  } catch {
    const expanded = await railToggle.getAttribute('aria-expanded')
    if (expanded !== 'true') {
      stackTestProgress('post-job: expand collapsed left rail for search')
      await railToggle.click()
    } else {
      // Explore→Search slide uses ~300ms transition (``LeftPanel``);
      // avoid toggling the rail (that collapses it).
      await page.waitForTimeout(450)
    }
    await expect(q).toBeVisible({ timeout: 25_000 })
  }
  await page.locator('#search-since-date').fill('')
  await q.fill('')
}

/** Corpus / index / UI checks that require a **finished** pipeline job — run only after ``succeeded``. */
async function validateCorpusDataLoadedAfterJob(
  page: import('@playwright/test').Page,
  request: import('@playwright/test').APIRequestContext,
): Promise<void> {
  stackTestProgress('post-job: API — run-summary episodes_processed sum vs fixture total')
  const runSummaryTotal = await fetchRunSummaryProcessedTotal(request, CORPUS)
  expect(
    runSummaryTotal,
    `run-summary feeds episodes_processed should sum to ${STACK_TEST_EXPECTED_EPISODE_TOTAL} (mock feeds)`,
  ).toBe(STACK_TEST_EXPECTED_EPISODE_TOTAL)

  stackTestProgress('post-job: API — corpus episode catalog count')
  const catalogTotal = await countCorpusEpisodesCatalog(request, CORPUS)
  expect(
    catalogTotal,
    `GET /api/corpus/episodes catalog size should match ${STACK_TEST_EXPECTED_EPISODE_TOTAL}`,
  ).toBe(STACK_TEST_EXPECTED_EPISODE_TOTAL)

  stackTestProgress(
    'post-job: Digest tab — align window to **All time** (ignore persisted lens), then recent rows + rail',
  )
  await page.getByTestId('main-tab-digest').click()
  await expect(page.getByTestId('digest-root')).toBeVisible({ timeout: 60_000 })
  await page
    .getByTestId('digest-toolbar-filters')
    .getByRole('button', { name: 'All time' })
    .click()
  const digestRows = page.locator('[data-digest-recent-row]')
  await expect(digestRows.first()).toBeVisible({ timeout: 120_000 })
  const digestApiRows = await fetchDigestRecentRowCount(request, CORPUS)
  expect(digestApiRows, 'digest API should list at least one recent row after job').toBeGreaterThan(
    0,
  )
  stackTestProgress(
    `post-job: digest API recent rows=${digestApiRows} (diversified sample; may be < ${STACK_TEST_EXPECTED_EPISODE_TOTAL})`,
  )
  await expect(digestRows).toHaveCount(digestApiRows)
  await digestRows.first().click()
  await expect(page.getByTestId('episode-detail-rail')).toBeVisible({ timeout: 60_000 })
  // Details/Neighbourhood tabs only mount on Graph tab (``SubjectRail``);
  // body is always shown for episodes.
  await expect(page.getByTestId('episode-detail-rail-body')).toBeVisible({ timeout: 60_000 })

  stackTestProgress('post-job: Library tab — episode row count + known titles + per-row rail')
  await page.getByTestId('main-tab-library').click()
  const libRoot = page.getByTestId('library-root')
  await expect(libRoot).toBeVisible({ timeout: 60_000 })
  // ``library-feed-list-scroll`` is the feed sidebar only; episode
  // titles live in ``[data-library-episode-row]`` below.
  await expect(page.getByTestId('library-feed-list-scroll')).toBeVisible({ timeout: 120_000 })
  for (const marker of STACK_TEST_LIBRARY_TITLE_MARKERS) {
    await expect(libRoot).toContainText(marker, { timeout: 60_000 })
  }
  const libRows = page.locator('[data-library-episode-row]')
  await expect(libRows).toHaveCount(STACK_TEST_EXPECTED_EPISODE_TOTAL, { timeout: 120_000 })
  for (let i = 0; i < STACK_TEST_EXPECTED_EPISODE_TOTAL; i += 1) {
    stackTestProgress(
      `post-job: Library row ${i + 1}/${STACK_TEST_EXPECTED_EPISODE_TOTAL} → episode rail`,
    )
    await libRows.nth(i).click()
    await expect(page.getByTestId('episode-detail-rail')).toBeVisible({ timeout: 60_000 })
    await expect(page.getByTestId('episode-detail-rail-body')).toBeVisible({ timeout: 60_000 })
  }

  // Semantic search asserts only when the active profile builds a local
  // vector index (airgapped_thin etc.). cloud_thin opts out via
  // ``vector_search: false`` — the API correctly returns
  // ``error: "no_index"`` and the UI shows a "no index" surface; both
  // are valid stack-test outcomes for that profile.
  const hasVector = stackTestProfileBuildsVectorIndex()
  if (hasVector) {
    stackTestProgress('post-job: semantic search UI (left panel) + results list')
    await prepareSemanticSearchUi(page)
    // Short query matches ``waitForSearchHits`` probes; long phrases can
    // return 0 rows on tiny FAISS corpora.
    const searchUiQuery = 'trails'
    const searchRespPromise = page.waitForResponse(
      (r) => {
        if (r.request().method() !== 'GET') {
          return false
        }
        let u: URL
        try {
          u = new URL(r.url())
        } catch {
          return false
        }
        if (!u.pathname.endsWith('/api/search')) {
          return false
        }
        const qv = u.searchParams.get('q') ?? ''
        return qv === searchUiQuery || decodeURIComponent(qv) === searchUiQuery
      },
      { timeout: 120_000 },
    )
    await page.locator('#search-q').fill(searchUiQuery)
    // Submit lives **outside** ``<form id="semantic-search-form">``
    // (``form="semantic-search-form"``), not inside it.
    await page.locator('button[type="submit"][form="semantic-search-form"]').click()
    const searchResp = await searchRespPromise
    expect(searchResp.ok(), `GET /api/search UI response status ${searchResp.status()}`).toBeTruthy()
    const searchJson = (await searchResp.json()) as { error?: string | null; results?: unknown[] }
    expect(
      searchJson.error,
      `UI-driven GET /api/search should not set error (query=${JSON.stringify(searchUiQuery)})`,
    ).toBeFalsy()
    expect(
      Array.isArray(searchJson.results) && searchJson.results.length > 0,
      `UI search should return hits for ${JSON.stringify(searchUiQuery)} (got ${
        Array.isArray(searchJson.results) ? searchJson.results.length : 'non-array'
      })`,
    ).toBeTruthy()
    const resultsHost = page.getByTestId('semantic-search-results-scroll')
    await expect(resultsHost.getByText(/\d+\s+results?/i).first()).toBeVisible({ timeout: 30_000 })
  } else {
    stackTestProgress(
      `post-job: semantic search skipped — profile ${stackTestOperatorProfile()} has vector_search disabled (enriched_search_available=false)`,
    )
    // Sanity: API still answers /api/search with a structured no_index
    // response (200 + ``error: "no_index"``), not a 5xx.
    const res = await request.get('/api/search', { params: { q: 'trails', path: CORPUS } })
    expect(res.ok(), `GET /api/search status ${res.status()} (no_index variant)`).toBeTruthy()
    const body = (await res.json()) as { error?: string | null }
    expect(
      body.error,
      'GET /api/search without index should report error=no_index, not crash',
    ).toBe('no_index')
  }

  if (hasVector) {
    stackTestProgress('post-job: GET /api/search (FAISS + embeddings)')
    await waitForSearchHits(request, 120_000)
  }

  stackTestProgress('post-job: Graph tab — graph canvas (last: heaviest / gesture overlay)')
  await page.getByTestId('main-tab-graph').click()
  await dismissGraphGestureOverlayIfPresent(page)
  await expect(page.locator('.graph-canvas')).toBeVisible({ timeout: 120_000 })
}

type JobRow = {
  job_id?: string
  status?: string
  command_type?: string
  queue_position?: number | null
  created_at?: string
  started_at?: string | null
  error_reason?: string | null
}

const JOB_IN_PROGRESS = new Set(['queued', 'running'])
const JOB_TERMINAL_OK = new Set(['succeeded'])
const JOB_TERMINAL_BAD = new Set(['failed', 'cancelled', 'stale'])

function formatJobSnapshot(j: JobRow, jobsTotal: number): string {
  const id = typeof j.job_id === 'string' ? j.job_id.slice(0, 8) : '?'
  const st = j.status ?? 'unknown'
  const qp = j.queue_position != null ? ` queue=${j.queue_position}` : ''
  const cmd = j.command_type ? ` cmd=${j.command_type}` : ''
  const err = j.error_reason ? ` err=${String(j.error_reason).slice(0, 120)}` : ''
  return `newest job ${id}… status=${st}${qp}${cmd} (jobs=${jobsTotal})${err}`
}

/** Newest row by ``created_at`` (ISO string); avoids relying on API array order. */
function newestJobByCreatedAt(jobs: JobRow[]): JobRow | undefined {
  if (jobs.length === 0) {
    return undefined
  }
  return [...jobs].sort((a, b) => {
    const ca = typeof a.created_at === 'string' ? a.created_at : ''
    const cb = typeof b.created_at === 'string' ? b.created_at : ''
    return ca.localeCompare(cb)
  })[jobs.length - 1]
}

/** Poll ``GET /api/jobs`` until the newest job reaches ``succeeded``; validate status each loop. */
async function waitForLatestJobSucceeded(
  request: import('@playwright/test').APIRequestContext,
  timeoutMs: number,
): Promise<void> {
  const deadline = Date.now() + timeoutMs
  const q = new URLSearchParams({ path: CORPUS })
  const pollMs = Math.max(
    1000,
    Number.parseInt(process.env.STACK_TEST_JOB_POLL_MS ?? '4000', 10) || 4000,
  )
  let lastBeat = Date.now()
  let lastLoggedSignature = ''
  stackTestProgress(
    `poll GET /api/jobs until newest job succeeded (every ${pollMs}ms, timeout ${timeoutMs}ms); docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.stack-test.yml logs -f`,
  )
  while (Date.now() < deadline) {
    const res = await request.get(`/api/jobs?${q.toString()}`)
    expect(res.ok()).toBeTruthy()
    const data = (await res.json()) as { jobs?: JobRow[] }
    const jobs = Array.isArray(data.jobs) ? data.jobs : []
    const left = Math.max(0, Math.round((deadline - Date.now()) / 1000))
    if (jobs.length === 0) {
      if (Date.now() - lastBeat >= 15_000) {
        stackTestProgress(`job poll: no jobs yet under ${CORPUS}; ~${left}s left`)
        lastBeat = Date.now()
      }
      await new Promise((r) => setTimeout(r, pollMs))
      continue
    }
    const last = newestJobByCreatedAt(jobs) as JobRow
    const st = last?.status ?? 'unknown'
    const sig = `${st}:${last?.job_id ?? ''}:${last?.queue_position ?? ''}`
    if (JOB_TERMINAL_BAD.has(st)) {
      stackTestProgress(`job poll: terminal failure — ${formatJobSnapshot(last, jobs.length)}`)
      throw new Error(
        `job ended with status ${st}${last?.error_reason ? `: ${last.error_reason}` : ''}`,
      )
    }
    if (JOB_TERMINAL_OK.has(st)) {
      stackTestProgress(`job poll: ${formatJobSnapshot(last, jobs.length)}`)
      return
    }
    if (!JOB_IN_PROGRESS.has(st)) {
      throw new Error(
        `unexpected job status "${st}" (expected queued|running|succeeded|failed|cancelled|stale): ${formatJobSnapshot(
          last,
          jobs.length,
        )}`,
      )
    }
    if (sig !== lastLoggedSignature) {
      stackTestProgress(`job poll: ${formatJobSnapshot(last, jobs.length)} ~${left}s budget left`)
      lastLoggedSignature = sig
      lastBeat = Date.now()
    } else if (Date.now() - lastBeat >= 15_000) {
      stackTestProgress(`job poll: still ${formatJobSnapshot(last, jobs.length)} ~${left}s left`)
      lastBeat = Date.now()
    }
    await new Promise((r) => setTimeout(r, pollMs))
  }
  throw new Error('timeout waiting for job succeeded')
}

test.describe('stack test — feeds UI + job + data', () => {
  test('configure mock feeds, run pipeline job, wait, evaluate', async ({ page, request }) => {
    // The global config timeout (120s) is for fast smoke specs. This test
    // drives a real pipeline job end-to-end (download → transcribe → GI/KG)
    // for 4 episodes; ``waitForLatestJobSucceeded`` allows 800s (13 min) for
    // the poll. Match that with margin for UI navigation + post-job asserts.
    test.setTimeout(15 * 60 * 1000)

    const pageErrors: string[] = []
    page.on('pageerror', (err) => {
      pageErrors.push(err.message)
    })

    stackTestProgress('browser: goto /')
    await page.goto('/')
    await waitForHealthFlag(request, 'feeds_api', 120_000)
    await waitForHealthFlag(request, 'jobs_api', 120_000)

    stackTestProgress('browser: clear corpus localStorage + reload')
    await page.evaluate(() => {
      try {
        localStorage.removeItem('ps_corpus_path')
      } catch {
        /* ignore */
      }
    })
    await page.reload()
    await waitForHealthFlag(request, 'feeds_api', 60_000)

    stackTestProgress(`browser: set corpus path to ${CORPUS}`)
    await page.getByTestId('status-bar-corpus-path').fill(CORPUS)
    await page.getByTestId('status-bar-corpus-path').press('Tab')
    await page.waitForTimeout(500)

    stackTestProgress('browser: open Sources → Feeds')
    await page.getByTestId('status-bar-sources-trigger').click()
    await expect(page.getByTestId('status-bar-sources-dialog')).toBeVisible()
    await page.getByTestId('sources-dialog-tab-feeds').click()

    stackTestProgress(`browser: add feed URL ${MOCK_FEED_EXTRA}`)
    await page.getByTestId('sources-dialog-feeds-add-url').fill(MOCK_FEED_EXTRA)
    await page.getByTestId('sources-dialog-feeds-add-btn').click()

    stackTestProgress('browser: expect second feed row (p01_episode_selection)')
    await expect(page.getByTestId('sources-dialog-feeds-row-1')).toContainText(
      'p01_episode_selection',
    )
    // addFeedFromInput() updates the list then awaits PUT /api/feeds
    // (sourcesBusy disables Add). Closing while persist is in flight
    // can leave Playwright waiting on the native <dialog> Close
    // (Firefox): wait for re-enabled Add first, then close with
    // force + Escape fallback.
    stackTestProgress('browser: wait for feeds PUT to finish (Add button re-enabled)')
    await expect(page.getByTestId('sources-dialog-feeds-add-btn')).toBeEnabled({
      timeout: 120_000,
    })

    stackTestProgress('validate GET /api/feeds (on-disk feeds.spec.yaml) before Pipeline')
    await assertFeedsPersistedOnCorpus(request, CORPUS, [
      'p01_fast_with_transcript',
      'p01_episode_selection',
    ])

    const operatorProfile = stackTestOperatorProfile()
    stackTestProgress(
      `validate GET /api/operator-config available_profiles includes "${operatorProfile}"`,
    )
    await assertOperatorPresetsInclude(request, CORPUS, operatorProfile)

    stackTestProgress(
      `browser: Job Profile tab — select "${operatorProfile}" then Save (PUT /api/operator-config)`,
    )
    await page.getByTestId('sources-dialog-tab-profile').click()
    const profileSelect = page.getByTestId('sources-dialog-profile-select')
    await expect(profileSelect).toBeVisible({ timeout: 30_000 })
    await profileSelect
      .locator(`option[value="${operatorProfile}"]`)
      .first()
      .waitFor({ state: 'attached', timeout: 120_000 })
    await profileSelect.selectOption(operatorProfile, { timeout: 30_000 })

    const operatorPutPromise = page.waitForResponse(
      (r) => {
        if (r.request().method() !== 'PUT') {
          return false
        }
        try {
          const u = new URL(r.url())
          return u.pathname.endsWith('/api/operator-config')
        } catch {
          return false
        }
      },
      { timeout: 120_000 },
    )
    await page.getByTestId('sources-dialog-save-profile').click()
    const operatorPut = await operatorPutPromise
    expect(
      operatorPut.ok(),
      `PUT /api/operator-config status ${operatorPut.status()}`,
    ).toBeTruthy()

    stackTestProgress('browser: close sources dialog')
    const sourcesDlg = page.getByTestId('status-bar-sources-dialog')
    await page.getByTestId('sources-dialog-close').click({ force: true, timeout: 30_000 })
    try {
      await expect(sourcesDlg).toBeHidden({ timeout: 5_000 })
    } catch {
      stackTestProgress('browser: dialog still visible after Close — sending Escape')
      await page.keyboard.press('Escape')
      await expect(sourcesDlg).toBeHidden({ timeout: 10_000 })
    }

    stackTestProgress('browser: Dashboard → Pipeline tab (pipeline-jobs-card up to 60s)')
    await page.getByTestId('main-tab-dashboard').click()
    await page.getByTestId('dashboard-tab-pipeline').click()
    await expect(page.getByTestId('pipeline-jobs-card')).toBeVisible({ timeout: 60_000 })

    stackTestProgress('browser: click Run pipeline job')
    await page.getByTestId('pipeline-jobs-run').click()

    stackTestProgress('polling /api/jobs until newest job succeeded (long step)')
    await waitForLatestJobSucceeded(request, 800_000)

    stackTestProgress('browser: reload after job so Graph/search pick up new corpus artifacts')
    await page.reload()
    await waitForHealthFlag(request, 'feeds_api', 60_000)
    await waitForHealthFlag(request, 'jobs_api', 60_000)
    await page.getByTestId('status-bar-corpus-path').fill(CORPUS)
    await page.getByTestId('status-bar-corpus-path').press('Tab')
    await page.waitForTimeout(500)

    await validateCorpusDataLoadedAfterJob(page, request)

    stackTestProgress('done (no unexpected page errors)')
    expect(pageErrors, `unexpected page errors:\n${pageErrors.join('\n')}`).toEqual([])
  })
})
