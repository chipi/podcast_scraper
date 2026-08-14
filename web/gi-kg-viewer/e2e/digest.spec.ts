import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  liveDigestRows,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
  type LiveDigestRow,
} from './helpers'

/**
 * #1619 — the Digest splits cleanly in two, and the split is the finding.
 *
 * `GET /api/corpus/digest` returns **`rows`** (catalogue data, populated for the v3 corpus — 36
 * episodes with real `cil_digest_topics`) and **`topics`** (the retrieval-grounded topic bands).
 * The bands are search output: `corpus_digest.py` runs each configured digest query through
 * `run_corpus_search`. With the index live and working, `topics` is **still `[]`** — the corpus
 * answers none of `DEFAULT_DIGEST_TOPICS` within the window, and it fails silently
 * (`topics_unavailable_reason: null`, which reads as "nothing configured").
 *
 * So everything about rows runs live here. Everything about bands is in the second describe,
 * still stubbed, recorded in docs/wip/CORPUS-V4-FIXTURE-LADDER.md §B.
 */

/** Land on the Digest against the live corpus. */
async function openDigest(page: Page): Promise<LiveDigestRow[]> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await expect(page.getByTestId('digest-root')).toBeVisible({ timeout: 30_000 })
  return liveDigestRows(page)
}

/** The accessible name of a Digest Recent card: `"<title>, <feed>"`. */
function digestCardName(row: LiveDigestRow): string {
  return `${row.episode_title}, ${row.feed_display_title}`
}

test.describe('Corpus Digest tab', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('digest episode cards omit graph/search actions (Episode subject rail has them)', async ({
    page,
  }) => {
    await openDigest(page)
    const digestRoot = page.getByTestId('digest-root')
    await expect(digestRoot.getByRole('button', { name: 'Open in graph' })).toHaveCount(0)
    await expect(
      digestRoot.getByRole('button', { name: 'Prefill semantic search' }),
    ).toHaveCount(0)
  })

  test('click digest Recent row opens Episode subject rail; stays on Digest', async ({ page }) => {
    const rows = await openDigest(page)
    const row = rows[0]!
    await page.getByRole('button', { name: digestCardName(row), exact: true }).click()
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: row.episode_title }),
    ).toBeVisible()
  })

  test('Digest ↔ Library keeps Episode subject rail when episode is in catalog', async ({
    page,
  }) => {
    const rows = await openDigest(page)
    const row = rows[0]!
    await page.getByRole('button', { name: digestCardName(row), exact: true }).click()
    const episodeRegion = page.getByRole('region', { name: 'Episode', exact: true })
    await expect(episodeRegion.getByRole('heading', { name: row.episode_title })).toBeVisible()
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await expect(episodeRegion.getByRole('heading', { name: row.episode_title })).toBeVisible()
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await expect(episodeRegion.getByRole('heading', { name: row.episode_title })).toBeVisible()
  })

  test('digest Recent CIL topic pills render from the row cil_digest_topics', async ({ page }) => {
    const rows = await openDigest(page)
    /* The pills render for topics that belong to a topic cluster. The v3 corpus supplies them on
     * every row, so pick the first row that has one rather than pinning a label. */
    const row = rows.find((r) => r.cil_digest_topics?.some((t) => t.in_topic_cluster))
    expect(row, 'expected at least one digest row with a clustered CIL topic').toBeTruthy()
    const topic = row!.cil_digest_topics.find((t) => t.in_topic_cluster)!

    await expect(page.getByTestId('digest-recent-cil-pills').first()).toBeVisible()
    await expect(
      page.getByRole('button', { name: `Open graph for topic: ${topic.label}` }).first(),
    ).toBeVisible()
  })

  test('digest Recent CIL topic pill opens the graph', async ({ page }) => {
    const rows = await openDigest(page)
    const row = rows.find((r) => r.cil_digest_topics?.some((t) => t.in_topic_cluster))!
    const topic = row.cil_digest_topics.find((t) => t.in_topic_cluster)!

    /* No artifact stub: the corpus's own GI artifacts are served by `/api/artifacts`, so this is
     * the real handoff — pill → graph load → canvas. */
    await page
      .getByRole('button', { name: `Open graph for topic: ${topic.label}` })
      .first()
      .click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
    await expect(page.locator('.graph-canvas')).toBeVisible()
  })
})

/**
 * #1619 category B — topic bands. Still stubbed, and it needs a v4 corpus, not a rewrite.
 *
 * `digest.topics` is retrieval-grounded: each configured digest query is run through the search
 * index and becomes a band. Verified 2026-08-14 with the index live and answering (848 vectors):
 * `GET /api/corpus/digest?include_topics=true` returns **`topics: []`** with
 * `topics_unavailable_reason: null`. `load_digest_topics()` falls back to `DEFAULT_DIGEST_TOPICS`,
 * so topics ARE configured — every band resolves to `None` because the fixture's content answers
 * none of the default editorial queries inside the window.
 *
 * Consequently the band title, the band's "Search topic" prefill, the show-more control, the
 * band-hit rows and the cross-show band all have no reachable state. The recency-dot test is here
 * for a different reason: it needs an episode published **today**, which a committed corpus with
 * fixed dates cannot be.
 *
 * v4 requirement: episodes that answer the default digest topic queries (or a corpus-local
 * `digest_topics` config), and at least one recent-dated episode.
 */
test.describe('Corpus Digest — topic bands (stubbed: corpus produces no bands)', () => {
  /** Local calendar `YYYY-MM-DD` for the recency-dot fixture (matches `digestRecency` parsing). */
  function localYmd(d = new Date()): string {
    const y = d.getFullYear()
    const m = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    return `${y}-${m}-${day}`
  }

  const ROW = {
    metadata_relative_path: 'metadata/ep1.metadata.json',
    feed_id: 'f1',
    feed_display_title: 'Mock Feed Show',
    episode_id: 'e1',
    episode_title: 'Digest Episode Alpha',
    summary_title: 'Digest summary',
    summary_bullets_preview: ['First bullet'],
    summary_bullet_graph_topic_ids: ['topic:first-bullet'],
    summary_preview: 'Digest summary — First bullet',
    gi_relative_path: 'metadata/ep1.gi.json',
    kg_relative_path: 'metadata/ep1.kg.json',
    has_gi: true,
    has_kg: false,
    cil_digest_topics: [],
  }

  const HIT = {
    metadata_relative_path: 'metadata/ep1.metadata.json',
    episode_title: 'Digest Episode Alpha',
    feed_id: 'f1',
    feed_display_title: 'Mock Feed Show',
    publish_date: '2024-06-05',
    score: 0.91,
    summary_preview: 'Digest summary — First bullet',
    episode_id: 'e1',
    gi_relative_path: 'metadata/ep1.gi.json',
    kg_relative_path: 'metadata/ep1.kg.json',
    has_gi: true,
    has_kg: false,
  }

  async function stubDigest(
    page: Page,
    opts: { rows?: Record<string, unknown>[]; topics?: Record<string, unknown>[] } = {},
  ): Promise<void> {
    await page.route('**/api/health**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok', corpus_library_api: true, corpus_digest_api: true }),
      }),
    )
    await page.route('**/api/artifacts?**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
      }),
    )
    await page.route('**/api/corpus/digest**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          window: 'all',
          window_start_utc: '1970-01-01T00:00:00Z',
          window_end_utc: '2024-06-08T00:00:00Z',
          compact: false,
          rows: opts.rows ?? [ROW],
          topics: opts.topics ?? [
            {
              topic_id: 't1',
              label: 'Mock Topic Band',
              query: 'climate science',
              graph_topic_id: 'topic:mock-topic-band',
              hits: [HIT],
            },
          ],
          topics_unavailable_reason: null,
        }),
      }),
    )
  }

  async function landOnStubbedDigest(page: Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()
  }

  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test('topic band renders and "Search topic" prefills the query', async ({ page }) => {
    await stubDigest(page)
    await landOnStubbedDigest(page)
    const bands = page.getByRole('region', { name: 'Topic bands' })
    await expect(bands.getByText('Mock Topic Band')).toBeVisible()
    await expect(bands.getByText('Strong match')).toBeVisible()
    await page.getByRole('button', { name: 'Search topic' }).first().click()
    await expect(page.locator('#search-q')).toHaveValue('climate science')
    // #671 — Since field replaced by DateChip; default chip label is "Since ▾".
    await expect(page.getByTestId('search-chip-since')).toContainText('Since ▾')
  })

  test('FR3.3: mapped topic band label opens the Topic Detail rail', async ({ page }) => {
    await stubDigest(page)
    await landOnStubbedDigest(page)
    const link = page.getByTestId('digest-band-topic-link')
    await expect(link).toBeVisible()
    await expect(link).toHaveText('Mock Topic Band')
    await link.click()
    await expect(page.getByTestId('topic-entity-view')).toBeVisible()
  })

  test('FR3.3: topic-band hit feed name scopes Library', async ({ page }) => {
    await stubDigest(page)
    await landOnStubbedDigest(page)
    const feedLink = page.getByTestId('digest-topic-hit-feed-link').first()
    await expect(feedLink).toBeVisible()
    await feedLink.click()
    await expect(page.getByTestId('library-root')).toBeVisible()
  })

  test('topic bands show-more expands the fourth band and hides the control', async ({ page }) => {
    const topics = ['Mock Topic Band', 'Second Band', 'Third Band', 'Fourth Band'].map(
      (label, i) => ({
        topic_id: `t${i + 1}`,
        label,
        query: `q${i + 1}`,
        graph_topic_id: `topic:${label.toLowerCase().replace(/\s+/g, '-')}`,
        hits: [HIT],
      }),
    )
    await stubDigest(page, { topics })
    await landOnStubbedDigest(page)
    const digestRoot = page.getByTestId('digest-root')
    const topicBands = digestRoot.getByRole('region', { name: 'Topic bands' })
    await expect(topicBands.getByText('Fourth Band')).toHaveCount(0)
    const showMore = digestRoot.getByTestId('digest-topic-bands-show-more')
    await expect(showMore).toBeVisible()
    await expect(showMore).toHaveText('Show 1 more topics')
    await showMore.click()
    await expect(showMore).toHaveCount(0)
    await expect(topicBands.getByText('Fourth Band')).toBeVisible()
  })

  test('FR3.2: cross-show band lists the top insight per show', async ({ page }) => {
    await stubDigest(page)
    await page.route('**/api/relational/cross-show**', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          subject: 'topic:mock-topic-band',
          groups: {
            'podcast:show-one': [
              { id: 'insight:1', type: 'insight', text: 'Show one position.', show_id: 'podcast:show-one', episode_id: 'e1' },
            ],
            'podcast:show-two': [
              { id: 'insight:2', type: 'insight', text: 'Show two position.', show_id: 'podcast:show-two', episode_id: 'e2' },
            ],
          },
          error: null,
        }),
      }),
    )
    await landOnStubbedDigest(page)
    await page.getByTestId('digest-cross-show-toggle').first().click()
    const band = page.getByTestId('digest-cross-show-band').first()
    await expect(band).toBeVisible()
    await expect(band.getByTestId('digest-cross-show-row')).toHaveCount(2)
    await expect(band).toContainText('Show one position.')
    await expect(band).toContainText('Show two position.')
  })

  /**
   * Separate reason from the bands: this needs an episode published **today**, and a committed
   * corpus has fixed publish dates. Time-dependent state, not a search or content gap.
   */
  test('digest Recent shows a recency dot when publish_date is today (local)', async ({ page }) => {
    await stubDigest(page, {
      rows: [{ ...ROW, publish_date: localYmd() }],
      topics: [],
    })
    await landOnStubbedDigest(page)
    await expect(
      page
        .getByTestId('digest-root')
        .getByRole('img', { name: /Published (less than 1 hour|\d+ hour)/ }),
    ).toBeVisible()
  })
})
