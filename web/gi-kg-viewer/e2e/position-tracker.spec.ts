import { expect, test, type Page, type Route } from '@playwright/test'
import { mainViewsNav, mockSignIn, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

/**
 * UXS-009 Position Tracker — a person's stated positions on ONE topic, as a timeline read oldest
 * first, narrowed by insight-type chips, with an honest empty state.
 *
 * Mocked, for two measured reasons (probed on the live v3 corpus 2026-10-05):
 *  * the only shipped entry to a person outside the graph is a search hit's `lifted.speaker` link,
 *    and no live result carries `lifted` (see `person-landing.spec.ts` header);
 *  * the live CIL layer is empty for every person tried — `/api/persons/maya/brief` returns
 *    `topics: {}` and `/api/persons/maya/positions?topic=…` returns `episodes: []` for all three of
 *    her top topics — so there is no arc to draw.
 *
 * Entry is the same lifted-speaker path `person-landing.spec.ts` uses. Topics are picked from the
 * Positions tab's "By topic" lens (`person-landing-insights-voiced-topic-button`) — the reachable
 * picker. `person-landing-ranked-topic-button` renders only in PersonLandingView's `view="full"`,
 * which nothing mounts any more (NodeDetail mounts `profile` and `positions`).
 */

const SPEAKER_ID = 'person:speaker-mock-1'
const SPEAKER_NAME = 'Marko Mock-Guest'
const TOPIC_ARC = 'topic:climate-policy'
const TOPIC_EMPTY = 'topic:quiet-topic'

const SEARCH_RESPONSE = {
  query: 'climate',
  query_type: 'raw_evidence',
  results: [
    {
      doc_id: 'insight:pt-entry',
      score: 0.9,
      text: `${SPEAKER_NAME} on carbon pricing.`,
      source_tier: 'insight',
      metadata: {
        doc_type: 'insight',
        source_id: 'insight:pt-entry',
        episode_id: 'ep-1',
        episode_title: 'Mock Episode',
        feed_id: 'sha256:mock',
        feed_title: 'Mock Feed',
        publish_date: '2026-04-18',
      },
      supporting_quotes: null,
      lifted: {
        insight: { id: 'insight:pt-entry', text: 'x', insight_type: 'claim', grounded: true },
        speaker: { id: SPEAKER_ID, display_name: SPEAKER_NAME },
        topic: null,
        quote: { timestamp_start_ms: 0, timestamp_end_ms: 1000 },
      },
    },
  ],
  lift_stats: null,
  error: null,
  detail: null,
}

function briefInsight(id: string, text: string, insightType: string) {
  return { insight: { id, properties: { text, insight_type: insightType } } }
}

const BRIEF = {
  path: '/mock/corpus',
  person_id: 'speaker-mock-1',
  topics: {
    [TOPIC_ARC]: [
      briefInsight('insight:a', 'Carbon pricing is the lever.', 'claim'),
      briefInsight('insight:b', 'Industry has started moving.', 'observation'),
    ],
    [TOPIC_EMPTY]: [briefInsight('insight:q', 'Is anyone measuring this?', 'question')],
  },
  quotes: [],
}

function arcInsight(id: string, text: string, insightType: string) {
  return { id, type: 'Insight', properties: { text, insight_type: insightType } }
}

function arcEpisode(episodeId: string, publishDate: string, insights: unknown[]) {
  return {
    episode_id: episodeId,
    publish_date: publishDate,
    episode_title: `Episode ${episodeId}`,
    feed_title: 'Mock Feed',
    episode_number: null,
    episode_image_url: null,
    episode_image_local_relpath: null,
    feed_image_url: null,
    insights,
  }
}

/** Server order: `cil_queries.position_arc` returns blocks sorted by `publish_date` ascending. */
const ARC_EPISODES = [
  arcEpisode('ep-2023', '2023-02-01', [arcInsight('insight:1', 'Early: a tax is enough.', 'claim')]),
  arcEpisode('ep-2024', '2024-05-10', [
    arcInsight('insight:2', 'Middle: markets are reacting.', 'observation'),
  ]),
  arcEpisode('ep-2025', '2025-09-30', [
    arcInsight('insight:3', 'Late: tax plus regulation.', 'claim'),
    arcInsight('insight:4', 'Late: will it hold?', 'question'),
  ]),
]

async function json(route: Route, body: unknown): Promise<void> {
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
}

async function openPositions(page: Page): Promise<void> {
  await mockSignIn(page, 'creator')
  await page.route('**/api/health**', (r) =>
    json(r, { status: 'ok', corpus_library_api: true, corpus_digest_api: true, search_api: true }),
  )
  await page.route('**/api/search?**', (r) => json(r, SEARCH_RESPONSE))
  await page.route('**/api/persons/*/brief?**', (r) => json(r, BRIEF))
  await page.route('**/api/persons/*/positions?**', (r) => {
    const topic = new URL(r.request().url()).searchParams.get('topic') ?? ''
    return json(r, {
      path: '/mock/corpus',
      person_id: 'speaker-mock-1',
      topic_id: topic,
      episodes: topic.endsWith('climate-policy') ? ARC_EPISODES : [],
    })
  })

  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill('/mock/corpus')
  await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
  await expect(page.locator('#search-q')).toBeEnabled({ timeout: 10_000 })
  await page.locator('#search-q').fill('climate')
  await page.locator('#search-q').press('Enter')
  await page.getByTestId('search-result-lifted-speaker-link').first().click()
  await expect(page.getByTestId('person-landing-view')).toBeVisible()

  await page.getByTestId('node-detail-rail-tab-position-tracker').click()
  await expect(page.getByTestId('person-landing-positions-view')).toBeVisible()
  await expect(page.getByTestId('person-landing-positions-lens-by-topic')).toHaveAttribute(
    'aria-selected',
    'true',
  )
}

function topicButton(page: Page, name: RegExp) {
  return page.getByTestId('person-landing-insights-voiced-topic-button').filter({ hasText: name })
}

test.describe('Position Tracker (UXS-009)', () => {
  test('picking a topic draws the arc oldest first; type chips narrow it; Clear returns to the picker', async ({
    page,
  }) => {
    await openPositions(page)
    await topicButton(page, /Climate Policy/i).click()

    const arc = page.getByTestId('position-tracker-arc')
    await expect(arc).toBeVisible()
    const rows = arc.getByTestId('position-tracker-row')
    await expect(rows).toHaveCount(4)
    const dates = await arc.getByTestId('position-tracker-row-date').allInnerTexts()
    expect(dates.map((d) => d.trim())).toEqual([
      '2023-02-01',
      '2024-05-10',
      '2025-09-30',
      '2025-09-30',
    ])
    await expect(rows.first()).toContainText('Early: a tax is enough.')
    await expect(rows.last()).toContainText('Late: will it hold?')

    const claim = page.getByTestId('position-tracker-filter-claim')
    await claim.click()
    await expect(claim).toHaveAttribute('aria-pressed', 'true')
    await expect(rows).toHaveCount(2)
    await expect(arc.getByTestId('position-tracker-row-type')).toHaveText(['claim', 'claim'])

    // Multi-select: adding "question" widens to claims + the question.
    await page.getByTestId('position-tracker-filter-question').click()
    await expect(rows).toHaveCount(3)

    await claim.click()
    await page.getByTestId('position-tracker-filter-question').click()
    await expect(rows).toHaveCount(4)

    // A type with no rows says so instead of drawing nothing.
    await page.getByTestId('position-tracker-filter-recommendation').click()
    await expect(page.getByTestId('position-tracker-filter-empty')).toHaveText(
      'No insights match the active filter.',
    )
    await expect(rows).toHaveCount(0)

    await arc.getByTestId('position-tracker-clear-topic').click()
    await expect(arc).toBeHidden()
    await expect(page.getByTestId('person-landing-positions-lens')).toBeVisible()
  })

  test('a topic with no arc for this person shows the empty state', async ({ page }) => {
    await openPositions(page)
    await topicButton(page, /Quiet Topic/i).click()
    const empty = page.getByTestId('position-tracker-empty')
    await expect(empty).toBeVisible()
    await expect(empty).toContainText(
      'No insights link this person to this topic in the corpus. Pick a different Topic.',
    )
    await expect(page.getByTestId('position-tracker-arc')).toHaveCount(0)
  })
})
