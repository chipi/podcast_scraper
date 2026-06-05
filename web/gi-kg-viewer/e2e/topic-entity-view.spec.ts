/**
 * Topic Entity View (TEV) Playwright spec — coverage gap surfaced by
 * #678 PR-A audit.
 *
 * TEV is the SubjectRail panel that renders when
 * ``subject.kind === 'topic'``. Entry point (per E2E_SURFACE_MAP §224):
 *
 *   * Digest topic-band title click → ``openTopicBandInGraph`` opens the
 *     graph for the band's top GI/KG hit and calls
 *     ``subject.focusTopic(graph_topic_id)`` so TEV renders.
 *
 * Smoke contract (per E2E_SURFACE_MAP):
 *
 *   - ``topic-entity-view`` root visible
 *   - ``topic-entity-view-kind`` shows "Topic" or "Entity"
 *   - ``topic-entity-view-name`` shows the topic name
 *   - ``topic-entity-view-stats`` shows the dated-mentions sentence
 *   - When mentions exist: ``topic-entity-view-mentions`` list appears;
 *     when empty: ``topic-entity-view-empty`` placeholder
 *   - Action buttons ``topic-entity-view-go-graph`` /
 *     ``topic-entity-view-prefill-search`` present
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

const TOPIC_LABEL = 'Mock TEV Topic'
const GRAPH_TOPIC_ID = 'topic:mock-tev'

// Minimal GI artifact containing one Episode + one Topic node so the
// graph workspace can render and ``subject.focusTopic`` lands on a real
// node.
const GI_BODY = JSON.stringify({
  schema_version: '1.0',
  model_version: 'stub',
  prompt_version: 'v1',
  episode_id: 'e1',
  nodes: [
    {
      id: 'episode:e1',
      type: 'Episode',
      properties: {
        podcast_id: 'podcast:unknown',
        title: 'TEV Episode Alpha',
        publish_date: '2024-06-05T00:00:00Z',
      },
    },
    {
      id: GRAPH_TOPIC_ID,
      type: 'Topic',
      properties: { label: TOPIC_LABEL },
    },
  ],
  edges: [],
})

test.describe('Topic / Entity rail panel (TEV)', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
        }),
      })
    })

    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          artifacts: [
            {
              name: 'ep1.gi.json',
              relative_path: 'metadata/ep1.gi.json',
              kind: 'gi',
              size_bytes: GI_BODY.length,
              mtime_utc: '2024-06-05T12:00:00Z',
              publish_date: '2024-06-05',
            },
          ],
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/ep1.gi.json**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: GI_BODY,
      })
    })

    await page.route('**/api/corpus/feeds**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          feeds: [{ feed_id: 'f1', display_title: 'Mock Feed Show', episode_count: 1 }],
        }),
      })
    })

    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          available: true,
          reason: null,
          stats: {
            total_vectors: 1,
            doc_type_counts: {},
            feeds_indexed: ['f1'],
            embedding_model: 'mock',
            embedding_dim: 8,
            last_updated: '2024-01-01T00:00:00Z',
            index_size_bytes: 0,
          },
          reindex_recommended: false,
        }),
      })
    })

    await page.route('**/api/corpus/digest**', async (route) => {
      const url = new URL(route.request().url())
      const win = url.searchParams.get('window') || 'all'
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          window: win,
          window_start_utc: '1970-01-01T00:00:00Z',
          window_end_utc: '2024-06-08T00:00:00Z',
          compact: false,
          rows: [
            {
              metadata_relative_path: 'metadata/ep1.metadata.json',
              feed_id: 'f1',
              feed_display_title: 'Mock Feed Show',
              episode_id: 'e1',
              episode_title: 'TEV Episode Alpha',
              publish_date: '2024-06-05',
              summary_title: 'TEV digest summary',
              summary_bullets_preview: ['First bullet'],
              summary_bullet_graph_topic_ids: [GRAPH_TOPIC_ID],
              summary_preview: 'TEV digest summary — First bullet',
              gi_relative_path: 'metadata/ep1.gi.json',
              kg_relative_path: null,
              has_gi: true,
              has_kg: false,
              cil_digest_topics: [],
            },
          ],
          topics: [
            {
              topic_id: 't1',
              label: TOPIC_LABEL,
              query: 'mock query',
              graph_topic_id: GRAPH_TOPIC_ID,
              hits: [
                {
                  metadata_relative_path: 'metadata/ep1.metadata.json',
                  episode_title: 'TEV Episode Alpha',
                  feed_id: 'f1',
                  feed_display_title: 'Mock Feed Show',
                  publish_date: '2024-06-05',
                  score: 0.9,
                  summary_preview: 'TEV digest summary — First bullet',
                  episode_id: 'e1',
                  gi_relative_path: 'metadata/ep1.gi.json',
                  kg_relative_path: null,
                  has_gi: true,
                  has_kg: false,
                },
              ],
            },
          ],
          topics_unavailable_reason: null,
        }),
      })
    })
  })

  test('TEV renders the contract surface for a focused topic', async ({ page }) => {
    // ENTRY POINT NOTE — The original spec drove this through the digest
    // topic-band-title click affordance. That click handler was removed
    // in the V2 architectural change (``DEFAULT_DIGEST_TOPICS`` editorial
    // labels don't correspond to KG nodes in arbitrary corpora — see the
    // headline ``<span>`` in ``DigestView.vue`` and the comment in
    // ``digest.spec.ts``). TEV is still reachable from other surfaces:
    //
    //   - Dashboard topic-cluster chip → ``activateGraphTab(topic:…)`` →
    //     ``subject.focusTopic`` (App.vue:150)
    //   - ``@go-graph`` emit with a ``topic:…`` target id from any surface
    //
    // The valuable assertion set is the TEV contract surface itself
    // (data-testids per E2E_SURFACE_MAP §224). Drive ``subject.focusTopic``
    // directly via the DEV-only ``__GIKG_SUBJECT__`` hook so this spec
    // verifies the panel contract without coupling to whichever entry
    // point happens to be wired today.
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()

    // Open the Graph workspace so the artifact slice loads and TEV can
    // resolve the topic node when ``subject.focusTopic`` fires.
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    // Drive TEV via the DEV hook (DEV-only mutator on ``__GIKG_SUBJECT__``).
    await page.evaluate((id) => {
      const subj = (window as unknown as { __GIKG_SUBJECT__?: { focusTopic: (i: string) => void } })
        .__GIKG_SUBJECT__
      if (!subj?.focusTopic) {
        throw new Error('__GIKG_SUBJECT__.focusTopic not exposed; DEV-only hook missing')
      }
      subj.focusTopic(id)
    }, GRAPH_TOPIC_ID)

    // TEV contract surface (E2E_SURFACE_MAP §224).
    const view = page.getByTestId('topic-entity-view')
    await expect(view).toBeVisible({ timeout: 10_000 })

    await expect(page.getByTestId('topic-entity-view-kind')).toContainText(/topic|entity/i)
    await expect(page.getByTestId('topic-entity-view-name')).toContainText(TOPIC_LABEL)

    // The fixture has a Topic node with no mentions, so the stats sentence
    // (gated on timeline.total > 0) is absent and the empty placeholder
    // renders instead. Either branch satisfies the contract.
    const stats = view.getByTestId('topic-entity-view-stats')
    const empty = view.getByTestId('topic-entity-view-empty')
    const mentions = view.getByTestId('topic-entity-view-mentions')
    const statsCount = await stats.count()
    const emptyCount = await empty.count()
    const mentionsCount = await mentions.count()
    expect(statsCount + emptyCount + mentionsCount).toBeGreaterThan(0)

    // Action buttons present.
    await expect(page.getByTestId('topic-entity-view-go-graph')).toBeVisible()
    await expect(page.getByTestId('topic-entity-view-prefill-search')).toBeVisible()
    // TEV does not launch the legacy topic timeline popup.
    await expect(page.getByTestId('topic-timeline-dialog')).toHaveCount(0)
  })

  test('FR4.2: TEV shows cross-show coverage and key voices (relational layer)', async ({
    page,
  }) => {
    await page.route('**/api/relational/cross-show**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          subject: GRAPH_TOPIC_ID,
          groups: {
            'podcast:show-one': [
              { id: 'insight:1', type: 'insight', text: 'Show one take on the topic.', show_id: 'podcast:show-one', episode_id: 'e1' },
            ],
            'podcast:show-two': [
              { id: 'insight:2', type: 'insight', text: 'Show two take on the topic.', show_id: 'podcast:show-two', episode_id: 'e2' },
            ],
          },
          error: null,
        }),
      })
    })
    await page.route('**/api/relational/who-said**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          subject: GRAPH_TOPIC_ID,
          groups: {
            'person:jane-doe': [
              { id: 'insight:1', type: 'insight', text: 'Jane on the topic.', show_id: 'podcast:show-one', episode_id: 'e1' },
            ],
          },
          error: null,
        }),
      })
    })
    await page.route('**/api/relational/topic-entities**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          subject: GRAPH_TOPIC_ID,
          results: [
            { id: 'org:acme', type: 'org', text: 'Acme Corp', show_id: '', episode_id: '' },
          ],
          error: null,
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
    await page.evaluate((id) => {
      const subj = (window as unknown as { __GIKG_SUBJECT__?: { focusTopic: (i: string) => void } })
        .__GIKG_SUBJECT__
      if (!subj?.focusTopic) throw new Error('__GIKG_SUBJECT__.focusTopic not exposed')
      subj.focusTopic(id)
    }, GRAPH_TOPIC_ID)

    const view = page.getByTestId('topic-entity-view')
    await expect(view).toBeVisible({ timeout: 10_000 })

    const cross = view.getByTestId('tev-cross-show')
    await expect(cross.getByTestId('tev-cross-show-row')).toHaveCount(2)
    await expect(cross).toContainText('Show one take on the topic.')

    const voices = view.getByTestId('tev-voices')
    await expect(voices.getByTestId('tev-voice-row')).toHaveCount(1)

    // FR4.2 entities-involved (3b): chip lists the mentioned entity.
    const entities = view.getByTestId('tev-entities')
    await expect(entities.getByTestId('tev-entity-chip')).toHaveCount(1)
    await expect(entities).toContainText('Acme Corp')

    // Clicking a voice opens the Person Landing rail.
    await voices.getByTestId('tev-voice-link').first().click()
    await expect(page.getByTestId('person-landing-view')).toBeVisible()
  })
})
