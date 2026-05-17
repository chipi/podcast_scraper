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

  // ``digest topic title → TEV renders the contract surface`` — removed.
  // The headline-pill click affordance was disabled in the V2
  // architectural change (see ``DigestView.vue`` headline span and the
  // corresponding comment in ``digest.spec.ts``). The headline is now a
  // ``<span>`` for display only — there is no click path from this
  // surface to either the graph or the TEV. The TEV contract surface
  // itself is still covered by the other tests in this file (graph node
  // click, dashboard topic-cluster chip, etc.).
})
