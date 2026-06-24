/**
 * #1076 chunk 4-B.3 — cloud_thin Tier-3 spec for graph node-detail's
 * Connections (Neighbourhood) section.
 *
 * The Connections section on a graph node's rail
 * (`graph-connections-section`, on the `node-detail-rail-tab-neighbourhood`
 * tab) lists 1-hop neighbors derived from the loaded graph slice's
 * edges. For Person / Topic / Organization nodes that means typed
 * `MENTIONS_PERSON` / `MENTIONS_ORG` / `ABOUT` edges from Insights —
 * the same edges PersonLandingView and TopicEntityView consume. If
 * those edges don't emit, the neighbor list is empty even when the
 * graph clearly has the connections.
 *
 * Under cloud_thin the LLM emits typed-MENTIONS at insight-emit time
 * so the Connections strip should populate strictly for any Insight
 * node that has supporting Quote+MENTIONS_PERSON evidence. This spec
 * asserts that.
 *
 * Gating: STACK_TEST_PROFILE=cloud_thin same as the sibling cloud specs.
 */
import { expect, test } from '@playwright/test'

const STACK_TEST_CORPUS_PATH = '/app/output'
const STACK_TEST_PROFILE = process.env.STACK_TEST_PROFILE ?? ''

test.describe('Stack cloud_thin — NodeDetail Connections (#1076 chunk 4-B.3)', () => {
  test.skip(
    STACK_TEST_PROFILE !== 'cloud_thin',
    'Cloud-profile spec gated on STACK_TEST_PROFILE=cloud_thin. ' +
      'Run via STACK_TEST_PROFILE=cloud_thin make ci-ui-full.',
  )

  test.beforeEach(async ({ page }) => {
    await page.addInitScript((corpus) => {
      try {
        window.localStorage.setItem('ps_corpus_path', corpus)
        window.localStorage.setItem('ps_graph_hints_seen', '1')
      } catch {
        /* private mode / quota — ignore */
      }
    }, STACK_TEST_CORPUS_PATH)
  })

  test('cloud_thin populates graph-connections-section for an Insight node with typed-MENTIONS evidence', async ({
    page,
    request,
  }) => {
    // Pick a real Insight id from one of the GI artifacts. Insights
    // are the node type that typically has the richest 1-hop graph —
    // SUPPORTED_BY → Quote, ABOUT → Topic, MENTIONS_PERSON → Person,
    // MENTIONS_ORG → Organization. If Connections is empty for one,
    // either the GI emit path or the graph-slice edge inclusion broke.
    const artifactsRes = await request.get(
      `/api/artifacts?path=${encodeURIComponent(STACK_TEST_CORPUS_PATH)}`,
    )
    expect(artifactsRes.status()).toBe(200)
    type ArtifactRow = { relative_path: string; kind: string }
    const artifactsBody = (await artifactsRes.json()) as { artifacts?: ArtifactRow[] }
    const giArtifacts = (artifactsBody.artifacts ?? []).filter((a) => a.kind === 'gi')
    expect(giArtifacts.length).toBeGreaterThan(0)

    const giRes = await request.get(
      `/api/artifacts/${encodeURIComponent(giArtifacts[0].relative_path)}?path=${encodeURIComponent(
        STACK_TEST_CORPUS_PATH,
      )}`,
    )
    expect(giRes.status()).toBe(200)
    type GiNode = { id?: string; type?: string }
    type GiEdge = { from?: string; to?: string; type?: string }
    const giBody = (await giRes.json()) as { nodes?: GiNode[]; edges?: GiEdge[] }
    const insightNodes = (giBody.nodes ?? []).filter((n) => n.type === 'Insight')
    expect(
      insightNodes.length,
      'cloud_thin GI artifact must contain Insight nodes',
    ).toBeGreaterThan(0)

    // Pick an Insight that has at least one MENTIONS_PERSON edge —
    // that's the rich-data invariant cloud_thin must hold.
    const insightIdsWithMP = new Set(
      (giBody.edges ?? [])
        .filter((e) => e.type === 'MENTIONS_PERSON')
        .map((e) => e.from)
        .filter((id): id is string => typeof id === 'string'),
    )
    expect(
      insightIdsWithMP.size,
      'cloud_thin GI artifact must contain at least one MENTIONS_PERSON edge ' +
        'sourced from an Insight node. If this fails, the LLM emit path or the ' +
        'typed-MENTIONS post-pass regressed.',
    ).toBeGreaterThan(0)
    const targetInsightId = Array.from(insightIdsWithMP)[0]

    const consoleErrors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text())
    })
    page.on('pageerror', (err) => consoleErrors.push(err.message))

    await page.goto('/')
    await page
      .getByRole('navigation', { name: 'Main views' })
      .getByRole('button', { name: 'Graph' })
      .click()
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible({ timeout: 10_000 })
    await expect(page.locator('.graph-canvas')).toBeVisible({ timeout: 60_000 })

    // Focus the Insight as a graph node — opens NodeDetail in the rail.
    await page.evaluate((id) => {
      const win = window as unknown as {
        __GIKG_SUBJECT__?: { focusGraphNode?: (id: string) => void }
      }
      const hook = win.__GIKG_SUBJECT__
      if (!hook || typeof hook.focusGraphNode !== 'function') {
        throw new Error('__GIKG_SUBJECT__.focusGraphNode hook unavailable')
      }
      hook.focusGraphNode(id)
    }, targetInsightId)

    // Activate the Neighbourhood tab — Connections lives there.
    await expect(page.getByTestId('graph-node-detail-rail')).toBeVisible({
      timeout: 10_000,
    })
    await page.getByTestId('node-detail-rail-tab-neighbourhood').click()

    // === Strict rich-data — what cloud_thin guarantees ===
    const connections = page.getByTestId('graph-connections-section')
    await expect(
      connections,
      'cloud_thin Insight node has no Connections section visible. The ' +
        '1-hop neighbor list should include the MENTIONS_PERSON target ' +
        'we confirmed from the GI artifact.',
    ).toBeVisible({ timeout: 10_000 })

    // At least one neighbor row should resolve to a graph node id —
    // the focus-graph button surfaces that id.
    const focusButtons = connections.getByTestId('graph-connection-focus-graph')
    expect(
      await focusButtons.count(),
      'Connections section has zero focus-graph rows. cloud_thin GI emit ' +
        'or graph-slice edge inclusion regressed.',
    ).toBeGreaterThan(0)

    if (consoleErrors.length) {
      const fatal = consoleErrors.filter(
        (e) =>
          !/HMR|deprecated|Vite|dmn_chk.*invalid domain|rejected for invalid domain/i.test(
            e,
          ) && !/"notify",\s*\w+ is null/i.test(e),
      )
      expect(
        fatal,
        `console errors during cloud_thin NodeDetail Connections walk:\n${fatal.join('\n')}`,
      ).toEqual([])
    }
  })
})
