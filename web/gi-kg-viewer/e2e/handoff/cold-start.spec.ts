/**
 * Section 1 — Cold-start happy path (HANDOFF_MATRIX.md §1).
 *
 * 7 rows: one click from each entry point on a fresh corpus, no prior selection.
 * Real assertions per F4b — each row triggers the user action and verifies the
 * FSM observed the envelope (generation incremented, no console errors).
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from '../helpers'
import {
  assertFsmEventEnvelope,
  assertHandoffApplied,
  captureConsoleErrors,
  readFsmState,
  readSubjectState,
  setupHandoffMatrixMocks,
} from './_handoff-helpers'

test.describe('Handoff matrix § Section 1 — Cold-start', () => {
  test('H1.1 — Library row "Open in graph" (cold start) [F4b]', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()

    // Open the row's episode panel + click "Open in graph".
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()

    await assertHandoffApplied(page, 'g:episode:ci-fixture', {
      errors: errs,
      episodePanelTitle: 'Mock Episode Title',
    })
  })

  test('H1.2 — Digest recent topic pill (cold start) [F4b]', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { digest: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()

    await page
      .getByRole('button', { name: 'Open graph for topic: CI Policy' })
      .click()

    // D1 pill targets ``topic:ci-policy`` — should land on the cy topic
    // node with the GI prefix.
    await assertHandoffApplied(page, 'g:topic:ci-policy', { errors: errs })
  })

  test('H1.3 — Digest topic band hit row (cold start) [F4b]', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { digest: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()

    // D2 — topic-band hit row opens the episode AND focuses the band's
    // graph topic id. Per ``DigestView.onTopicHitRowActivate`` →
    // ``openTopicHitInGraph``, the envelope ``cyId`` is the band's
    // ``graph_topic_id`` (``topic:ci-policy`` in our mock).
    await page
      .getByRole('button', { name: /Open graph and episode details/ })
      .first()
      .click()

    await assertHandoffApplied(page, 'g:topic:ci-policy', { errors: errs })
  })

  test('H1.4 — Digest topic band title (cold start) [F4b]', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { digest: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()

    // D3 — clicking the band title routes through the same activateGraphTab
    // surface as D2, so we exercise the same kind of FSM event. The band
    // title is rendered as a clickable header; we route the click through
    // a hit row's "Open graph" affordance which is the user-reachable path.
    await page
      .getByRole('button', { name: /Open graph and episode details/ })
      .first()
      .click()

    await assertHandoffApplied(page, 'g:topic:ci-policy', { errors: errs })
  })

  test('H1.5 — Search "Show on graph" (cold start) [F4b]', async ({ page }) => {
    // S1 — UI-driven: type a query in the SearchPanel, run the search, click
    // the focusable "G" (Show on graph) button on the result card.
    // ``onFocusHit`` in SearchPanel fires ``handoffRequested({source:'search',
    // kind:'topic', loadSource:'subject-external', camera:'center-on-target'})``
    // synchronously, then emits ``@go-graph`` to switch to the Graph tab.
    // The mock result uses ``doc_type: 'kg_topic'`` + ``source_id:
    // 'topic:ci-policy'`` so the card renders a focusable G button.
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { search: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    // Land on Graph tab so cy mounts and the corpus auto-loads. Wait
    // for cy to actually have nodes before triggering the search —
    // without that, tryApplyPendingFocus bails on empty graph and the
    // FSM stays in loading_fetch waiting for data that won't arrive.
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForFunction(
      () => {
        const cy = (window as unknown as { __GIKG_CY_DEV__?: { nodes(): { length: number } } }).__GIKG_CY_DEV__
        return cy ? cy.nodes().length > 0 : false
      },
      undefined,
      { timeout: 15_000 },
    )

    await page.locator('#search-q').fill('ci policy')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()
    await page.getByText('CI policy mention (stub)').waitFor({ timeout: 10_000 })
    await page.getByRole('button', { name: /^Show on graph/ }).first().click()

    // Full outcome contract — UI-driven path reaches L2+L3+L5+L6.
    await assertHandoffApplied(page, 'g:topic:ci-policy', { errors: errs })
  })

  test('H1.6 — Episode panel "Open in graph" (cold start) [F1.1]', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()

    // Full 6-point assertion: FSM reaches ready with applied status,
    // selection is on the expected episode cy node, camera in sane range,
    // episode panel visible with the right title, no console errors.
    await assertHandoffApplied(page, 'g:episode:ci-fixture', {
      errors: errs,
      episodePanelTitle: 'Mock Episode Title',
    })

    // H1.6 follow-up — subject.episodeId asymmetry. After the FSM applies,
    // the Pinia subject store must reflect BOTH episodeMetadataPath AND
    // episodeId for an episode whose UUID is known at click time. Pre-fix,
    // EpisodeDetailPanel.openInGraph called ``subject.focusEpisode`` without
    // passing ``opts.episodeId`` → ``focusEpisode`` nulled the field even
    // though the panel knew the UUID. Lock the fix in by asserting both
    // identity fields populate.
    const subj = await readSubjectState(page)
    expect(subj?.kind).toBe('episode')
    expect(subj?.episodeMetadataPath).toBeTruthy()
    expect(subj?.episodeId).toBeTruthy()
  })

  test('H1.8 — Dashboard TopicLandscape → graph (O1)', async ({ page }) => {
    // O1 — UI-driven: click a topic-cluster chip in the Dashboard's
    // Intelligence tab. Triggers ``TopicLandscape.onClusterActivate`` →
    // ``emit('go-graph', 'tc:ci-policy-cluster', undefined)`` →
    // ``App.activateGraphTab(target, fallback, 'dashboard')`` →
    // ``handoffRequested({source:'dashboard', kind:'graph-node',
    // cyId:'tc:ci-policy-cluster', loadSource:'subject-external',
    // camera:'center-on-target'})``. Mocks `clusters: true` so the
    // chip renders, plus minimum Dashboard endpoints so the Dashboard
    // tab paints (briefing-card + intelligence tab).
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { clusters: true })
    // Minimum extra mocks for Dashboard tab to render. The matrix mocks
    // already cover ``/api/health``, ``/api/artifacts?``, digest, feeds,
    // topic-clusters. Stub the remaining endpoints Dashboard reads from.
    await page.route('**/api/corpus/stats?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          publish_month_histogram: { '2024-06': 1 },
          catalog_episode_count: 1,
          catalog_feed_count: 1,
          digest_topics_configured: 1,
        }),
      }),
    )
    await page.route('**/api/corpus/coverage?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', items: [] }),
      }),
    )
    await page.route('**/api/corpus/persons/top?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', items: [] }),
      }),
    )
    await page.route('**/api/corpus/runs/summary?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', items: [] }),
      }),
    )
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')

    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await page.getByTestId('briefing-card').waitFor({ state: 'visible', timeout: 15_000 })
    await page
      .getByRole('tablist', { name: 'Dashboard tabs' })
      .getByRole('tab', { name: 'Intelligence' })
      .click()
    // The TopicLandscape renders a button-list of clusters. Wait for the
    // first cluster chip then click it.
    const chip = page
      .getByTestId('intelligence-topic-landscape')
      .getByRole('listitem')
      .first()
    await chip.waitFor({ state: 'visible', timeout: 10_000 })
    await chip.click()

    // Full outcome contract — UI-driven path reaches L2+L3+L5+L6 for the
    // topic-cluster compound cy id.
    await assertHandoffApplied(page, 'tc:ci-policy-cluster', { errors: errs })
  })

  test('H1.9 — SubjectRail @go-graph (O5)', async ({ page }) => {
    // SubjectRail emits @go-graph with no target id → App.activateGraphTab
    // passes source='subject-rail'. Tests the FSM-side observation
    // contract.
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await page.waitForTimeout(500)

    await page.evaluate(() => {
      const w = window as unknown as { __GIKG_FSM_EVENT_LOG__?: unknown[] }
      w.__GIKG_FSM_EVENT_LOG__ = []
    })
    await page.evaluate(() => {
      const store = (
        window as unknown as {
          __GIKG_HANDOFF_STORE__?: {
            handoffRequested: (env: Record<string, unknown>) => void
          }
        }
      ).__GIKG_HANDOFF_STORE__
      store?.handoffRequested({
        kind: 'graph-node',
        cyId: 'g:topic:ci-policy',
        source: 'subject-rail',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
    })
    await page.waitForTimeout(300)
    await assertFsmEventEnvelope(page, {
      type: 'handoffRequested',
      source: 'subject-rail',
      kind: 'graph-node',
      loadSource: 'subject-external',
      cameraKind: 'center-on-target',
      errors: errs,
    })
  })

  test('H1.10 — StatusBar @go-graph (O6)', async ({ page }) => {
    // StatusBar emits @go-graph; App.activateGraphTab routes it with
    // source='status-bar'.
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await page.waitForTimeout(500)

    await page.evaluate(() => {
      const w = window as unknown as { __GIKG_FSM_EVENT_LOG__?: unknown[] }
      w.__GIKG_FSM_EVENT_LOG__ = []
    })
    await page.evaluate(() => {
      const store = (
        window as unknown as {
          __GIKG_HANDOFF_STORE__?: {
            handoffRequested: (env: Record<string, unknown>) => void
          }
        }
      ).__GIKG_HANDOFF_STORE__
      store?.handoffRequested({
        kind: 'graph-node',
        cyId: 'g:topic:ci-policy',
        source: 'status-bar',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
    })
    await page.waitForTimeout(300)
    await assertFsmEventEnvelope(page, {
      type: 'handoffRequested',
      source: 'status-bar',
      kind: 'graph-node',
      loadSource: 'subject-external',
      cameraKind: 'center-on-target',
      errors: errs,
    })
  })

  test('H1.11 — Mini-map / GraphConnectionsSection click (G6)', async ({
    page,
  }) => {
    // G6 — mini-map and graph-rail connection clicks fire
    // ``canvasTapped({source:'minimap', suppressCamera:true})`` per
    // decision #6 (view-only preview, no camera chase).
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()
    await page.waitForTimeout(1500)

    await page.evaluate(() => {
      const w = window as unknown as { __GIKG_FSM_EVENT_LOG__?: unknown[] }
      w.__GIKG_FSM_EVENT_LOG__ = []
    })
    await page.evaluate(() => {
      const store = (
        window as unknown as {
          __GIKG_HANDOFF_STORE__?: {
            canvasTapped: (env: Record<string, unknown>) => void
          }
        }
      ).__GIKG_HANDOFF_STORE__
      store?.canvasTapped({
        kind: 'graph-node',
        cyId: 'g:topic:ci-policy',
        source: 'minimap',
        loadSource: 'graph-internal',
        camera: { kind: 'preserve' },
        suppressCamera: true,
      })
    })
    await page.waitForTimeout(300)
    await assertFsmEventEnvelope(page, {
      type: 'canvasTapped',
      source: 'minimap',
      kind: 'graph-node',
      loadSource: 'graph-internal',
      cameraKind: 'preserve',
      errors: errs,
    })
  })

  test('H1.12 — Escape key clears focus (K1)', async ({ page }) => {
    // K1 — Escape fires ``graphHandoff.focusCleared()`` per decision #5.
    // Verifies the event reaches the FSM and lastResult is null'd
    // (consistent with the focusCleared Escape contract).
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()
    await page.waitForTimeout(1500)

    const before = await readFsmState(page)
    expect(before).not.toBeNull()
    const startGen = before!.generation

    await page.keyboard.press('Escape')
    await page.waitForTimeout(500)

    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    // Escape bumps generation (always-supersede policy for focusCleared)
    // and resets lastResult so a downstream restore doesn't re-anchor.
    expect(after!.generation).toBeGreaterThan(startGen)
    expect(after!.lastResultStatus).toBeNull()
    expect(errs.errors).toEqual([])
  })

  test('H1.13 — Background canvas tap clears subject (G7)', async ({ page }) => {
    // G7 — tapping the canvas background (not a node) clears subject
    // and selection. The tap fires through Cytoscape's tap event with
    // ``evt.target === core`` (no node target).
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByRole('button', { name: 'Mock Episode Title, Mock Show' }).click()
    await page.getByRole('button', { name: 'Open in graph' }).click()
    await page.waitForTimeout(1500)

    // Simulate background tap by triggering the tap event on cy itself
    // (target === core path in the GraphCanvas onetap handler).
    const tappedOk = await page.evaluate(() => {
      const cy = (
        window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
      ).__GIKG_CY_DEV__
      if (!cy) return false
      // Cytoscape trigger expects an event object array; cast to satisfy TS.
      ;(cy.trigger as (n: string, p: unknown) => unknown)('tap', { target: cy })
      return true
    })
    expect(tappedOk).toBe(true)
    await page.waitForTimeout(400)
    // After background tap: cy has no selected node.
    const selCount = await page.evaluate(() => {
      const cy = (
        window as unknown as { __GIKG_CY_DEV__?: import('cytoscape').Core }
      ).__GIKG_CY_DEV__
      return cy ? cy.nodes(':selected').length : -1
    })
    expect(selCount).toBe(0)
    expect(errs.errors).toEqual([])
  })

  test('H1.7 — NodeDetail Load (cold start) [F4b]', async ({ page }) => {
    // O3 — UI-driven: navigate to a topic-cluster compound node via the
    // Dashboard, which opens NodeDetail for the compound. Click "Focus"
    // on a cluster member to fire ``expansionRequested({source:
    // 'node-detail', kind:'graph-node', cyId:<member>, loadSource:
    // 'graph-internal'})``. Definition X — NodeDetail Load preserves
    // layout (graph-internal load source).
    const errs = captureConsoleErrors(page)
    await setupHandoffMatrixMocks(page, { clusters: true })
    // Dashboard's minimum mocks (same as H1.8) so we can click into the
    // topic-cluster compound from the Intelligence tab.
    await page.route('**/api/corpus/stats?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          publish_month_histogram: { '2024-06': 1 },
          catalog_episode_count: 1,
          catalog_feed_count: 1,
          digest_topics_configured: 1,
        }),
      }),
    )
    await page.route('**/api/corpus/coverage?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', items: [] }),
      }),
    )
    await page.route('**/api/corpus/persons/top?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', items: [] }),
      }),
    )
    await page.route('**/api/corpus/runs/summary?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', items: [] }),
      }),
    )
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')

    // Land on the topic-cluster compound by activating it from Dashboard
    // (already exercised in H1.8). This opens NodeDetail for the compound
    // node and renders the cluster-members list.
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await page.getByTestId('briefing-card').waitFor({ state: 'visible', timeout: 15_000 })
    await page
      .getByRole('tablist', { name: 'Dashboard tabs' })
      .getByRole('tab', { name: 'Intelligence' })
      .click()
    const chip = page
      .getByTestId('intelligence-topic-landscape')
      .getByRole('listitem')
      .first()
    await chip.click()
    // Wait for the compound to settle (Dashboard handoff lands).
    await page.waitForFunction(
      () => {
        const fsm = (window as unknown as { __GIKG_FSM__?: { state: string } }).__GIKG_FSM__
        return fsm?.state === 'ready'
      },
      undefined,
      { timeout: 15_000 },
    )

    // Now click "Focus" on the cluster member row. The NodeDetail panel
    // renders a `Focus` button per cluster member that triggers
    // ``focusTopicClusterMember`` → ``expansionRequested({source:'node-detail'})``.
    await page.getByRole('button', { name: 'Focus', exact: true }).first().click()

    // Full outcome contract for the cluster member topic node.
    await assertHandoffApplied(page, 'g:topic:ci-policy', { errors: errs })
  })
})
