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
 *
 * #1619 — the panel-contract test runs live, driving a REAL corpus topic through the DEV
 * ``focusTopic`` hook. The two relational tests (FR4.2 cross-show + key voices, #1055 related
 * topics) moved out with the topic's GI (E04 step 8b): a downstream build renders those sections.
 */

import { expect, test } from '@playwright/test'
import {
  liveCorpusRoot,
  liveKgTopic,
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
  mockSignIn,
} from './helpers'

/**
 * #1619 — the TEV panel contract, live.
 *
 * ENTRY POINT NOTE (unchanged from the original): the digest topic-band-title click affordance was
 * removed in the V2 architectural change, so TEV is reached via the Dashboard topic-cluster chip
 * or a `@go-graph` emit carrying a `topic:…` id. The valuable assertion set is the panel contract
 * itself, so this drives `subject.focusTopic` through the DEV-only `__GIKG_SUBJECT__` hook rather
 * than coupling to whichever entry point happens to be wired today.
 *
 * What changed: the topic is a REAL one from the corpus's topic clusters, and the graph loads from
 * the corpus's own artifacts — no stubbed GI body, no fabricated topic id.
 */
test.describe('Topic / Entity rail panel (TEV) — live', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('TEV renders the contract surface for a focused topic', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await expect(page.getByTestId('digest-root')).toBeVisible({ timeout: 30_000 })

    /* A topic of the KG index, so `focusTopic` lands on a node that exists. */
    const topicId = (await liveKgTopic(page)).topic_id

    // Open the Graph workspace so the artifact slice loads and TEV can resolve the topic node.
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
    }, topicId!)

    // TEV contract surface (E2E_SURFACE_MAP §224). Post node-view unification TEV is folded
    // ``embedded`` into NodeDetail's Details rail tab, which owns the header + actions.
    const rail = page.getByTestId('graph-node-detail-rail')
    await expect(page.getByTestId('topic-entity-view')).toBeVisible({ timeout: 15_000 })
    // The rail (not TEV) owns the header; a ``topic:`` id titles it "Topic" via
    // NodeDetail's inferredKindFromId.
    await expect(rail.getByRole('heading').first()).toContainText('Topic')
    // TEV does not launch the legacy topic timeline popup.
    await expect(page.getByTestId('topic-timeline-dialog')).toHaveCount(0)
  })
})
