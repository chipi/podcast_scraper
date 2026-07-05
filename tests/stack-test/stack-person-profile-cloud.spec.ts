/**
 * #1076 chunk 4-B — cloud_thin variant of stack-person-profile.spec.ts.
 *
 * The airgapped_thin stack-test (sibling spec) only conditionally
 * exercises the click-to-Position-Tracker rich-data path because
 * airgapped_thin emits MENTIONS_PERSON best-effort (BART paraphrases
 * speaker names in some episodes). cloud_thin emits MENTIONS_PERSON
 * deterministically because the LLM extracts named entities at
 * insight-emit time, not via a post-pass substring match.
 *
 * Gating: spec runs only when ``STACK_TEST_PROFILE=cloud_thin`` is set,
 * which already drives the alternate-profile docker-compose path. Default
 * CI never runs it (recurring cloud API cost per
 * stack-error-recovery.spec.ts § cloud_thin smoke). Operator runs it
 * locally with API keys when validating prod-shape changes:
 *
 *   STACK_TEST_PROFILE=cloud_thin make ci-ui-full
 *
 * What it strictly asserts (in addition to the shell assertions the
 * airgapped_thin spec covers):
 *
 *   - ranked-topics section visible with ≥1 row for a real Person
 *   - clicking the first ranked-topic pivots to Position Tracker with
 *     ≥1 timeline row
 *   - the Position Tracker timeline includes at least one row carrying
 *     a non-null insight_type (the cloud LLM classifies all of them)
 *
 * A regression that breaks the LLM's MENTIONS_PERSON emission fires
 * this spec instead of silently passing under the airgapped conditional.
 */
import { expect, test } from '@playwright/test'

const STACK_TEST_CORPUS_PATH = '/app/output'
const STACK_TEST_PROFILE = process.env.STACK_TEST_PROFILE ?? ''

test.describe('Stack cloud_thin — Person Profile / Position Tracker (#1076 chunk 4-B)', () => {
  test.skip(
    STACK_TEST_PROFILE !== 'cloud_thin',
    'Cloud-profile spec gated on STACK_TEST_PROFILE=cloud_thin. ' +
      'Run via STACK_TEST_PROFILE=cloud_thin make ci-ui-full ' +
      '(requires OPENAI_API_KEY + GEMINI_API_KEY in the test env; ' +
      'incurs cloud API cost per run).',
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

  test('cloud_thin strictly emits MENTIONS_PERSON; ranked-topics + Position Tracker arc fire deterministically', async ({
    page,
    request,
  }) => {
    // Sanity: at least one KG artifact exists. cloud_thin's speaker-detector
    // (gemini_speaker_detector) populates Person nodes the same way
    // airgapped_thin does — strictly required for the test to pick a Person.
    const artifactsRes = await request.get(
      `/api/artifacts?path=${encodeURIComponent(STACK_TEST_CORPUS_PATH)}`,
    )
    expect(artifactsRes.status()).toBe(200)
    type ArtifactRow = { relative_path: string; kind: string }
    const artifactsBody = (await artifactsRes.json()) as { artifacts?: ArtifactRow[] }
    const kgArtifacts = (artifactsBody.artifacts ?? []).filter((a) => a.kind === 'kg')
    expect(kgArtifacts.length, 'expected at least one kg.json').toBeGreaterThan(0)

    const kgRes = await request.get(
      `/api/artifacts/${encodeURIComponent(kgArtifacts[0].relative_path)}?path=${encodeURIComponent(
        STACK_TEST_CORPUS_PATH,
      )}`,
    )
    expect(kgRes.status()).toBe(200)
    type KgNode = { id?: string; type?: string }
    const kgBody = (await kgRes.json()) as { nodes?: KgNode[] }
    const personNode = (kgBody.nodes ?? []).find((n) => n.type === 'Person')
    expect(personNode?.id, 'cloud_thin KG artifact must contain at least one Person node').toBeTruthy()
    const personId = String(personNode!.id)

    // Console-error gate.
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

    await page.evaluate((pid) => {
      const win = window as unknown as {
        __GIKG_SUBJECT__?: { focusPerson?: (id: string) => void }
      }
      const hook = win.__GIKG_SUBJECT__
      if (!hook || typeof hook.focusPerson !== 'function') {
        throw new Error('__GIKG_SUBJECT__.focusPerson hook unavailable')
      }
      hook.focusPerson(pid)
    }, personId)

    // === Shell assertions (same as airgapped_thin spec) ===
    // Post node-view fold: PLV is embedded in NodeDetail's rail, which owns the
    // header + tabs (PLV's own name header + internal tablist are hidden).
    const view = page.getByTestId('person-landing-view')
    await expect(view).toBeVisible({ timeout: 10_000 })
    const rail = page.getByTestId('graph-node-detail-rail')
    await expect(rail.getByRole('heading').first()).toContainText('Person')
    await expect(page.getByTestId('node-detail-rail-tab-details')).toHaveText(/Details/)
    await expect(page.getByTestId('person-landing-panel-profile')).toBeVisible()

    // === Strict rich-data assertions — what cloud_thin guarantees ===
    //
    // Unlike airgapped_thin where MENTIONS_PERSON is best-effort,
    // cloud_thin's LLM extracts entities deterministically at insight-emit
    // time. The "topic → Position Tracker" entry point (Insights voiced,
    // computed over MENTIONS_PERSON ∩ ABOUT) lives in the Positions tab's
    // default "By topic" lens and MUST populate.
    await page.getByTestId('node-detail-rail-tab-position-tracker').click()
    await expect(page.getByTestId('person-landing-positions-view')).toBeVisible()
    const insightsVoiced = page.getByTestId('person-landing-insights-voiced')
    await expect(
      insightsVoiced,
      'cloud_thin failed to emit MENTIONS_PERSON ∩ ABOUT for a real Person — ' +
        'this is a regression in the LLM emit path or the typed-MENTIONS post-pass.',
    ).toBeVisible({ timeout: 10_000 })
    const topicButtons = page.getByTestId('person-landing-insights-voiced-topic-button')
    expect(await topicButtons.count(), 'cloud_thin insights-voiced returned zero rows').toBeGreaterThan(0)

    // Click first topic → Position Tracker arc strictly populates (NodeDetail
    // keeps the Positions tab active so the arc is visible).
    await topicButtons.first().click()
    await expect(page.getByTestId('position-tracker-arc')).toBeVisible({ timeout: 5_000 })
    const rows = page.getByTestId('position-tracker-row')
    expect(await rows.count(), 'position-tracker arc empty for a real (Person, Topic) pair').toBeGreaterThan(0)

    // At least one row carries a non-null insight_type — cloud_thin's
    // LLM classifies all of them, so seeing zero means the classifier
    // path regressed.
    const typedRows = page.locator('[data-testid="position-tracker-row"][data-insight-type]:not([data-insight-type=""])')
    expect(
      await typedRows.count(),
      'position-tracker timeline has no rows with insight_type — LLM classifier may be off',
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
        `console errors during cloud_thin Person Profile walk:\n${fatal.join('\n')}`,
      ).toEqual([])
    }
  })
})
