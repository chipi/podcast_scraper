/**
 * #1076 chunk 4-B.2 — cloud_thin Tier-3 spec for TopicEntityView's
 * relational-query sections (tev-voices, tev-entities, tev-cross-show).
 *
 * TopicEntityView's PRD-033 FR4.2 sections all depend on typed edges
 * the LLM emits at insight-emit time under cloud_thin:
 *
 *   - tev-voices reads `who_said(topic)` which traverses
 *     `Insight ─ABOUT→ Topic` + `Insight ─MENTIONS_PERSON→ Person`
 *   - tev-entities reads `entities_in_topic` which traverses
 *     `Insight ─MENTIONS_PERSON / MENTIONS_ORG→ Person|Org`
 *   - tev-cross-show reads `cross_show_synthesis(topic)` which scopes
 *     hybrid search to topic then groups by show
 *
 * Under airgapped_thin all three sections degrade (BART paraphrasing
 * misses the typed-MENTIONS post-pass). Under cloud_thin they should
 * populate strictly because the LLM emits MENTIONS_PERSON/ORG at
 * insight-emit time, not via post-pass.
 *
 * Gating: same STACK_TEST_PROFILE=cloud_thin gate as
 * stack-person-profile-cloud.spec.ts. Default CI never runs it.
 */
import { expect, test } from '@playwright/test'

const STACK_TEST_CORPUS_PATH = '/app/output'
const STACK_TEST_PROFILE = process.env.STACK_TEST_PROFILE ?? ''

test.describe('Stack cloud_thin — TopicEntityView relational sections (#1076 chunk 4-B.2)', () => {
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

  test('cloud_thin strictly populates TopicEntityView relational sections (who_said + entities_in_topic)', async ({
    page,
    request,
  }) => {
    // Pull a real Topic id from one of the KG artifacts.
    const artifactsRes = await request.get(
      `/api/artifacts?path=${encodeURIComponent(STACK_TEST_CORPUS_PATH)}`,
    )
    expect(artifactsRes.status()).toBe(200)
    type ArtifactRow = { relative_path: string; kind: string }
    const artifactsBody = (await artifactsRes.json()) as { artifacts?: ArtifactRow[] }
    const kgArtifacts = (artifactsBody.artifacts ?? []).filter((a) => a.kind === 'kg')
    expect(kgArtifacts.length).toBeGreaterThan(0)

    const kgRes = await request.get(
      `/api/artifacts/${encodeURIComponent(kgArtifacts[0].relative_path)}?path=${encodeURIComponent(
        STACK_TEST_CORPUS_PATH,
      )}`,
    )
    expect(kgRes.status()).toBe(200)
    type KgNode = { id?: string; type?: string }
    const kgBody = (await kgRes.json()) as { nodes?: KgNode[] }
    const topicNode = (kgBody.nodes ?? []).find((n) => n.type === 'Topic')
    expect(
      topicNode?.id,
      'cloud_thin KG artifact must contain at least one Topic node',
    ).toBeTruthy()
    const topicId = String(topicNode!.id)

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

    // Focus the Topic via the DEV-gated subject hook.
    await page.evaluate((tid) => {
      const win = window as unknown as {
        __GIKG_SUBJECT__?: { focusTopic?: (id: string) => void }
      }
      const hook = win.__GIKG_SUBJECT__
      if (!hook || typeof hook.focusTopic !== 'function') {
        throw new Error('__GIKG_SUBJECT__.focusTopic hook unavailable')
      }
      hook.focusTopic(tid)
    }, topicId)

    // === Strict shell — what TopicEntityView always renders ===
    await expect(page.getByTestId('topic-entity-view')).toBeVisible({ timeout: 10_000 })
    await expect(page.getByTestId('topic-entity-view-kind')).toBeVisible()
    await expect(page.getByTestId('topic-entity-view-name')).toBeVisible()

    // === Strict rich-data — what cloud_thin guarantees ===
    //
    // tev-voices reads who_said(topic), which depends on
    // MENTIONS_PERSON ∩ ABOUT. Under cloud_thin this should populate.
    const voices = page.getByTestId('tev-voices')
    await expect(
      voices,
      'cloud_thin failed to populate TopicEntityView tev-voices — ' +
        'either MENTIONS_PERSON or ABOUT regressed at LLM emit time, ' +
        'or who_said relational query path broke.',
    ).toBeVisible({ timeout: 10_000 })
    expect(
      await voices.getByTestId('tev-voice-row').count(),
      'tev-voices empty for a real Topic under cloud_thin',
    ).toBeGreaterThan(0)

    // tev-entities reads entities_in_topic — MENTIONS_PERSON/ORG ∩ ABOUT.
    const entities = page.getByTestId('tev-entities')
    await expect(
      entities,
      'cloud_thin failed to populate TopicEntityView tev-entities — ' +
        'either MENTIONS_PERSON/ORG or ABOUT regressed at LLM emit time.',
    ).toBeVisible({ timeout: 10_000 })
    expect(
      await entities.getByTestId('tev-entity-chip').count(),
      'tev-entities empty for a real Topic under cloud_thin',
    ).toBeGreaterThan(0)

    // Click a tev-voice-link → Person Landing renders for that person.
    // Closes the chain: TopicEntityView → who_said → Person Landing.
    await voices.getByTestId('tev-voice-link').first().click()
    await expect(page.getByTestId('person-landing-view')).toBeVisible({ timeout: 5_000 })

    if (consoleErrors.length) {
      const fatal = consoleErrors.filter(
        (e) =>
          !/HMR|deprecated|Vite|dmn_chk.*invalid domain|rejected for invalid domain/i.test(
            e,
          ) && !/"notify",\s*\w+ is null/i.test(e),
      )
      expect(
        fatal,
        `console errors during cloud_thin TopicEntityView walk:\n${fatal.join('\n')}`,
      ).toEqual([])
    }
  })
})
