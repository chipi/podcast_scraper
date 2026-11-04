import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'
import { readFsmEventLog, resetFsmEventLog } from './handoff/_handoff-helpers'

/**
 * UXS-007 "Action buttons": the topic view offers **View in graph** (Graph tab, topic node
 * focused) and **Search this topic** (Search prefilled with the topic label).
 *
 * Driven the way `topic-entity-view.spec.ts`'s live contract test drives it — a real topic from
 * the corpus, focused through the DEV `__GIKG_SUBJECT__.focusTopic` hook — but from
 * the DIGEST tab, because both actions are about leaving where you are.
 */

type Topic = { topic_id: string; label: string }

async function openTopicRailOnDigest(page: Page): Promise<Topic> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await expect(page.getByTestId('digest-root')).toBeVisible({ timeout: 30_000 })
  const leaders = (await (await page.request.get('/api/topics/perspective-leaders?limit=20')).json()) as {
    topics: { topic_id: string; topic_label: string }[]
  }
  const row = leaders.topics.find((t) => t.topic_label?.trim())
  expect(row, 'expected the corpus to expose at least one labelled topic').toBeTruthy()
  const topic: Topic = { topic_id: row!.topic_id, label: row!.topic_label }
  await page.evaluate((id) => {
    const subj = (window as unknown as { __GIKG_SUBJECT__?: { focusTopic: (i: string) => void } })
      .__GIKG_SUBJECT__
    if (!subj?.focusTopic) throw new Error('__GIKG_SUBJECT__.focusTopic not exposed (DEV-only)')
    subj.focusTopic(id)
  }, topic!.topic_id)
  await expect(page.getByTestId('topic-entity-view')).toBeVisible({ timeout: 15_000 })
  return topic!
}

test.describe('Topic Entity View actions (UXS-007)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('Search this topic: the topic rail prefills Search with the topic label', async ({
    page,
  }) => {
    const topic = await openTopicRailOnDigest(page)
    const searchReq = page.waitForRequest((r) => new URL(r.url()).pathname === '/api/search')
    await page.getByTestId('node-detail-topic-prefill-search').click()

    await expect(page.getByTestId('search-workspace')).toBeVisible()
    await expect(page.getByTestId('digest-root')).toBeHidden()
    await expect(page.locator('#search-q')).toHaveValue(
      new RegExp(`^${topic.label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'i'),
    )
    const q = new URL((await searchReq).url()).searchParams.get('q') ?? ''
    expect(q.toLowerCase()).toBe(topic.label.toLowerCase())
  })

  /**
   * NodeDetail's topic shortcut row owns "View in graph" (`node-detail-topic-view-in-graph`, shown
   * only off the Graph tab). `TopicEntityView`'s own action row (`topic-entity-view-go-graph`) is
   * `v-if="!embedded"` and its only mount is embedded, so that testid never renders.
   */
  test('View in graph: the topic rail opens the Graph with that topic selected', async ({
    page,
  }) => {
    const topic = await openTopicRailOnDigest(page)
    await resetFsmEventLog(page)
    await page.getByTestId('node-detail-topic-view-in-graph').click({ timeout: 10_000 })
    // The Graph is asked for THIS topic (a topic handoff carrying its id) — which is what brings a
    // topic outside the default time window onto the canvas. A bare tab switch also selects the
    // topic when it happens to be in the default slice, so the selection below cannot prove this.
    await expect
      .poll(async () =>
        (await readFsmEventLog(page)).some(
          (e) => e.type === 'handoffRequested' && e.envelope?.kind === 'topic' && e.envelope?.cyId === topic.topic_id,
        ),
      )
      .toBe(true)
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
    // The canvas id carries its layer prefix (`g:topic:X` / `k:topic:X`), which the handoff helpers
    // normalise to `topic:X` the same way (`handoff/_handoff-helpers.ts`).
    await page.waitForFunction(
      (id) => {
        const cy = (
          window as unknown as {
            __GIKG_CY_DEV__?: { nodes: (s: string) => { map: (f: (n: { id: () => string }) => string) => string[] } }
          }
        ).__GIKG_CY_DEV__
        const selected = cy?.nodes(':selected').map((n) => n.id().replace(/^[gk]:/, '')) ?? []
        return selected.includes(id)
      },
      topic.topic_id,
      { timeout: 15_000 },
    )
  })
})
