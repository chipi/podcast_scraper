import { expect, test, type Page } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * UXS-007 "Action buttons": the topic view offers **View in graph** (Graph tab, topic node
 * focused) and **Search this topic** (Search prefilled with the topic label).
 *
 * Driven the way `topic-entity-view.spec.ts`'s live contract test drives it — a real clustered
 * topic from the corpus, focused through the DEV `__GIKG_SUBJECT__.focusTopic` hook — but from
 * the DIGEST tab, because both actions are about leaving where you are.
 */

type Topic = { topic_id: string; label: string }

async function openTopicRailOnDigest(page: Page): Promise<Topic> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await expect(page.getByTestId('digest-root')).toBeVisible({ timeout: 30_000 })
  const clusters = (await (await page.request.get('/api/corpus/topic-clusters')).json()) as {
    clusters: { members: Topic[] }[]
  }
  const topic = clusters.clusters.flatMap((c) => c.members ?? []).find((m) => m.label?.trim())
  expect(topic, 'expected the corpus to expose at least one labelled clustered topic').toBeTruthy()
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
   * Located by role + name, not testid, so it holds whichever component ends up owning the button.
   * On 2026-10-05 nothing renders it: `TopicEntityView`'s action row (`topic-entity-view-go-graph`
   * / `-prefill-search`) is `v-if="!embedded"`, and its only mount is NodeDetail's embedded one;
   * NodeDetail supplies "Prefill semantic search" (covered above) but no "View in graph" — probed
   * live, the rail's buttons are "Prefill semantic search" and "Set Search topic filter" only.
   */
  test('View in graph: the topic rail opens the Graph with that topic selected', async ({
    page,
  }) => {
    const topic = await openTopicRailOnDigest(page)
    await page
      .getByTestId('graph-node-detail-rail')
      .getByRole('button', { name: /^(View|Open) in graph$/ })
      .click({ timeout: 10_000 })
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
    await page.waitForFunction(
      (id) => {
        const cy = (
          window as unknown as {
            __GIKG_CY_DEV__?: { $id: (i: string) => { nonempty: () => boolean; selected: () => boolean } }
          }
        ).__GIKG_CY_DEV__
        const n = cy?.$id(id)
        return Boolean(n?.nonempty() && n.selected())
      },
      topic.topic_id,
      { timeout: 15_000 },
    )
  })
})
