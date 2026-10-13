import { mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * #2344 and #2345, through the real app and API.
 *
 * The summaries and the reference flags are written by the board-summarizer sidecar, which a
 * browser run does not have (it calls a model). These specs write what it writes — the same
 * files, in the API's data dir — and assert everything from there on: the API reads them, the
 * pages show them, the actions work.
 */

const STATE = fileURLToPath(new URL('./.app-state', import.meta.url))
const STORYLINE = 'thc:managing-risk'
const ANCHOR = 'topic:risk-management'
const SUMMARY = 'A thread on naming the risk before it names you, across finance, diving and racing.'

function writeSummary(): void {
  mkdirSync(STATE, { recursive: true })
  writeFileSync(
    `${STATE}/storyline_summaries.json`,
    JSON.stringify({
      [STORYLINE]: {
        text: SUMMARY,
        // Not the storyline's fingerprint: the page must then say the storyline grew since.
        fingerprint: 'written-before',
        model: 'e2e',
        prompt: 'e2e',
        created_at: Math.floor(Date.now() / 1000),
      },
    }),
  )
}

async function userId(page: Page): Promise<string> {
  const me = await page.request.get('/api/app/me')
  expect(me.ok()).toBe(true)
  return (await me.json()).user_id as string
}

test.describe('a storyline summary (#2344)', () => {
  test('leads the storyline page, opened by its anchor topic or its own id, and says it is stale', async ({
    page,
  }, testInfo) => {
    writeSummary()
    await signInIsolated(page, 'storyline-summary', testInfo)
    for (const id of [ANCHOR, STORYLINE]) {
      await page.goto(`/storyline/${encodeURIComponent(id)}`)
      const summary = page.getByTestId('storyline-summary')
      await expect(summary).toContainText(SUMMARY)
      await expect(page.getByTestId('storyline-summary-when')).toContainText('the storyline grew since')
    }
  })

  test('leads a storyline Learn path', async ({ page }, testInfo) => {
    writeSummary()
    await signInIsolated(page, 'storyline-summary-learn', testInfo)
    const started = await page.request.post('/api/app/discover/learn', { data: { subject_id: STORYLINE } })
    expect(started.ok()).toBe(true)
    await page.goto(`/learn?subject=${encodeURIComponent(STORYLINE)}`)
    await expect(page.getByTestId('learn-view-title')).toBeVisible()
    await expect(page.getByTestId('storyline-summary')).toContainText(SUMMARY)
  })

  test('a storyline is saved by its own id, not the topic its page is opened by', async ({ page }, testInfo) => {
    await signInIsolated(page, 'storyline-save-id', testInfo)
    await page.goto(`/storyline/${encodeURIComponent(ANCHOR)}`)
    await expect(page.getByTestId('storyline-view')).toBeVisible()
    await page.getByTestId('storyline-view').getByTestId('favorite-button').first().click()
    await expect
      .poll(async () => {
        const favs = (await (await page.request.get('/api/app/favorites')).json()) as {
          entities?: Array<{ kind: string; ref: string }>
        }
        return (favs.entities ?? []).filter((f) => f.kind === 'storyline').map((f) => f.ref)
      })
      .toEqual([STORYLINE])
  })
})

test.describe('a reference the corpus no longer has (#2345)', () => {
  test('a gone follow stays, muted and "No longer available", cannot be opened, and can be unfollowed', async ({
    page,
  }, testInfo) => {
    await signInIsolated(page, 'gone-follow', testInfo)
    const uid = await userId(page)
    const dir = `${STATE}/users/${uid}`
    mkdirSync(dir, { recursive: true })
    // What the reference pass leaves: the follow kept, and flagged.
    writeFileSync(`${dir}/interests.json`, JSON.stringify([STORYLINE, 'thc:dissolved-thread']))
    writeFileSync(
      `${dir}/reference_flags.json`,
      JSON.stringify({ gone: [{ ref: 'thc:dissolved-thread', kind: 'storyline' }], checked_at: 1 }),
    )

    await page.goto('/library?tab=shows')
    const list = page.getByTestId('followed-interests')
    const gone = list.locator('li', { has: page.getByTestId('gone-badge') })
    await expect(gone).toHaveCount(1)
    await expect(gone).toContainText('No longer available')
    await expect(gone.getByRole('button').first()).toBeDisabled()
    // The live follow next to it is untouched.
    await expect(list.locator('li').filter({ hasNot: page.getByTestId('gone-badge') })).not.toHaveCount(0)

    await gone.getByTestId('unfollow').click()
    await expect(list.getByTestId('gone-badge')).toHaveCount(0)
    // The unfollow is written in the background: poll the API until it has it.
    await expect
      .poll(async () => JSON.stringify(await (await page.request.get('/api/app/interests')).json()))
      .not.toContain('thc:dissolved-thread')
  })
})
