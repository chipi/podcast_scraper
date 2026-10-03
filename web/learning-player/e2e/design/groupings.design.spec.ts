/**
 * Where a THEME and a STORYLINE show up once both are first-class — Library and Boards.
 *
 * Making themes a real kind touched five surfaces, and three of them had gaps that no unit test
 * could see because they are about what a person finds on a page:
 *
 *   Library › Following   themes and storylines shared one "Storylines" heading and one chip
 *   Library › Saved       a saved STORYLINE was a dead row (`entityRoute` returned null for it)
 *   Boards                neither grouping had a kind label, so a collected one showed a raw key
 *   Boards › Notes        a theme note listed but had no filter chip
 *
 * These shots exist to check those by eye. The design identity starts EMPTY — no follows, no
 * saves, no boards — so everything is seeded through the API first; without that the surfaces
 * render their empty states and the shot proves nothing while looking fine.
 *
 * Not an assertion suite. It produces images; a person decides.
 */
import { expect, test, type Page } from '@playwright/test'

const VARIANT = process.env.DESIGN_VARIANT || process.env.DESIGN_DIRECTION || 'baseline'
const dir = (name: string) =>
  `design-results/${VARIANT}/${test.info().project.name}/${name}.png`

/** From `tests/fixtures/app-validation-corpus/v3` — read out of the artifacts, not invented. */
const THEME = 'tc:safety-practices'
const STORYLINE_ANCHOR = 'topic:risk-management' // `/storyline/:id` routes by anchor topic
const TOPIC = 'topic:systems-thinking'

/**
 * A fresh identity per TEST, not per module.
 *
 * This spec MUTATES state — follows, saves, boards, notes — so a shared identity accumulates. A
 * module-level constant was shared by all three tests, which each seed: the Boards shot came back
 * with THREE identical "Risk & rates" boards and nine notes where there should have been one and
 * three. The `queue-2026-09-23` spec hit the version of this that spans runs; this is the version
 * that spans tests in one run.
 */
let seq = 0
function freshIdentity(): string {
  return `design-groupings-${Date.now()}-${seq++}`
}

async function signIn(page: Page): Promise<string> {
  const who = freshIdentity()
  await page.goto(`/api/app/auth/login?as=${who}`)
  await page.goto('/')
  return who
}

async function settle(page: Page): Promise<void> {
  await page.waitForLoadState('networkidle').catch(() => undefined)
  await page.evaluate(() => document.fonts?.ready).catch(() => undefined)
  await page.waitForTimeout(400)
}

async function shoot(page: Page, name: string): Promise<void> {
  await settle(page)
  await page.screenshot({ path: dir(`${name}-full`), fullPage: true })
  await page.screenshot({ path: dir(`${name}-viewport`), fullPage: false })
}

/** Follow + save + collect + annotate BOTH groupings and a topic, so the three sit side by side. */
async function seed(page: Page, ids: { theme: string; storyline: string; topic: string }) {
  await page.evaluate(async (x) => {
    const post = (u: string, body: unknown, method = 'POST') =>
      fetch(u, {
        method,
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

    // FOLLOW all three kinds — the point of the Following shot is that they are three groups now.
    await post('/api/app/interests', { items: [x.topic, x.theme, 'thc:managing-risk'] }, 'PUT')

    // SAVE both groupings. A saved storyline used to render as a row that could not be opened.
    await post(
      '/api/app/favorites',
      { kind: 'theme', ref: x.theme, label: 'safety practices' },
      'PUT',
    )
    await post(
      '/api/app/favorites',
      { kind: 'storyline', ref: x.storyline, label: 'Managing risk across domains' },
      'PUT',
    )
    await post('/api/app/favorites', { kind: 'topic', ref: x.topic, label: 'systems thinking' }, 'PUT')

    // COLLECT both into a board — neither kind was collectable at all until the contract admitted
    // them, and neither had a kind label to render once it was.
    const board = await post('/api/app/collections', { name: 'Risk & rates' }).then((r) => r.json())
    const cid = board?.id ?? board?.collection?.id
    if (cid) {
      await post(`/api/app/collections/${cid}/items`, { kind: 'theme', ref: x.theme })
      await post(`/api/app/collections/${cid}/items`, { kind: 'storyline', ref: x.storyline })
      await post(`/api/app/collections/${cid}/items`, { kind: 'topic', ref: x.topic })
    }

    // NOTE on each, so the notes strip offers a chip per kind.
    for (const [target, target_id, text] of [
      ['theme', x.theme, 'These all say the same thing in different words.'],
      ['storyline', x.storyline, 'This thread keeps resurfacing across shows.'],
      ['topic', x.topic, 'Worth tracking how this develops.'],
    ]) {
      await post('/api/app/notes', { target, target_id, text })
    }
  }, ids)
}

test('library-following', async ({ page }) => {
  await signIn(page)
  await seed(page, { theme: THEME, storyline: STORYLINE_ANCHOR, topic: TOPIC })
  // `?tab=shows`, not `following` — the tab's KEY is `shows` while its label is "Following"
  // (TAB_KEYS in LibraryView). An unrecognised key silently falls back to Saved, so this spec
  // shot the wrong tab and only failed because the testid assertion caught it.
  await page.goto('/library?tab=shows')
  // Proof the seed landed: an empty Following renders its empty state, and a shot of that would
  // look like a working page with nothing followed.
  await expect(page.getByTestId('followed-interests')).toBeVisible()
  await shoot(page, 'grouping-1-library-following')
})

test('library-saved', async ({ page }) => {
  await signIn(page)
  await seed(page, { theme: THEME, storyline: STORYLINE_ANCHOR, topic: TOPIC })
  await page.goto('/library?tab=saved')
  await shoot(page, 'grouping-2-library-saved')
})

test('boards', async ({ page }) => {
  await signIn(page)
  await seed(page, { theme: THEME, storyline: STORYLINE_ANCHOR, topic: TOPIC })
  // Boards is a Library TAB, keyed `collections`.
  await page.goto('/library?tab=collections')
  await shoot(page, 'grouping-3-boards')
})
