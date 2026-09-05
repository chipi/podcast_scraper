import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

/**
 * Home sections + corpus-search entry — REAL API over the COMMITTED validation corpus, NO mocks.
 * Asserts the What's-new + All-shows sections (real shows from the backend) and that the search
 * entry routes to /search and returns REAL grounded results against the two-tier index that
 * e2e/globalSetup.ts builds for the committed corpus.
 *
 * Shows come from the committed corpus: "Long Horizon Notes" (p05), "Practical Systems" (p02),
 * "Below the Surface" (p03).
 */
test('Home shows sections; search routes to /search and returns grounded results', async ({
  page,
}) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: "What's new" })).toBeVisible()

  // "Your shows" is per-user since #1585 — it lists the shows you FOLLOW, not the corpus. Signed
  // out there is nothing to show, and it must NOT fall back to the catalogue, which is what the
  // section did when this assertion was written against "All shows".
  await expect(page.getByRole('heading', { name: 'Your shows' })).toHaveCount(0)

  // Show names still reach Home via the trending-shows rail, which is corpus-wide.
  await expect(page.getByTestId('trending-shows-rail')).toBeVisible()

  const homeAxe = await new AxeBuilder({ page }).analyze()
  expect(homeAxe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious')).toEqual(
    [],
  )

  // Search entry → /search → real grounded results against the two-tier index (globalSetup builds
  // it; the warmup.setup project warms the serve's embedding model, so the summary line — which
  // renders only when results exist — is deterministic).
  await page.getByLabel('Ask across every episode').fill('investing')
  await page.getByRole('button', { name: 'Search', exact: true }).first().click()
  await expect(page).toHaveURL(/\/search\?q=investing/)
  await expect(page.getByText(/\d+ passages across \d+ episodes/)).toBeVisible()
})

/**
 * The hero's topic chips are real, tappable, and run the search they name (#1964).
 *
 * They exist because the `topic`-toned kicker was the only topic-coloured thing on Home, so a token
 * meaning "this is a topic" was carrying no meaning — and because the hero asked you to search
 * across every episode and then offered an empty box you had to already know what to type into.
 *
 * Asserted as ABSENT-OR-WORKING rather than always-present: they are sourced from
 * `getTrendingTopics()` and the contract is that a corpus with no velocity data renders no chips
 * rather than placeholders. A test demanding they always appear would encode the opposite contract.
 */
test('a hero topic chip runs its own search', async ({ page }) => {
  // Branch on the API RESPONSE, never on the rendered chip count. Counting chips cannot tell
  // "the corpus has no velocity data" apart from "the chips are broken" — both give zero, and an
  // early return on zero makes the test unfailable, which is how the first version of this passed
  // in 528ms while asserting nothing at all. The server states which case it is, so ask it.
  const trending = page.waitForResponse((r) => r.url().includes('/corpus/trending-topics'))
  await page.goto('/')
  const payload = await (await trending).json()
  const expected: string[] = (payload.topics ?? [])
    .slice(0, 4)
    .map((t: { topic_id: string; topic_label?: string }) => t.topic_label || t.topic_id.split(':').pop())
    .filter(Boolean)

  const chips = page.getByTestId('home-topic-chip')

  if (!payload.has_velocity_data || expected.length === 0) {
    // The documented empty contract: no data renders NO chips — not placeholders, not a shell.
    await expect(page.getByTestId('home-topic-chips')).toHaveCount(0)
    return
  }

  // Data exists, so the chips must exist. This is the assertion the element-count version could
  // never make, and the one that fails if the hero stops rendering them.
  await expect(chips, 'the corpus has velocity data, so the hero must offer chips').toHaveCount(
    expected.length,
  )
  await expect(chips.first()).toHaveText(expected[0]!)

  await chips.first().click()
  await expect(page).toHaveURL(/\/search\?/)
  await expect(page.getByRole('searchbox')).toHaveValue(expected[0]!)
})
