import { expect, test } from '@playwright/test'
import { openTranscript, signInIsolated } from './helpers'

/**
 * P3 Consolidation end-to-end — REAL API over the COMMITTED validation corpus (now carrying RFC-088
 * enrichment envelopes), NO mocks. Covers the consumer enrichment read surface (#1121), the Recall
 * scope toggle (#1124), the "your corpus" entity lens (#1125), and the Revisit inbox (#1125).
 * Per-user state is the gitignored APP_DATA_DIR.
 */
test('enrichment read surface + recall toggle + your-corpus lens + Revisit inbox', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'consolidation', testInfo)

  // Open an episode and capture its slug from the URL.
  await page.goto('/')
  await page.goto('/podcast/p09') // #1148: open a RECENT episode so it isn't due for resurfacing
  await page.getByText('Risk Is a Systems Property').first().click()
  await openTranscript(page) // transcript is opt-in on mobile — reveal it (no-op on desktop)
  await expect(page.getByText(/Risk lives in the couplings/).first()).toBeVisible()
  const slug = new URL(page.url()).pathname.split('/').pop()!

  // #1121: the consumer enrichment read surface serves the committed envelopes.
  const epEnrich = await page.request.get(`/api/app/episodes/${slug}/enrichment`)
  expect(epEnrich.ok()).toBeTruthy()
  expect((await epEnrich.json()).signals).toHaveProperty('insight_density')
  const corpusEnrich = await page.request.get('/api/app/corpus/enrichment')
  expect((await corpusEnrich.json()).signals).toHaveProperty('temporal_velocity')

  // #1125: the entity-card "your listening" lens — open a topic chip from the Insights panel, then
  // toggle to My listening (the card refetches scoped to the heard set; it stays rendered).
  await page.getByRole('button', { name: 'Insights' }).first().click()
  // insight_density strip renders at the head of the Insights list (Plan B #2).
  await expect(page.getByTestId('episode-density')).toBeVisible()
  await page.getByTestId('kp-topic-chip').or(page.getByTestId('kp-person-chip')).first().click()
  // A radiogroup, not a tablist (#1594 item 7): the card's corpus scope re-queries the one card
  // body rather than switching between panels.
  const cardScope = page.getByRole('radiogroup', { name: 'Card scope' })
  await expect(cardScope).toBeVisible()
  // Library's tab strip is `Tabs.vue` now, so these are `role="tab"` (#1594 item 7). They
  // previously carried NO role at all — which is why `getByRole('button')` matched them, and
  // why the strip did not announce as tabs to anyone using one.
  await cardScope.getByRole('radio', { name: 'My listening' }).click()
  await expect(cardScope.getByRole('radio', { name: 'My listening' })).toHaveAttribute(
    'aria-checked',
    'true',
  )

  // #1124: Recall — switch the scope to "My listening". This isolated user has captured nothing, so the
  // scoped search is honest-empty. scope=mine short-circuits to an empty result set server-side
  // (app_search.py — the user's heard∪captured set is empty, so it never touches the corpus index
  // or embedding model), so the "Nothing in your listening" recall message is deterministic — no cold
  // index/model race to tolerate (the corpus index is built by e2e/globalSetup.ts anyway).
  await page.goto('/search?q=index')
  // Radiogroup, same reason as the card scope above: it re-runs the query into one results region.
  const searchScope = page.getByRole('radiogroup', { name: 'Search scope' })
  await expect(searchScope).toBeVisible()
  await searchScope.getByRole('radio', { name: 'My listening' }).click()
  await expect(page.getByText(/Nothing in your listening on this yet/)).toBeVisible()

  // #1125: the Revisit inbox — a fresh user has nothing due; the pacing control pauses.
  //
  // The pause PERSISTS, so this ran correctly exactly once per api container: on the next run the
  // tab opened already paused, showed "Resurfacing is paused." where the empty state belonged, and
  // failed on an assertion about a state the previous run had left behind. Resume first if needed,
  // so the spec starts from the state it claims to be testing, and restore that state at the end.
  await page.goto('/library')

  // WAIT FOR THE STATE, NOT FOR THE BUTTON.
  //
  // `ResurfacingInbox` renders its toggle immediately from `const paused = ref(false)` and only
  // then fetches the real flag, so every "is it paused?" check that races the response reads
  // `false` — whether it is `isVisible()` on the label or `getAttribute('aria-pressed')` on the
  // button. Both were tried; both skipped the reset and then asserted an empty state that the
  // arriving `paused: true` had already replaced. The only honest signal that the component knows
  // anything is the GET completing.
  const loaded = page.waitForResponse(
    (r) => r.url().includes('/resurfacing') && r.request().method() === 'GET',
  )
  await page.getByRole('tab', { name: 'Revisit' }).click()
  await loaded

  // One button whose LABEL flips (Pause <-> Resume), so match either and drive it by `aria-pressed`.
  const toggle = page.getByRole('button', { name: /^(Pause|Resume)$/ })
  await expect(toggle).toBeVisible()
  if ((await toggle.getAttribute('aria-pressed')) === 'true') {
    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-pressed', 'false')
  }

  await expect(page.getByText(/Nothing to revisit right now/)).toBeVisible()
  await toggle.click()
  await expect(page.getByText('Resurfacing is paused.')).toBeVisible()

  // Leave no trace: the next run of this spec must meet the same unpaused account this one did.
  await toggle.click()
  await expect(toggle).toHaveAttribute('aria-pressed', 'false')
  await expect(page.getByText(/Nothing to revisit right now/)).toBeVisible()
})
