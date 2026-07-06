import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * EntitySignals enricher rows (#1150 — e2e parity with #1146's perspectives.spec).
 *
 * The per-enricher signal *rendering* rows (momentum / similar for a topic) were vitest-only.
 * This drives them in a real browser: mock the corpus-enrichment envelope (the committed
 * validation corpus carries thin signals), open a topic entity card, and assert the rows.
 * EntitySignals filters the envelope by the *opened* topic id, so we capture that id from the
 * card's own request and key the mocked signals to it.
 */
test('topic entity card renders momentum + similar enricher signal rows', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'entity-signals', testInfo)

  let resolveId: (v: string) => void = () => {}
  const topicId = new Promise<string>((r) => {
    resolveId = r
  })

  // Capture the opened topic id from the topic-card request (not the /perspectives sub-path).
  await page.route('**/api/app/topics/*', async (route) => {
    const url = route.request().url()
    const m = url.match(/\/topics\/([^/?]+)(?:\?|$)/)
    if (m && !url.includes('/perspectives')) resolveId(decodeURIComponent(m[1]))
    await route.continue()
  })

  // Serve the enrichment envelope keyed to whichever topic the card opened.
  await page.route('**/api/app/corpus/enrichment', async (route) => {
    const tid = await topicId
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        signals: {
          temporal_velocity: {
            topics: [{ topic_id: tid, topic_label: 'x', velocity_last_over_6mo: 2.6, total: 40 }],
          },
          topic_similarity: {
            topics: [
              {
                topic_id: tid,
                top_k: [
                  { topic_id: 'topic:machine-learning', topic_label: 'Machine Learning', similarity: 0.9 },
                  { topic_id: 'topic:llms', topic_label: 'LLMs', similarity: 0.8 },
                ],
              },
            ],
          },
        },
      }),
    })
  })

  // Open an episode → Insights → click a topic chip → the topic entity card opens.
  await page.goto('/')
  await page.getByText('Index Investing Without the Myths').first().click()
  await page.getByRole('button', { name: 'Insights' }).first().click()
  await page.locator('button.text-topic').first().click()

  // The enricher signal rows render from the mocked envelope.
  await expect(page.getByTestId('es-momentum')).toBeVisible()
  const similar = page.getByTestId('es-similar')
  await expect(similar).toBeVisible()
  await expect(similar.getByText('Machine Learning')).toBeVisible()
})
