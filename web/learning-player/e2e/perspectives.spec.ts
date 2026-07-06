import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * Multi-perspective synthesis (#1146) — the topic card's Perspectives section.
 *
 * The committed validation corpus now carries real speaker-attributed insights on a shared
 * topic: the panel episode "The Risk Panel: Diversify or Concentrate?" (p05_e04) plus the
 * risk-management insights threaded across the corpus give topic:risk-management seven distinct
 * speakers — Daniel Cho arguing diversification against Scott Bessent's concentration, and
 * others. The first test exercises this end-to-end against the REAL backend (no mocks), matching
 * the suite's full-stack contract. Only the per-speaker "show more" cap — which the corpus scale
 * can't reach (≤2 insights per speaker on any topic) — is covered with a focused mock below.
 */

test('topic card shows real per-speaker perspectives from the corpus + speaker nav', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'perspectives', testInfo)

  // Open the panel episode → Insights → the contested "risk management" topic chip.
  await page.goto('/')
  await page.goto('/podcast/p05') // #1148: reach the episode via its show page (date-independent)
  await page.getByText('The Risk Panel: Diversify or Concentrate?').first().click()
  await page.getByRole('button', { name: 'Insights' }).first().click()
  await page.locator('button.text-topic').filter({ hasText: 'risk management' }).first().click()

  // The Perspectives section renders the real, corpus-derived speakers.
  const section = page.getByTestId('topic-perspectives')
  await expect(section).toBeVisible()
  await expect(section.getByText('7 perspectives')).toBeVisible()
  await expect(section.getByRole('button', { name: 'Daniel Cho' })).toBeVisible()
  await expect(section.getByRole('button', { name: 'Scott Bessent' })).toBeVisible()
  // The engineered opposition surfaces verbatim as a grounded claim.
  await expect(
    section.getByText(/Diversification is the only real risk control/),
  ).toBeVisible()

  // Tapping a speaker navigates the card to that person (perspectives are topic-only → gone).
  await section.getByRole('button', { name: 'Daniel Cho' }).click()
  await expect(page.getByTestId('topic-perspectives')).toHaveCount(0)
})

function insight(id: string, text: string) {
  return {
    id,
    text,
    grounded: true,
    insight_type: 'claim',
    confidence: null,
    position_hint: null,
    quotes: [],
  }
}

test('per-speaker show-more toggle reveals insights past the preview cap', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'perspectives-showmore', testInfo)

  // Mock a >3-insight speaker: the preview cap (3) + "show more" toggle is a client behavior the
  // real corpus can't reach (≤2 insights/speaker), so this one case fakes the endpoint to cover it.
  await page.route('**/api/app/topics/**/perspectives**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        topic_id: 'topic:mock',
        topic_label: 'mock',
        perspective_count: 1,
        perspectives: [
          {
            person_id: 'person:jack-clark',
            person_name: 'Jack Clark',
            insight_count: 4,
            episode_count: 2,
            insights: [
              insight('i1', 'Perspective insight one'),
              insight('i2', 'Perspective insight two'),
              insight('i3', 'Perspective insight three'),
              insight('i4', 'Perspective insight four'),
            ],
          },
        ],
      }),
    })
  })

  await page.goto('/')
  await page.goto('/podcast/p05')
  await page.getByText('The Risk Panel: Diversify or Concentrate?').first().click()
  await page.getByRole('button', { name: 'Insights' }).first().click()
  await page.locator('button.text-topic').filter({ hasText: 'risk management' }).first().click()

  const section = page.getByTestId('topic-perspectives')
  await expect(section).toBeVisible()
  await expect(section.getByText('Perspective insight three')).toBeVisible()
  await expect(section.getByText('Perspective insight four')).toBeHidden()

  // "Show more" reveals the 4th insight.
  await section.getByRole('button', { name: 'Show 1 more' }).click()
  await expect(section.getByText('Perspective insight four')).toBeVisible()
})
