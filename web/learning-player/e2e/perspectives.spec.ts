import { expect, test } from '@playwright/test'

import { signInIsolated } from './helpers'

/**
 * Multi-perspective synthesis (#1146) — the topic card's Perspectives section.
 *
 * The committed validation corpus has no speaker-attributed cross-topic insights (perspectives
 * are near-absent at that scale), so the perspectives endpoint is mocked via route interception
 * (as elsewhere in this suite) to exercise the UI end-to-end: render per-speaker groups, the
 * per-speaker "show more" toggle, and speaker → person navigation.
 */
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

test('topic card shows per-speaker perspectives with show-more + speaker nav', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'perspectives', testInfo)

  // Mock the perspectives endpoint before any topic card opens (any topic id → the same set).
  await page.route('**/api/app/topics/**/perspectives**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        topic_id: 'topic:mock',
        topic_label: 'mock',
        perspective_count: 2,
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
          {
            person_id: 'person:amy-ng',
            person_name: 'Amy Ng',
            insight_count: 1,
            episode_count: 1,
            insights: [insight('a1', 'Amy perspective insight')],
          },
        ],
      }),
    })
  })

  // Open an episode → Insights → click a topic chip → the topic entity card opens.
  await page.goto('/')
  await page.getByText('Index Investing Without the Myths').first().click()
  await page.getByRole('button', { name: 'Insights' }).first().click()
  await page.locator('button.text-topic').first().click()

  // The Perspectives section renders the mocked speakers, capped at 3 insights.
  const section = page.getByTestId('topic-perspectives')
  await expect(section).toBeVisible()
  await expect(section.getByText('2 perspectives')).toBeVisible()
  await expect(section.getByRole('button', { name: 'Jack Clark' })).toBeVisible()
  await expect(section.getByText('Perspective insight three')).toBeVisible()
  await expect(section.getByText('Perspective insight four')).toBeHidden()

  // "Show more" reveals the 4th insight.
  await section.getByRole('button', { name: 'Show 1 more' }).click()
  await expect(section.getByText('Perspective insight four')).toBeVisible()

  // Tapping a speaker navigates the card to that person (perspectives are topic-only → gone).
  await section.getByRole('button', { name: 'Jack Clark' }).click()
  await expect(page.getByTestId('topic-perspectives')).toHaveCount(0)
})
