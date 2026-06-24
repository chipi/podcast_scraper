import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'

/**
 * C-slice flow e2e (#1082–#1085): catalog → open player → transcript + Knowledge Panel.
 * Backend fully stubbed via route mocks (no server, no real audio). Includes an axe a11y
 * pass on both surfaces (DoD: accessibility checked in e2e).
 */

const EP = {
  slug: 'show-abc123',
  title: 'How Memory Consolidates During Sleep',
  feed_id: 'show',
  podcast_title: 'Huberman Lab',
  publish_date: '2024-03-10',
  duration_seconds: 2880,
  episode_image_url: null,
  feed_image_url: null,
  artwork_url: null,
  status: 'ready',
  summary_preview: 'How sleep consolidates memory.',
  summary_title: 'Memory & sleep',
  summary_text: 'Sleep consolidates memory via hippocampal replay.',
  topics: ['memory', 'sleep'],
  has_transcript: true,
  has_summary: true,
  has_gi: true,
  has_kg: true,
  has_bridge: false,
}

async function stubBackend(page: Page): Promise<void> {
  const json = (body: unknown) => ({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
  const unauth = { status: 401, contentType: 'application/json', body: '{}' }

  // One deterministic handler keyed on the exact pathname (no glob/regex precedence games).
  await page.route(/\/api\/app\//, (route) => {
    const path = new URL(route.request().url()).pathname
    const base = `/api/app/episodes/${EP.slug}`
    if (path === '/api/app/me') return route.fulfill(unauth)
    if (path === '/api/app/episodes') return route.fulfill(json({ items: [EP], page: 1, page_size: 20, total: 1, has_more: false }))
    if (path === base) return route.fulfill(json(EP))
    if (path === `${base}/segments`)
      return route.fulfill(
        json({
          version: '1.0',
          episode_slug: EP.slug,
          segments: [
            { id: 'seg_0000', start: 0, end: 2.5, text: 'Hello and welcome.', speaker: 'person:matthew-walker' },
            { id: 'seg_0001', start: 2.5, end: 6, text: 'Today we discuss memory.', speaker: null },
          ],
        }),
      )
    if (path === `${base}/audio-source`)
      return route.fulfill(json({ episode_slug: EP.slug, url: 'data:audio/mpeg;base64,', mime: 'audio/mpeg', duration_seconds: 2880, media_id: null, strategy: 'direct', resolved_url: null, verified: null, content_length: null }))
    if (path === `${base}/insights`)
      return route.fulfill(
        json({
          episode_slug: EP.slug,
          insights: [
            {
              id: 'i1', text: 'Sleep spindles gate memory transfer.', grounded: true,
              insight_type: 'claim', confidence: null, position_hint: null,
              quotes: [{ text: 'the spindles act as a gate', speaker: 'person:matthew-walker', char_start: null, char_end: null, start_ms: 1000, end_ms: 4000 }],
            },
          ],
        }),
      )
    if (path === `${base}/entities`)
      return route.fulfill(json({ episode_slug: EP.slug, persons: [{ id: 'person:matthew-walker', name: 'Matthew Walker', kind: 'person' }], orgs: [], topics: [{ id: 'topic:memory', label: 'memory' }] }))
    if (path.startsWith('/api/app/playback/')) return route.fulfill(unauth)
    return route.fulfill({ status: 404, contentType: 'application/json', body: '{}' })
  })
}

test('catalog → player → transcript + knowledge panel, with a11y checks', async ({ page }) => {
  await stubBackend(page)

  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Your Library' })).toBeVisible()
  await expect(page.getByText(EP.title)).toBeVisible()

  const catalogAxe = await new AxeBuilder({ page }).analyze()
  expect(
    catalogAxe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious'),
  ).toEqual([])

  // Open the player.
  await page.getByText(EP.title).first().click()
  await expect(page.getByRole('heading', { name: EP.title })).toBeVisible()
  await expect(page.getByText('Today we discuss memory.')).toBeVisible()

  // Reveal the Knowledge Panel and confirm an insight is shown.
  await page.getByRole('button', { name: /insights/i }).first().click()
  await expect(page.getByText('Sleep spindles gate memory transfer.')).toBeVisible()

  const playerAxe = await new AxeBuilder({ page }).analyze()
  expect(
    playerAxe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious'),
  ).toEqual([])
})
