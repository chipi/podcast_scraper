/**
 * The shape every Discover rail shares (ADR-163): cards with their reason, hidden when empty, and
 * measured — shown once, each tap with its rail and action.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const tracked: { name: string; props: unknown }[] = []
vi.mock('../services/analytics', async (orig) => ({
  ...(await orig<typeof import('../services/analytics')>()),
  track: (name: string, props: unknown) => tracked.push({ name, props }),
}))

import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import type { TrendingEpisodeCard } from '../services/api'
import RecommendationRail from './RecommendationRail.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'browse', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

const card = (slug: string): TrendingEpisodeCard =>
  ({
    episode: { slug, title: `Title ${slug}`, podcast_title: 'A show with a long name' },
    why: {
      kind: 'captured',
      quote: { text: 'Said it.', episode_slug: slug, speaker: 'Ann', start_ms: 61_000, insight_id: null },
      subject: { kind: 'topic', id: 'topic:x', label: 'X' },
      capture: { kind: 'highlight', episode_slug: 'ep-0', text: 'kept', start_ms: 1 },
      events: null,
      outside_habits: false,
    },
  }) as unknown as TrendingEpisodeCard

async function mountRail(load: () => Promise<TrendingEpisodeCard[]>) {
  const w = mount(RecommendationRail, {
    props: {
      rail: 'captured',
      title: 'More from what you captured',
      load,
      scope: 'mine',
      cardsTestid: 'captured-cards',
      cardTestid: 'captured-card',
      openTestid: 'captured-card-open',
    },
    global: { plugins: [i18n, router] },
  })
  await flushPromises()
  return w
}

beforeEach(() => {
  tracked.length = 0
})

describe('RecommendationRail', () => {
  it('renders each card with its episode and its reason, and reports the rail once', async () => {
    const w = await mountRail(async () => [card('a'), card('b')])
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(2)
    expect(w.get('[data-testid="captured-card-open"]').attributes('href')).toBe('/episode/a')
    expect(w.get('[data-testid="why-reason"]').text()).toBe('Because you highlighted “kept”')
    expect(tracked.filter((e) => e.name === 'discover_rail_shown')).toHaveLength(1)
  })

  it('a tap reports the rail, the action and the rank', async () => {
    const w = await mountRail(async () => [card('a'), card('b')])
    await w.findAll('[data-testid="captured-card-open"]')[1].trigger('click')
    await w.findAll('[data-testid="why-play-from"]')[0].trigger('click')
    expect(tracked.filter((e) => e.name === 'discover_rail_tap').map((e) => e.props)).toEqual([
      expect.objectContaining({ rail: 'captured', action: 'open', rank: '2-3' }),
      expect.objectContaining({ rail: 'captured', action: 'play_from', rank: '1' }),
    ])
  })

  it('hides itself with nothing to show, or when loading fails', async () => {
    expect((await mountRail(async () => [])).find('[data-testid="captured-cards"]').exists()).toBe(false)
    const failing = await mountRail(async () => {
      throw new Error('offline')
    })
    expect(failing.find('[data-testid="captured-cards"]').exists()).toBe(false)
    expect(tracked.filter((e) => e.name === 'discover_rail_shown')).toHaveLength(0)
  })
})
