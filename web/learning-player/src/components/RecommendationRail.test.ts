/**
 * The shape every Discover rail shares (ADR-163): cards with their reason, measured, and paged on
 * the SERVER (operator 2026-10-11) — "Show more" asks for the next page, never reveals a hidden one.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const tracked: { name: string; props: unknown }[] = []
const lessApi = vi.hoisted(() => ({ suggestLess: vi.fn(async () => []), undoSuggestLess: vi.fn(async () => []) }))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...lessApi }))
vi.mock('../services/analytics', async (orig) => ({
  ...(await orig<typeof import('../services/analytics')>()),
  track: (name: string, props: unknown) => tracked.push({ name, props }),
}))

import { createPinia, setActivePinia } from 'pinia'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import type { DiscoverRail, TrendingEpisodeCard } from '../services/api'
import RecommendationRail from './RecommendationRail.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'browse', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
  ],
})

const card = (slug: string, subject = 'topic:x', label = 'X'): TrendingEpisodeCard =>
  ({
    episode: { slug, title: `Title ${slug}`, podcast_title: 'A show with a long name', feed_id: 'p01' },
    why: {
      kind: 'captured',
      quote: { text: 'Said it.', episode_slug: slug, speaker: 'Ann', start_ms: 61_000, insight_id: null },
      subject: { kind: 'topic', id: subject, label },
      capture: { kind: 'highlight', episode_slug: 'ep-0', text: 'kept', start_ms: 1 },
      events: null,
    },
  }) as unknown as TrendingEpisodeCard

const page = (items: TrendingEpisodeCard[], total: number, fallback = false): DiscoverRail => ({
  rail: 'captured',
  items,
  total,
  fallback,
})

async function mountRail(first: DiscoverRail, more = vi.fn(async () => page([], first.total))) {
  const w = mount(RecommendationRail, {
    props: {
      rail: 'captured',
      title: 'More from what you captured',
      page: first,
      more,
      scope: 'mine',
      cardsTestid: 'captured-cards',
      cardTestid: 'captured-card',
      openTestid: 'captured-card-open',
    },
    slots: { 'fallback-note': '<p data-testid="note">Everyone’s</p>' },
    global: { plugins: [i18n, router, createPinia()] },
  })
  await flushPromises()
  return { w, more }
}

beforeEach(() => {
  setActivePinia(createPinia())
  tracked.length = 0
  lessApi.suggestLess.mockClear()
  lessApi.undoSuggestLess.mockClear()
})

describe('RecommendationRail', () => {
  it('renders the first page with each reason, the whole card a way into the episode, and reports the rail once', async () => {
    const { w } = await mountRail(page([card('a'), card('b')], 2))
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(2)
    expect(w.get('[data-testid="captured-card-open"]').attributes('href')).toBe('/episode/a')
    // The title's link stretches over the card; the show is named, not linked (operator 2026-10-11).
    expect(w.get('[data-testid="captured-card-open"]').classes()).toContain('after:inset-0')
    expect(w.find('a[href^="/podcast/"]').exists()).toBe(false)
    expect(w.get('[data-testid="why-reason"]').text()).toBe('Because you highlighted “kept”')
    expect(tracked.filter((e) => e.name === 'discover_rail_shown')).toHaveLength(1)
    expect(w.find('[data-testid="show-all-toggle"]').exists()).toBe(false) // nothing more to page
  })

  it('"Show more" asks the server for the next page and appends it; "Show less" returns', async () => {
    const more = vi.fn(async (offset: number, limit: number) => {
      expect([offset, limit]).toEqual([2, 4])
      return page([card('c', 'topic:y'), card('d', 'topic:y')], 4)
    })
    const { w } = await mountRail(page([card('a'), card('b')], 4), more)
    const toggle = w.get('[data-testid="show-all-toggle"]')
    expect(toggle.text()).toBe('Show more (2)')
    await toggle.trigger('click')
    await flushPromises()
    expect(more).toHaveBeenCalledTimes(1)
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(4)
    expect(w.get('[data-testid="show-all-toggle"]').text()).toBe('Show less')
    await w.get('[data-testid="show-all-toggle"]').trigger('click')
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(2)
  })

  it('a tap reports the rail, the action and the rank', async () => {
    const { w } = await mountRail(page([card('a'), card('b')], 2))
    await w.findAll('[data-testid="captured-card-open"]')[1].trigger('click')
    await w.findAll('[data-testid="why-play-from"]')[0].trigger('click')
    expect(tracked.filter((e) => e.name === 'discover_rail_tap').map((e) => e.props)).toEqual([
      expect.objectContaining({ rail: 'captured', action: 'open', rank: '2-3' }),
      expect.objectContaining({ rail: 'captured', action: 'play_from', rank: '1' }),
    ])
  })

  it('every card carries the standard episode actions (save, queue, ⋯)', async () => {
    const { w } = await mountRail(page([card('a')], 1))
    expect(w.findComponent({ name: 'EpisodeActions' }).exists()).toBe(true)
  })

  it('cold start: the server-flagged fallback is labelled and measured as Everyone, without "Less like this"', async () => {
    const { w } = await mountRail(page([card('e1')], 1, true))
    expect(w.find('[data-testid="note"]').exists()).toBe(true)
    expect(w.find('[data-testid="rail-less"]').exists()).toBe(false)
    expect(tracked.find((e) => e.name === 'discover_rail_shown')?.props).toMatchObject({ scope: 'everyone' })
  })

  it("the listener's own cards carry no fallback note", async () => {
    const { w } = await mountRail(page([card('a')], 1))
    expect(w.find('[data-testid="note"]').exists()).toBe(false)
    expect(tracked.find((e) => e.name === 'discover_rail_shown')?.props).toMatchObject({ scope: 'mine' })
  })

  it('"Less like this" hides every card with that reason, records it, and Undo asks the page to reload', async () => {
    const { w } = await mountRail(page([card('a'), card('b'), card('c', 'topic:y', 'Y')], 3))
    await w.findAll('[data-testid="rail-less"]')[0].trigger('click')
    await flushPromises()
    expect(lessApi.suggestLess).toHaveBeenCalledWith('topic:x', 'X')
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(1)
    expect(w.get('[data-testid="rail-less-done"]').text()).toContain('Less about X from now on.')
    expect(tracked.filter((e) => e.name === 'discover_rail_tap').map((e) => e.props)).toEqual([
      expect.objectContaining({ action: 'less', rank: '1' }),
    ])
    await w.get('[data-testid="rail-less-undo"]').trigger('click')
    await flushPromises()
    expect(lessApi.undoSuggestLess).toHaveBeenCalledWith('topic:x')
    expect(w.emitted('changed')).toHaveLength(1)
  })

  it('when "Less like this" removes the last card, Undo stays reachable', async () => {
    const { w } = await mountRail(page([card('a')], 1))
    await w.get('[data-testid="rail-less"]').trigger('click')
    await flushPromises()
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(0)
    expect(w.find('[data-testid="rail-less-undo"]').exists()).toBe(true)
  })

  it('a new first page (a reload, a scope switch) replaces the cards and counts a new impression', async () => {
    const { w } = await mountRail(page([card('a'), card('b')], 2))
    await w.setProps({ page: page([card('z')], 1) })
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(1)
    expect(tracked.filter((e) => e.name === 'discover_rail_shown')).toHaveLength(2)
  })
})
