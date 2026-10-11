/**
 * The shape every Discover rail shares (ADR-163): cards with their reason, hidden when empty, and
 * measured — shown once, each tap with its rail and action.
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

const card = (slug: string, hasMoments = false): TrendingEpisodeCard =>
  ({
    episode: { slug, title: `Title ${slug}`, podcast_title: 'A show with a long name', has_moments: hasMoments },
    why: {
      kind: 'captured',
      quote: { text: 'Said it.', episode_slug: slug, speaker: 'Ann', start_ms: 61_000, insight_id: null },
      subject: { kind: 'topic', id: 'topic:x', label: 'X' },
      capture: { kind: 'highlight', episode_slug: 'ep-0', text: 'kept', start_ms: 1 },
      events: null,
      outside_habits: false,
    },
  }) as unknown as TrendingEpisodeCard

async function mountRail(
  load: () => Promise<TrendingEpisodeCard[]>,
  fallback?: () => Promise<TrendingEpisodeCard[]>,
) {
  const w = mount(RecommendationRail, {
    props: {
      rail: 'captured',
      title: 'More from what you captured',
      load,
      scope: 'mine',
      cardsTestid: 'captured-cards',
      cardTestid: 'captured-card',
      openTestid: 'captured-card-open',
      fallback,
    },
    slots: { 'fallback-note': '<p data-testid="note">Everyone’s</p>' },
    global: { plugins: [i18n, router, createPinia()] },
  })
  await flushPromises()
  return w
}

beforeEach(() => {
  setActivePinia(createPinia())
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

  it('offers "▶ Moments" only for an episode with moments, and reports the sample as its own action', async () => {
    const w = await mountRail(async () => [card('a'), card('b', true)])
    const links = w.findAll('[data-testid="moments-link"]')
    expect(links).toHaveLength(1)
    expect(links[0].attributes('href')).toBe('/episode/b?moments=1')
    await links[0].trigger('click')
    expect(tracked.filter((e) => e.name === 'discover_rail_tap').map((e) => e.props)).toEqual([
      expect.objectContaining({ rail: 'captured', action: 'moments', rank: '2-3' }),
    ])
  })

  it("cold start: nothing of the listener's shows the fallback, labelled, measured as Everyone", async () => {
    const w = await mountRail(async () => [], async () => [card('e1')])
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(1)
    expect(w.find('[data-testid="note"]').exists()).toBe(true)
    expect(tracked.find((e) => e.name === 'discover_rail_shown')?.props).toMatchObject({ scope: 'everyone' })
  })

  it("the listener's own cards never carry the fallback note, and the fallback is not loaded", async () => {
    const fallback = vi.fn(async () => [card('e1')])
    const w = await mountRail(async () => [card('a')], fallback)
    expect(w.find('[data-testid="note"]').exists()).toBe(false)
    expect(fallback).not.toHaveBeenCalled()
    expect(tracked.find((e) => e.name === 'discover_rail_shown')?.props).toMatchObject({ scope: 'mine' })
  })

  it('"Less like this" hides every card with that reason, records it, and offers Undo', async () => {
    const other = card('c')
    other.why = { ...other.why, subject: { kind: 'topic', id: 'topic:y', label: 'Y' } }
    const w = await mountRail(async () => [card('a'), card('b'), other])
    await w.findAll('[data-testid="rail-less"]')[0].trigger('click')
    await flushPromises()
    expect(lessApi.suggestLess).toHaveBeenCalledWith('topic:x', 'X')
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(1) // only topic:y's card stays
    expect(w.get('[data-testid="rail-less-done"]').text()).toContain('Less about X from now on.')
    expect(tracked.filter((e) => e.name === 'discover_rail_tap').map((e) => e.props)).toEqual([
      expect.objectContaining({ action: 'less', rank: '1' }),
    ])
    await w.get('[data-testid="rail-less-undo"]').trigger('click')
    await flushPromises()
    expect(lessApi.undoSuggestLess).toHaveBeenCalledWith('topic:x')
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(3) // reloaded
  })

  it('a slower, older load never overwrites a newer one (a scope switch mid-flight)', async () => {
    let releaseOld: (v: TrendingEpisodeCard[]) => void = () => {}
    const old = new Promise<TrendingEpisodeCard[]>((r) => (releaseOld = r))
    const w = mount(RecommendationRail, {
      props: {
        rail: 'quotes', title: 'T', scope: 'mine', cardsTestid: 'q', cardTestid: 'qc', openTestid: 'qo',
        load: () => old, reloadKey: 'mine',
      },
      global: { plugins: [i18n, router, createPinia()] },
    })
    await w.setProps({ load: async () => [card('new')], reloadKey: 'everyone', scope: 'everyone' })
    await flushPromises()
    releaseOld([])
    await flushPromises()
    expect(w.findAll('[data-testid="qc"]')).toHaveLength(1)
  })

  it('when "Less like this" leaves only outside-share cards, the rail goes too', async () => {
    const outside = card('o')
    outside.why = { ...outside.why, subject: { kind: 'topic', id: 'topic:z', label: 'Z' }, outside_habits: true }
    const w = await mountRail(async () => [card('a'), outside])
    await w.findAll('[data-testid="rail-less"]')[0].trigger('click')
    await flushPromises()
    expect(w.findAll('[data-testid="captured-card"]')).toHaveLength(0)
    expect(w.find('[data-testid="rail-less-undo"]').exists()).toBe(true) // Undo stays reachable
  })

  it('offers no "Less like this" on the cold-start fallback (those are Everyone\'s, not a reason of theirs)', async () => {
    const w = await mountRail(async () => [], async () => [card('e1')])
    expect(w.find('[data-testid="rail-less"]').exists()).toBe(false)
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
