/** The whole week of a path (`/learn`): every day, the open ones markable; Stop is confirmed. */
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'

const api = vi.hoisted(() => ({ getLearn: vi.fn(), learnStepDone: vi.fn(), stopLearn: vi.fn(), learnOpenNext: vi.fn() }))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))

import LearnPathView from './LearnPathView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/learn', name: 'learn', component: LearnPathView },
      { path: '/browse', name: 'browse', component: { template: '<div/>' } },
      { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    ],
  })
}
const step = (day: number, unlocked: boolean) => ({
  day,
  episode: { slug: `ep-${day}`, title: `Episode ${day}`, podcast_title: 'Show' },
  insight_id: `i${day}`,
  text: 'A claim',
  quote_text: 'Said',
  speaker: null,
  start_ms: 1000,
  end_ms: 9000,
  unlocked,
  done: false,
})

afterEach(() => vi.clearAllMocks())

describe('LearnPathView', () => {
  it('lists every day, and Stop asks first, then stops and returns to Discover', async () => {
    api.getLearn.mockResolvedValue({
      path: {
        subject_id: 'tc:x',
        subject_kind: 'theme',
        subject_label: 'Reliability',
        started_at: 0,
        day: 2,
        steps: [1, 2, 3, 4, 5, 6, 7].map((d) => step(d, d <= 2)),
      },
      candidates: [],
    })
    api.stopLearn.mockResolvedValue(undefined)
    setActivePinia(createPinia())
    const router = makeRouter()
    await router.push('/learn')
    await router.isReady()
    const w = mount(LearnPathView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.get('[data-testid="learn-view-title"]').text()).toBe('Reliability')
    expect(w.findAll('[data-testid="learn-step"]')).toHaveLength(7)
    expect(w.findAll('[data-testid="learn-step-locked"]')).toHaveLength(5)
    await w.get('[data-testid="learn-stop"]').trigger('click')
    expect(api.stopLearn).not.toHaveBeenCalled()
    await w.get('[data-testid="learn-stop-yes"]').trigger('click')
    await flushPromises()
    expect(api.stopLearn).toHaveBeenCalled()
    expect(router.currentRoute.value.name).toBe('browse')
  })

  it('opens only today; an earlier day is closed with whether it was done, and opens on tap', async () => {
    const steps = [1, 2, 3, 4, 5, 6, 7].map((d) => ({ ...step(d, d <= 3), done: d === 1 }))
    api.getLearn.mockResolvedValue({
      path: { subject_id: 'tc:x', subject_kind: 'theme', subject_label: 'R', started_at: 0, day: 3, steps },
      candidates: [],
    })
    setActivePinia(createPinia())
    const router = makeRouter()
    await router.push('/learn')
    await router.isReady()
    const w = mount(LearnPathView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const card = (d: number) => w.get(`[data-testid="learn-step"][data-day="${d}"]`)
    expect(w.findAll('[data-open="true"]').map((c) => c.attributes('data-day'))).toEqual(['3'])
    expect(card(3).find('[data-testid="learn-step-play"]').exists()).toBe(true)
    expect(card(1).find('[data-testid="learn-step-done-mark"]').exists()).toBe(true)
    expect(card(1).find('[data-testid="learn-step-play"]').exists()).toBe(false)
    expect(card(2).get('[data-testid="learn-step-not-done"]').text()).toContain('Not done yet')
    expect(card(4).find('[data-testid="learn-step-open"]').exists()).toBe(false) // later: not openable
    await card(1).get('[data-testid="learn-step-open"]').trigger('click')
    expect(card(1).attributes('data-open')).toBe('true')
    expect(card(1).get('[data-testid="learn-step-play"]').attributes('href')).toBe('/episode/ep-1?t=1&play=1')
    expect(card(1).find('[data-testid="learn-step-mark-done"]').exists()).toBe(false) // already done
    await card(1).get('[data-testid="learn-step-toggle"]').trigger('click')
    expect(card(1).attributes('data-open')).toBe('false')
    await card(3).get('[data-testid="learn-step-toggle"]').trigger('click') // today can close too
    expect(w.findAll('[data-open="true"]')).toHaveLength(0)
  })

  it('"Next day now" on the done newest day opens the next one, which becomes the open card', async () => {
    const steps = [1, 2, 3, 4, 5, 6, 7].map((d) => ({ ...step(d, d === 1), done: d === 1 }))
    const base = { subject_id: 'tc:x', subject_kind: 'theme', subject_label: 'R', started_at: 0 }
    api.getLearn.mockResolvedValue({ path: { ...base, day: 1, can_open_next: true, steps }, candidates: [] })
    api.learnOpenNext.mockResolvedValue({
      ...base,
      day: 2,
      can_open_next: false,
      steps: steps.map((s) => ({ ...s, unlocked: s.day <= 2 })),
    })
    setActivePinia(createPinia())
    const router = makeRouter()
    await router.push('/learn')
    await router.isReady()
    const w = mount(LearnPathView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.findAll('[data-testid="learn-step-next"]')).toHaveLength(1)
    await w.get('[data-testid="learn-step-next"]').trigger('click')
    await flushPromises()
    expect(api.learnOpenNext).toHaveBeenCalledOnce()
    expect(w.findAll('[data-open="true"]').map((c) => c.attributes('data-day'))).toEqual(['2'])
    expect(w.find('[data-testid="learn-step-next"]').exists()).toBe(false)
  })

  it('without a path says so and points to Discover', async () => {
    api.getLearn.mockResolvedValue({ path: null, candidates: [] })
    setActivePinia(createPinia())
    const router = makeRouter()
    await router.push('/learn')
    await router.isReady()
    const w = mount(LearnPathView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.get('[data-testid="learn-none"]').text()).toContain('not learning anything')
  })
})
