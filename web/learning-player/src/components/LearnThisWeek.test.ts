/**
 * "Learn something this week" on Discover (operator 2026-10-11): where a path starts — subjects in
 * their kind's colours, one tap starts a path and opens its week — and the paths already on (up to
 * three), each a link to its week. The day to do lives on Home's "In progress".
 */
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'

const api = vi.hoisted(() => ({ getLearn: vi.fn(), startLearn: vi.fn() }))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))
const tracked = vi.hoisted(() => [] as Array<{ name: string; props: unknown }>)
vi.mock('../services/analytics', async (orig) => ({
  ...(await orig<typeof import('../services/analytics')>()),
  track: (name: string, props: unknown) => tracked.push({ name, props }),
}))

import LearnThisWeek from './LearnThisWeek.vue'
import { keptAlive } from '../test/keptAlive'
import type { LearningPath, PathStep } from '../services/api'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', component: { template: '<div/>' } },
    { path: '/learn', name: 'learn', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function step(day: number, unlocked: boolean, done = false): PathStep {
  return {
    day,
    episode: { slug: `ep-${day}`, title: `Episode ${day}`, podcast_title: 'Show' } as PathStep['episode'],
    insight_id: `i${day}`,
    text: 'A claim',
    quote_text: `Said on day ${day}`,
    speaker: 'Ann',
    start_ms: 84_000,
    end_ms: 96_000,
    unlocked,
    done,
  }
}
function path(day: number, done: number[] = []): LearningPath {
  return {
    subject_id: 'thc:risk',
    subject_kind: 'storyline',
    subject_label: 'Managing risk',
    started_at: 0,
    day,
    steps: Array.from({ length: 7 }, (_, i) => step(i + 1, i < day, done.includes(i))),
  }
}
function mountIt() {
  setActivePinia(createPinia())
  return mount(LearnThisWeek, { props: { scope: 'mine' }, global: { plugins: [i18n, router] } })
}

afterEach(() => {
  vi.clearAllMocks()
  tracked.length = 0
})

const offer = { subject_id: 'thc:risk', subject_kind: 'storyline' as const, subject_label: 'Managing risk', steps: 7 }

describe('LearnThisWeek', () => {
  it('offers subjects in their kind colours; one tap starts a path and opens its week', async () => {
    api.getLearn.mockResolvedValue({ paths: [], candidates: [offer], max_paths: 3 })
    api.startLearn.mockResolvedValue(path(1))
    await router.push('/')
    const w = mountIt()
    await flushPromises()
    expect(api.getLearn).toHaveBeenCalledWith('mine')
    expect(w.get('[data-testid="learn-title"]').text()).toBe('Learn something this week')
    const chip = w.get('[data-testid="learn-subject"]')
    expect(chip.text()).toBe('Managing risk · 7 days')
    expect(chip.classes()).toContain('text-storyline')
    await chip.trigger('click')
    await flushPromises()
    expect(api.startLearn).toHaveBeenCalledWith('thc:risk')
    expect(router.currentRoute.value.fullPath).toBe('/learn?subject=thc:risk')
    expect(tracked).toEqual([
      { name: 'learn_path', props: { action: 'start', kind: 'storyline', source: 'discover' } },
    ])
  })

  it('lists the paths already on as links to their weeks; at three it offers nothing to start', async () => {
    api.getLearn.mockResolvedValue({ paths: [path(3, [0]), { ...path(1), subject_id: 'tc:b', subject_label: 'B' }], candidates: [offer], max_paths: 3 })
    const w = mountIt()
    await flushPromises()
    const current = w.findAll('[data-testid="learn-current"]')
    expect(current.map((c) => c.text())).toEqual(['Managing risk · day 3 of 7 ›', 'B · day 1 of 7 ›'])
    expect(current[0].attributes('href')).toBe('/learn?subject=thc:risk')
    expect(w.text()).toContain('Learning now')

    api.getLearn.mockResolvedValue({ paths: [path(1), path(1), path(1)], candidates: [], max_paths: 3 })
    const full = mountIt()
    await flushPromises()
    expect(full.find('[data-testid="learn-subject"]').exists()).toBe(false)
    expect(full.text()).toContain('What you are learning')
  })

  it('is hidden with neither a path nor anything to offer', async () => {
    api.getLearn.mockResolvedValue({ paths: [], candidates: [], max_paths: 3 })
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="learn-this-week"]').exists()).toBe(false)
  })
})

describe('LearnStep', () => {
  it('a day not yet come says when it opens and offers nothing to do', async () => {
    const { default: LearnStep } = await import('./LearnStep.vue')
    const w = mount(LearnStep, { props: { step: step(4, false) }, global: { plugins: [i18n, router] } })
    expect(w.get('[data-testid="learn-step-locked"]').text()).toBe('Opens on day 4')
    expect(w.find('[data-testid="learn-step-play"]').exists()).toBe(false)
    expect(w.find('[data-testid="learn-step-mark-done"]').exists()).toBe(false)
  })
})

describe('LearnThisWeek on a kept-alive Discover', () => {
  it('re-reads on return: a path stopped elsewhere is gone, and subjects are offered again', async () => {
    api.getLearn
      .mockResolvedValueOnce({ paths: [path(1), path(1), path(1)], candidates: [], max_paths: 3 })
      .mockResolvedValue({ paths: [], candidates: [offer], max_paths: 3 })
    setActivePinia(createPinia())
    const tab = await keptAlive(LearnThisWeek, { plugins: [i18n, router], props: { scope: 'mine' } })
    expect(tab.wrapper.findAll('[data-testid="learn-current"]')).toHaveLength(3)
    await tab.leave()
    await tab.back()
    await flushPromises()
    expect(tab.wrapper.find('[data-testid="learn-current"]').exists()).toBe(false)
    expect(tab.wrapper.find('[data-testid="learn-subject"]').exists()).toBe(true)
  })
})
