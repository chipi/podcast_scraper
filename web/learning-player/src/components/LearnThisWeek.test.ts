/**
 * "Learn X this week" on Discover (operator 2026-10-11): subjects to start when there is no path,
 * then the path's day, a dot per day and the step to do now; a step opens on its day.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'

const api = vi.hoisted(() => ({ getLearn: vi.fn(), startLearn: vi.fn(), learnStepDone: vi.fn() }))
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

describe('LearnThisWeek', () => {
  it('without a path offers subjects in their kind colours, and one tap starts a path', async () => {
    api.getLearn.mockResolvedValue({
      path: null,
      candidates: [{ subject_id: 'thc:risk', subject_kind: 'storyline', subject_label: 'Managing risk', steps: 7 }],
    })
    api.startLearn.mockResolvedValue(path(1))
    const w = mountIt()
    await flushPromises()
    expect(api.getLearn).toHaveBeenCalledWith('mine')
    const chip = w.get('[data-testid="learn-subject"]')
    expect(chip.text()).toBe('Managing risk · 7 days')
    expect(chip.classes()).toContain('text-storyline')
    await chip.trigger('click')
    await flushPromises()
    expect(api.startLearn).toHaveBeenCalledWith('thc:risk')
    expect(w.get('[data-testid="learn-title"]').text()).toBe('Learn Managing risk this week')
    expect(w.get('[data-testid="learn-day"]').text()).toBe('Day 1 of 7')
    expect(tracked).toEqual([
      { name: 'learn_path', props: { action: 'start', kind: 'storyline', source: 'discover' } },
    ])
  })

  it('shows the earliest open day not yet done, playable from its clip, and marks it done', async () => {
    api.getLearn.mockResolvedValue({ path: path(3, [0]), candidates: [] })
    api.learnStepDone.mockResolvedValue(path(3, [0, 1]))
    const w = mountIt()
    await flushPromises()
    expect(w.findAll('[data-testid="learn-dot"]')).toHaveLength(7)
    const now = w.get('[data-testid="learn-step"]')
    expect(now.attributes('data-day')).toBe('2') // day 1 done, day 2 open and not done
    expect(w.get('[data-testid="learn-step-play"]').attributes('href')).toBe('/episode/ep-2?t=84&play=1')
    await w.get('[data-testid="learn-step-mark-done"]').trigger('click')
    await flushPromises()
    expect(api.learnStepDone).toHaveBeenCalledWith(1)
    expect(w.get('[data-testid="learn-step"]').attributes('data-day')).toBe('3')
    expect(tracked.at(-1)).toEqual({ name: 'learn_path', props: { action: 'step_done', kind: 'storyline', day: 2 } })
    await w.get('[data-testid="learn-step-play"]').trigger('click')
    expect(tracked.at(-1)).toEqual({ name: 'learn_path', props: { action: 'step_play', kind: 'storyline', day: 3 } })
  })

  it('is hidden with neither a path nor anything to offer', async () => {
    api.getLearn.mockResolvedValue({ path: null, candidates: [] })
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
  it('re-reads the path on return: stopped on the whole-week page, it offers subjects again', async () => {
    api.getLearn
      .mockResolvedValueOnce({ path: path(1), candidates: [] })
      .mockResolvedValue({
        path: null,
        candidates: [{ subject_id: 'thc:risk', subject_kind: 'storyline', subject_label: 'Managing risk', steps: 7 }],
      })
    setActivePinia(createPinia())
    const tab = await keptAlive(LearnThisWeek, { plugins: [i18n, router], props: { scope: 'mine' } })
    expect(tab.wrapper.find('[data-testid="learn-title"]').exists()).toBe(true)
    await tab.leave()
    await tab.back()
    await flushPromises()
    expect(tab.wrapper.find('[data-testid="learn-subject"]').exists()).toBe(true)
  })
})
