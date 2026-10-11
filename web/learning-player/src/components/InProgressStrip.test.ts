/**
 * Home's "In progress" (operator 2026-10-11): the door into what was started — every Learn path
 * (the day to do now, or "Next day now", or done) and the last unfinished short listen, resumable at
 * its clip. Hidden with nothing in progress; re-read when Home is returned to.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import type { LearningPath, PathStep, ReelProgress } from '../services/api'

const api = vi.hoisted(() => ({ getInProgress: vi.fn(), learnOpenNext: vi.fn() }))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))
const tracked = vi.hoisted(() => [] as Array<{ name: string; props: unknown }>)
vi.mock('../services/analytics', async (orig) => ({
  ...(await orig<typeof import('../services/analytics')>()),
  track: (name: string, props: unknown) => tracked.push({ name, props }),
}))

import InProgressStrip from './InProgressStrip.vue'
import { keptAlive } from '../test/keptAlive'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', component: { template: '<div/>' } },
    { path: '/learn', name: 'learn', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/listen/topic/:id', name: 'topic-reel', component: { template: '<div/>' } },
  ],
})
const step = (day: number, unlocked: boolean, done: boolean): PathStep =>
  ({
    day,
    episode: { slug: `ep-${day}`, title: `Episode ${day}`, podcast_title: 'Show' },
    insight_id: `i${day}`,
    text: '',
    quote_text: '',
    speaker: null,
    start_ms: 84_000,
    end_ms: 96_000,
    unlocked,
    done,
  }) as PathStep
const path = (id: string, label: string, open: number, done: number, canNext = false): LearningPath => ({
  subject_id: id,
  subject_kind: 'storyline',
  subject_label: label,
  started_at: 0,
  day: open,
  can_open_next: canNext,
  steps: Array.from({ length: 7 }, (_, i) => step(i + 1, i < open, i < done)),
})
const reel: ReelProgress = {
  subject_id: 'topic:systems-thinking',
  subject_kind: 'topic',
  subject_label: 'systems thinking',
  index: 4,
  clips: 9,
  updated_at: 0,
  speaker: 'Scott Bessent',
  quote_text: 'Systems thinking applies.',
}

afterEach(() => {
  vi.clearAllMocks()
  tracked.length = 0
})

async function mountIt() {
  const w = mount(InProgressStrip, { global: { plugins: [i18n, router] } })
  await flushPromises()
  return w
}

describe('InProgressStrip', () => {
  it('plays each path’s day to do, and resumes the short listen at its clip', async () => {
    api.getInProgress.mockResolvedValue({ paths: [path('thc:a', 'Second-order effects', 3, 2)], reel })
    const w = await mountIt()
    const learn = w.get('[data-testid="in-progress-learn"]')
    expect(learn.text()).toContain('Learn · day 3 of 7')
    expect(learn.text()).toContain('Second-order effects')
    expect(learn.text()).toContain('Episode 3')
    expect(learn.get('[data-testid="in-progress-learn-open"]').attributes('href')).toBe('/learn?subject=thc:a')
    const play = learn.get('[data-testid="in-progress-learn-play"]')
    expect(play.attributes('href')).toBe('/episode/ep-3?t=84&play=1')
    await play.trigger('click')
    expect(tracked.at(-1)).toEqual({
      name: 'learn_path',
      props: { action: 'step_play', kind: 'storyline', day: 3, source: 'in_progress' },
    })
    const r = w.get('[data-testid="in-progress-reel"]')
    expect(r.text()).toContain('Short listen · clip 5 of 9')
    expect(r.text()).toContain('Systems thinking applies.')
    const resume = r.get('[data-testid="in-progress-reel-resume"]')
    expect(resume.attributes('href')).toBe('/listen/topic/topic:systems-thinking?minutes=15&at=5')
    await resume.trigger('click')
    expect(tracked.at(-1)).toEqual({ name: 'reel_open', props: { source: 'in_progress', kind: 'topic' } })
  })

  it('every open day done: "Next day now" opens it in place; otherwise it says tomorrow, or all done', async () => {
    api.getInProgress.mockResolvedValue({
      paths: [path('thc:a', 'A', 2, 2, true), path('thc:b', 'B', 2, 2, false), path('thc:c', 'C', 7, 7)],
      reel: null,
    })
    api.learnOpenNext.mockResolvedValue(path('thc:a', 'A', 3, 2))
    const w = await mountIt()
    const cards = w.findAll('[data-testid="in-progress-learn"]')
    expect(cards[1].text()).toContain('Day 2 done ✓')
    expect(cards[1].text()).toContain('The next day opens tomorrow')
    expect(cards[0].text()).toContain('Day 2 done ✓')
    expect(cards[2].get('[data-testid="in-progress-learn-finished"]').text()).toBe('All 7 days done ✓')
    await cards[0].get('[data-testid="in-progress-learn-next"]').trigger('click')
    await flushPromises()
    expect(api.learnOpenNext).toHaveBeenCalledWith('thc:a')
    expect(w.findAll('[data-testid="in-progress-learn"]')[0].get('[data-testid="in-progress-learn-play"]').attributes('href')).toBe(
      '/episode/ep-3?t=84&play=1',
    )
    expect(tracked.at(-1)).toEqual({
      name: 'learn_path',
      props: { action: 'open_next', kind: 'storyline', day: 3, source: 'in_progress' },
    })
  })

  it('is hidden with nothing in progress, and re-reads when Home is returned to', async () => {
    api.getInProgress
      .mockResolvedValueOnce({ paths: [], reel: null })
      .mockResolvedValue({ paths: [], reel })
    const tab = await keptAlive(InProgressStrip, { plugins: [i18n, router] })
    expect(tab.wrapper.find('[data-testid="home-in-progress"]').exists()).toBe(false)
    await tab.leave()
    await tab.back()
    await flushPromises()
    expect(tab.wrapper.find('[data-testid="in-progress-reel"]').exists()).toBe(true)
  })
})
