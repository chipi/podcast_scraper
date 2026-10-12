// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import type { UserStats } from '../services/types'
import LearningStats from './LearningStats.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/learn', name: 'learn', component: { template: '<div/>' } }],
})

function stats(over: Partial<UserStats> = {}): UserStats {
  return { episodes: 1, shows: 1, listening_seconds: 0, active_days: 1, day_streak: 0, daily: [], ...over }
}
const mountIt = (s: UserStats) => mount(LearningStats, { props: { stats: s }, global: { plugins: [i18n, router] } })

describe('LearningStats (operator 2026-10-12)', () => {
  it('lists each path on now with its kind, day and days done, linked to its week', async () => {
    const w = mountIt(
      stats({
        learning_now: [
          { subject_id: 'tc:rel', subject_kind: 'theme', subject_label: 'reliability', day: 3, days: 7, done: 2 },
        ],
        paths_finished: 2,
        paths_left: 1,
        learning_days_total: 16,
      }),
    )
    const row = w.get('[data-testid="stats-learning-path"]')
    expect(row.text()).toContain('Theme reliability')
    expect(row.text()).toContain('Day 3 of 7 · 2 done')
    expect(row.attributes('href')).toBe('/learn?subject=tc:rel')
    expect(w.get('[data-testid="stats-learning-totals"]').text()).toBe(
      '2 finished · 1 left · 16 learning days in all',
    )
  })

  it('says nothing for a listener who never learned', () => {
    expect(mountIt(stats()).find('[data-testid="stats-learning"]').exists()).toBe(false)
  })

  it('still shows the totals once every path has ended', () => {
    const w = mountIt(stats({ paths_finished: 1, learning_days_total: 7 }))
    expect(w.find('[data-testid="stats-learning-path"]').exists()).toBe(false)
    expect(w.get('[data-testid="stats-learning-totals"]').text()).toContain('1 finished')
  })
})
