import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import type { EpisodeSummary } from '../services/types'
import ShowActivityChart from './ShowActivityChart.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const ep = (slug: string, publish_date: string) => ({ slug, publish_date }) as EpisodeSummary
const mountChart = (episodes: EpisodeSummary[]) =>
  mount(ShowActivityChart, { props: { episodes }, global: { plugins: [i18n] } })

describe('ShowActivityChart', () => {
  const eps = [ep('a', '2026-04-03'), ep('b', '2026-04-20'), ep('c', '2026-06-01')]

  it('says what a bar is: episodes per month, each bar named and counted', () => {
    const w = mountChart(eps)
    expect(w.get('[data-testid="show-activity-unit"]').text()).toBe('Episodes per month')
    expect(w.get('[data-testid="show-activity-bar-2026-04"]').attributes('aria-label')).toBe('April 2026: 2 episodes')
    expect(w.get('[data-testid="show-activity-bar-2026-05"]').attributes('title')).toBe('May 2026: no episodes')
  })

  it('puts a month under every bar and the year at the first one', () => {
    const axis = mountChart(eps).get('[data-testid="show-activity-axis"]').text()
    expect(axis).toContain('Apr')
    expect(axis).toContain('May')
    expect(axis).toContain('Jun')
    expect(axis).toContain('2026')
  })

  it('a month with episodes is a button that selects it; an empty month is not', async () => {
    const w = mountChart(eps)
    const april = w.get('[data-testid="show-activity-bar-2026-04"]')
    expect(april.element.tagName).toBe('BUTTON')
    await april.trigger('click')
    expect(w.emitted('select')).toEqual([['2026-04']])
    expect(w.get('[data-testid="show-activity-bar-2026-05"]').element.tagName).not.toBe('BUTTON')
  })

  it('bars are data, not a kind: neither the topic hue nor the accent', () => {
    const html = mountChart(eps).html()
    expect(html).not.toMatch(/bg-topic|bg-accent/)
  })
})
