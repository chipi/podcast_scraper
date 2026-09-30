import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import type { YourWeekItem } from '../services/types'
import YourWeekCard from './YourWeekCard.vue'

/**
 * The card's links advance the spaced ladder (#35). Moved here from YourWeek.test.ts when Home
 * stopped showing the digest's revisit section (2026-09-30): the card keeps the behaviour, so it
 * keeps the test, at the level where the behaviour lives.
 */
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
})

function hrefOf(item: YourWeekItem): string {
  const w = mount(YourWeekCard, { props: { item }, global: { plugins: [i18n, router] } })
  return w.find('a').attributes('href') ?? ''
}

describe('YourWeekCard links', () => {
  it("a user's own capture links with ?revisit and its moment", () => {
    const href = hrefOf({
      episode_slug: 'ep-a',
      episode_title: 'Episode A',
      highlight_id: 'h-a',
      deep_link: '/episode/ep-a?t=10&revisit=h-a',
      t_ms: 10000,
      graph_refs: [],
    })
    expect(href).toContain('/episode/ep-a')
    expect(href).toContain('revisit=h-a')
    expect(href).toContain('t=10')
  })

  it('an item with no highlight_id links without one', () => {
    // Auto-picks and follows/trending rows have no ladder behind them; a marker there would record
    // a review against a highlight that does not exist.
    const href = hrefOf({
      episode_slug: 'ep-b',
      episode_title: 'Episode B',
      deep_link: '/episode/ep-b',
      graph_refs: [],
    })
    expect(href).toContain('/episode/ep-b')
    expect(href).not.toContain('revisit')
  })
})
