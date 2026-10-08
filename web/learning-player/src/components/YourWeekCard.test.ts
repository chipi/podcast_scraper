import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import YourWeekCard from './YourWeekCard.vue'

/**
 * The card opens the episode, title-forward. It carried a quote, a timestamp and `?revisit=` for
 * REVISIT items until Home stopped showing that section (2026-09-30); the items Home shows now
 * (new in follows / interests, trending) have none of those, so the card is a plain episode link.
 */
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
})

describe('YourWeekCard', () => {
  it('links to the episode and shows its title', () => {
    const w = mount(YourWeekCard, {
      props: {
        item: {
          episode_slug: 'ep-b',
          episode_title: 'Episode B',
          deep_link: '/episode/ep-b',
          graph_refs: [{ id: 'topic:y', kind: 'topic', label: 'Topic Y' }],
        },
      },
      global: { plugins: [i18n, router] },
    })
    expect(w.find('a').attributes('href')).toBe('/episode/ep-b')
    expect(w.text()).toContain('Episode B')
    expect(w.text()).toContain('Topic Y')
  })

  it('falls back to the lead graph label when the episode title is missing', () => {
    const w = mount(YourWeekCard, {
      props: { item: { episode_slug: 'gone', deep_link: '/episode/gone', graph_refs: [{ id: 'topic:z', kind: 'topic', label: 'Topic Z' }] } },
      global: { plugins: [i18n, router] },
    })
    expect(w.text()).toContain('Topic Z')
  })

  it('names the show under a display-size title, like Continue listening (operator 2026-10-08)', () => {
    const w = mount(YourWeekCard, {
      props: { item: { episode_slug: 'ep-c', episode_title: 'Episode C', podcast_title: 'Show C', deep_link: '/episode/ep-c' } },
      global: { plugins: [i18n, router] },
    })
    expect(w.get('[data-testid="yourweek-card-title"]').classes()).toContain('font-display')
    expect(w.get('[data-testid="yourweek-card-show"]').text()).toBe('Show C')
    const text = w.text()
    expect(text.indexOf('Episode C')).toBeLessThan(text.indexOf('Show C'))
  })

  it('shows no empty show line when the show is unknown', () => {
    const w = mount(YourWeekCard, {
      props: { item: { episode_slug: 'ep-d', episode_title: 'Episode D', deep_link: '/episode/ep-d' } },
      global: { plugins: [i18n, router] },
    })
    expect(w.find('[data-testid="yourweek-card-show"]').exists()).toBe(false)
  })
})
