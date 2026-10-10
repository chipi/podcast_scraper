/**
 * Why a recommendation is here (ADR-163): every reason kind in words, the quote with where it is
 * said, and no quote at all when nothing grounds it.
 */
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import type { Reason } from '../services/types'
import RecommendationWhy from './RecommendationWhy.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
})

const QUOTE = { text: 'Rates will fall.', episode_slug: 'ep-1', speaker: 'Ann', start_ms: 1_394_000, insight_id: 'i1' }

function why(over: Partial<Reason>): Reason {
  return { kind: 'rising_topic', quote: null, subject: null, capture: null, events: null, outside_habits: false, ...over }
}
function mountWhy(w: Reason) {
  return mount(RecommendationWhy, { props: { why: w }, global: { plugins: [i18n, router] } })
}
const reason = (w: Reason) => mountWhy(w).get('[data-testid="why-reason"]').text()

describe('RecommendationWhy: the reason in words, from its parts', () => {
  it('names each kind of reason', () => {
    const topic = { kind: 'topic' as const, id: 'topic:rates', label: 'Rates' }
    expect(reason(why({ kind: 'rising_topic', subject: topic }))).toBe('Rising: Rates')
    expect(reason(why({ kind: 'most_engaged', events: 4 }))).toBe('Most listened to and saved')
    expect(reason(why({ kind: 'thread', subject: { kind: 'storyline', id: 'thc:1', label: 'AI rules' } }))).toBe(
      'New in AI rules',
    )
    expect(reason(why({ kind: 'person', subject: { kind: 'person', id: 'person:ann', label: 'Ann' } }))).toBe(
      'Ann, whom you keep hearing',
    )
    expect(reason(why({ kind: 'another_view', subject: topic }))).toBe('Another view on Rates')
  })

  it('a captured reason quotes the listener’s own capture', () => {
    const capture = { kind: 'highlight' as const, episode_slug: 'ep-0', text: 'sleep spindles', start_ms: 1 }
    expect(reason(why({ kind: 'captured', capture }))).toBe('Because you highlighted “sleep spindles”')
    expect(reason(why({ kind: 'captured', capture: { ...capture, kind: 'note', text: 'look into' } }))).toBe(
      'Because of your note “look into”',
    )
    expect(reason(why({ kind: 'captured', capture: null }))).toBe('Because of what you captured')
  })
})

describe('RecommendationWhy: the quote that grounds it', () => {
  it('shows the quote, who says it, and plays from where it is said', () => {
    const w = mountWhy(why({ quote: QUOTE }))
    expect(w.get('[data-testid="why-quote"]').text()).toContain('“Rates will fall.”')
    expect(w.get('[data-testid="why-speaker"]').text()).toBe('Ann')
    const play = w.get('[data-testid="why-play-from"]')
    expect(play.text()).toContain('Play from 23:14')
    expect(play.attributes('href')).toBe('/episode/ep-1?t=1394&play=1')
    expect(play.attributes('aria-label')).toBe('Play “Rates will fall.” from 23:14')
  })

  it('tapping "Play from" tells the rail, so it can report the tap', async () => {
    const w = mountWhy(why({ quote: QUOTE }))
    await w.get('[data-testid="why-play-from"]').trigger('click')
    expect(w.emitted('play')).toHaveLength(1)
  })

  it('a quote with no time is shown, with nothing to play from', () => {
    const w = mountWhy(why({ quote: { ...QUOTE, start_ms: null } }))
    expect(w.find('[data-testid="why-quote"]').exists()).toBe(true)
    expect(w.find('[data-testid="why-play-from"]').exists()).toBe(false)
  })

  it('nothing grounds it: no quote at all, not a stand-in', () => {
    const w = mountWhy(why({ quote: null }))
    expect(w.find('[data-testid="why-quote"]').exists()).toBe(false)
    expect(w.get('[data-testid="why-reason"]').text()).not.toBe('')
  })

  it('labels the serendipity share', () => {
    expect(mountWhy(why({ outside_habits: true })).get('[data-testid="why-outside"]').text()).toBe('Outside your usual')
    expect(mountWhy(why({})).find('[data-testid="why-outside"]').exists()).toBe(false)
  })
})
