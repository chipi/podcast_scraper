/**
 * #1931 — the chip list RE-ORDERS the server's rows to group them into theme colour blocks.
 * That re-ordering must use the server's own ranking key (``trend_score``), not the velocity
 * ratio it renders as the "×" badge.
 *
 * Why this is not cosmetic: since #1931 the server ranks the rail by ``trend_score``
 * (volume-with-recency) because velocity is an acceleration RATIO — a topic mentioned twice ever,
 * both times last month, scores a huge ratio on two data points, while the topic genuinely
 * dominating the corpus sits near 1.0. Ordering the colour blocks by peak *velocity* re-introduced
 * exactly the bias the server fix removed, one layer down and invisible to the backend tests.
 */
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import type { RisingTopic, TopicTheme } from './trending'
import TrendingSparkChips from './TrendingSparkChips.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

const topic = (id: string, v: number, score?: number): RisingTopic => ({
  id,
  label: id.replace(/^topic:/, ''),
  v,
  score,
  total: 10,
  series: [1, 2, 3],
})

/** Two single-topic themes, so group order is decided purely by each group's peak. */
const THEMES: Record<string, TopicTheme> = {
  'topic:loud': { color: '#38bdf8', label: 'loud storyline', group: 0 },
  'topic:spiky': { color: '#a78bfa', label: 'spiky storyline', group: 1 },
}

const labelsInOrder = (topics: RisingTopic[], themes = THEMES): string[] => {
  const w = mount(TrendingSparkChips, {
    props: { topics, topicTheme: themes, collapseAt: 20 },
    global: { plugins: [i18n] },
  })
  return w.findAll('[data-testid="trend-spark-row"]').map((n) => n.text())
}

describe('TrendingSparkChips ordering (#1931)', () => {
  it('ranks theme blocks by trend_score, not by the velocity badge', () => {
    // `spiky` wins on the RATIO (5× vs 1.1×) but `loud` is what the corpus is actually about.
    const order = labelsInOrder([topic('topic:spiky', 5, 2.0), topic('topic:loud', 1.1, 14.0)])
    expect(order[0]).toContain('loud')
    expect(order[1]).toContain('spiky')
  })

  it('orders within a theme by trend_score too', () => {
    const themes: Record<string, TopicTheme> = {
      'topic:a': { color: '#38bdf8', label: 'one storyline', group: 0 },
      'topic:b': { color: '#38bdf8', label: 'one storyline', group: 0 },
    }
    const order = labelsInOrder([topic('topic:a', 9, 1.0), topic('topic:b', 0.4, 12.0)], themes)
    expect(order[0]).toContain('b')
  })

  it('falls back to velocity when the artifact predates trend_score', () => {
    // Pre-#1931 envelopes carry no trend_score; the rail must still order sensibly rather than
    // collapsing every group to the same peak and rendering an arbitrary order.
    const order = labelsInOrder([topic('topic:spiky', 5), topic('topic:loud', 1.1)])
    expect(order[0]).toContain('spiky')
  })

  it('still shows the velocity ratio as the badge — it is a different, honest number', () => {
    const w = mount(TrendingSparkChips, {
      props: { topics: [topic('topic:loud', 0.3, 14.0)], topicTheme: THEMES, collapseAt: 20 },
      global: { plugins: [i18n] },
    })
    // The top row of a "trending" rail legitimately carries a sub-1.0 ratio. The badge is labelled
    // "vs recent average" precisely so that reads as informative rather than contradictory.
    expect(w.text()).toContain('0.3×')
  })
})
