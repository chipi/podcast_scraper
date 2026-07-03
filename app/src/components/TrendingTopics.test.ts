import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { CorpusEnrichmentSignals } from '../services/types'
import TrendingTopics from './TrendingTopics.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountIt = () => mount(TrendingTopics, { global: { plugins: [i18n] } })

const velocity = (topics: CorpusEnrichmentSignals['temporal_velocity']) =>
  vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue({ temporal_velocity: topics })

afterEach(() => vi.restoreAllMocks())

describe('TrendingTopics', () => {
  it('shows only rising topics (>=1.5x, total>=3), sorted by velocity desc', async () => {
    velocity({
      topics: [
        { topic_id: 'topic:ai', topic_label: 'ai', velocity_last_over_6mo: 2, total: 10 },
        { topic_id: 'topic:policy', topic_label: 'foreign policy', velocity_last_over_6mo: 4, total: 3 },
        { topic_id: 'topic:steady', topic_label: 'steady', velocity_last_over_6mo: 1, total: 20 }, // not rising
        { topic_id: 'topic:noise', topic_label: 'noise', velocity_last_over_6mo: 5, total: 1 }, // below floor
      ],
    })
    const w = mountIt()
    await flushPromises()
    const chips = w.findAll('[data-testid="home-trending-chip"]')
    // policy (4x) before ai (2x); steady + noise excluded.
    expect(chips.map((c) => c.text().replace(/\s+/g, ' ').trim())).toEqual([
      'foreign policy ↑ 4×',
      'ai ↑ 2×',
    ])
  })

  it('emits open with the topic id when a chip is clicked', async () => {
    velocity({
      topics: [{ topic_id: 'topic:policy', topic_label: 'foreign policy', velocity_last_over_6mo: 4, total: 3 }],
    })
    const w = mountIt()
    await flushPromises()
    await w.get('[data-testid="home-trending-chip"]').trigger('click')
    expect(w.emitted('open')![0]).toEqual(['topic:policy'])
  })

  it('renders nothing when no topic is rising', async () => {
    velocity({
      topics: [{ topic_id: 'topic:flat', topic_label: 'flat', velocity_last_over_6mo: 0.9, total: 50 }],
    })
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="home-trending"]').exists()).toBe(false)
  })

  it('renders nothing when the velocity enricher is absent', async () => {
    vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue({})
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="home-trending"]').exists()).toBe(false)
  })
})
