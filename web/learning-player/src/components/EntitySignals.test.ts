import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { CorpusEnrichmentSignals } from '../services/types'
import EntitySignals from './EntitySignals.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function mountSignals(kind: 'person' | 'topic', id: string) {
  return mount(EntitySignals, { props: { kind, id }, global: { plugins: [i18n] } })
}

const SIGNALS: CorpusEnrichmentSignals = {
  grounding_rate: {
    persons: [
      { person_id: 'person:jane-doe', person_name: 'Jane Doe', total_insights: 20, grounded_insights: 17, rate: 0.85 },
    ],
  },
  guest_coappearance: {
    pairs: [
      { person_a_id: 'person:jane-doe', person_b_id: 'person:bob-lee', person_b_name: 'Bob Lee', episode_count: 4 },
      { person_a_id: 'person:amy-ng', person_b_id: 'person:jane-doe', person_a_name: 'Amy Ng', episode_count: 9 },
    ],
  },
  temporal_velocity: {
    topics: [
      { topic_id: 'topic:ai', topic_label: 'AI', velocity_last_over_6mo: 2.1, total: 40 },
    ],
  },
  topic_similarity: {
    topics: [
      {
        topic_id: 'topic:ai',
        top_k: [
          { topic_id: 'topic:ml', topic_label: 'Machine Learning', similarity: 0.9 },
          { topic_id: 'topic:llms', topic_label: 'LLMs', similarity: 0.8 },
        ],
      },
    ],
  },
  topic_cooccurrence_corpus: {
    pairs: [
      { topic_a_id: 'topic:ai', topic_b_id: 'topic:policy', topic_b_label: 'Policy', episode_count: 5, lift: 3.2 },
      { topic_a_id: 'topic:weak', topic_b_id: 'topic:ai', topic_a_label: 'Weak', episode_count: 1, lift: 4 },
    ],
  },
}

afterEach(() => vi.restoreAllMocks())

describe('EntitySignals — person', () => {
  it('shows grounding and co-appears (sorted)', async () => {
    vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue(SIGNALS)
    const w = mountSignals('person', 'person:jane-doe')
    await flushPromises()

    expect(w.get('[data-testid="es-grounding"]').text()).toContain('17 of 20')
    expect(w.get('[data-testid="es-grounding"]').text()).toContain('85%')

    // Co-appears sorted by episode_count desc: Amy Ng (9) before Bob Lee (4).
    const co = w.get('[data-testid="es-coappears"]').findAll('button')
    expect(co.map((b) => b.text().replace(/\s+/g, ' '))).toEqual(['Amy Ng · 9', 'Bob Lee · 4'])
  })

  it('emits open when a co-appears chip is clicked', async () => {
    vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue(SIGNALS)
    const w = mountSignals('person', 'person:jane-doe')
    await flushPromises()
    await w.get('[data-testid="es-coappears"]').findAll('button')[0].trigger('click')
    expect(w.emitted('open')![0]).toEqual([{ kind: 'person', id: 'person:amy-ng' }])
  })

  it('renders nothing for a person with no matching signals', async () => {
    vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue(SIGNALS)
    const w = mountSignals('person', 'person:nobody')
    await flushPromises()
    expect(w.find('[data-testid="entity-signals"]').exists()).toBe(false)
  })
})

describe('EntitySignals — topic', () => {
  it('shows momentum, similar topics, and discussed-alongside (lift-filtered)', async () => {
    vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue(SIGNALS)
    const w = mountSignals('topic', 'topic:ai')
    await flushPromises()

    expect(w.get('[data-testid="es-momentum"]').text()).toContain('Rising')
    expect(w.get('[data-testid="es-momentum"]').text()).toContain('2.1×')

    const sim = w.get('[data-testid="es-similar"]').findAll('button')
    expect(sim.map((b) => b.text())).toEqual(['Machine Learning', 'LLMs'])

    // Only the lift>1 & episode_count>=2 pair survives (Policy); the weak pair is dropped.
    const along = w.get('[data-testid="es-alongside"]').findAll('button')
    expect(along.map((b) => b.text())).toEqual(['Policy'])
  })

  it('emits open when a similar-topic chip is clicked', async () => {
    vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue(SIGNALS)
    const w = mountSignals('topic', 'topic:ai')
    await flushPromises()
    await w.get('[data-testid="es-similar"]').findAll('button')[0].trigger('click')
    expect(w.emitted('open')![0]).toEqual([{ kind: 'topic', id: 'topic:ml' }])
  })
})
