// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import EnrichmentEdgesPanel from './EnrichmentEdgesPanel.vue'
import { useSubjectStore } from '../../stores/subject'
import { invalidateEnrichmentCache } from '../../composables/useEnrichmentEnvelopeCache'

/**
 * RFC-088 chunk-9 follow-up — mount tests for the graph-view
 * enrichment edges panel. Verifies the panel reads both envelopes,
 * filters by subject focus, and binds row clicks to the subject store.
 */

function res(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function stubFetch(payloads: Record<string, unknown>): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      for (const [key, body] of Object.entries(payloads)) {
        if (url.includes(key)) return res(body)
      }
      return res({}, 404)
    }),
  )
}

beforeEach(() => {
  setActivePinia(createPinia())
  invalidateEnrichmentCache()
})

afterEach(() => {
  vi.unstubAllGlobals()
})


describe('EnrichmentEdgesPanel — mount + behaviour', () => {
  const SIM = {
    schema_version: '1.0',
    enricher_id: 'topic_similarity',
    data: {
      topic_count: 2,
      topics: [
        {
          topic_id: 'topic:ai',
          topic_label: 'AI',
          top_k: [
            { topic_id: 'topic:ml', topic_label: 'ML', similarity: 0.91 },
            { topic_id: 'topic:safety', topic_label: 'Safety', similarity: 0.83 },
          ],
        },
        {
          topic_id: 'topic:climate',
          topic_label: 'Climate',
          top_k: [{ topic_id: 'topic:policy', topic_label: 'Policy', similarity: 0.7 }],
        },
      ],
    },
  }

  const CONTRA = {
    schema_version: '1.0',
    enricher_id: 'nli_contradiction',
    data: {
      contradictions: [
        {
          topic_id: 'topic:ai',
          person_a_id: 'person:alice',
          person_a_name: 'Alice',
          person_b_id: 'person:bob',
          person_b_name: 'Bob',
          insight_a_id: 'i1',
          insight_b_id: 'i2',
          contradiction_score: 0.92,
        },
        {
          topic_id: 'topic:climate',
          person_a_id: 'person:carol',
          person_b_id: 'person:dave',
          insight_a_id: 'i3',
          insight_b_id: 'i4',
          contradiction_score: 0.7,
        },
      ],
    },
  }

  it('hides itself when both envelopes return 404', async () => {
    stubFetch({})  // every URL → 404
    const w = mount(EnrichmentEdgesPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    expect(w.find('[data-testid="enrichment-edges-panel"]').exists()).toBe(false)
  })

  it('renders similarity + contradiction rows when both envelopes present', async () => {
    stubFetch({
      '/api/corpus/enrichments/topic_similarity': SIM,
      '/api/corpus/enrichments/nli_contradiction': CONTRA,
    })
    const w = mount(EnrichmentEdgesPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    expect(w.find('[data-testid="enrichment-edges-panel"]').exists()).toBe(true)
    expect(w.find('[data-testid="enrichment-edges-similarity"]').exists()).toBe(true)
    expect(w.find('[data-testid="enrichment-edges-contradictions"]').exists()).toBe(true)
    // 2 similarity edges (top 5 across corpus) + 1 contradiction visible.
    expect(w.findAll('[data-testid^="enrichment-edges-sim-"]').length).toBeGreaterThan(0)
    expect(w.findAll('[data-testid^="enrichment-edges-contra-"]').length).toBe(2)
  })

  it('narrows similarity to the focused topic', async () => {
    stubFetch({
      '/api/corpus/enrichments/topic_similarity': SIM,
      '/api/corpus/enrichments/nli_contradiction': CONTRA,
    })
    const subject = useSubjectStore()
    subject.focusTopic('topic:ai')
    const w = mount(EnrichmentEdgesPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    // Only topic:ai's neighbours render (ml + safety) — topic:climate's neighbour
    // (policy) does NOT appear.
    expect(w.find('[data-testid="enrichment-edges-sim-topic:ai--topic:ml"]').exists()).toBe(true)
    expect(w.find('[data-testid="enrichment-edges-sim-topic:ai--topic:safety"]').exists()).toBe(true)
    expect(w.find('[data-testid="enrichment-edges-sim-topic:climate--topic:policy"]').exists()).toBe(false)
  })

  it('narrows contradictions to the focused person', async () => {
    stubFetch({
      '/api/corpus/enrichments/topic_similarity': SIM,
      '/api/corpus/enrichments/nli_contradiction': CONTRA,
    })
    const subject = useSubjectStore()
    subject.focusPerson('person:alice')
    const w = mount(EnrichmentEdgesPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    expect(w.find('[data-testid="enrichment-edges-contra-i1--i2"]').exists()).toBe(true)
    // The carol/dave contradiction is filtered out.
    expect(w.find('[data-testid="enrichment-edges-contra-i3--i4"]').exists()).toBe(false)
  })

  it('clicking a similarity row pivots subject focus to the partner topic', async () => {
    stubFetch({
      '/api/corpus/enrichments/topic_similarity': SIM,
      '/api/corpus/enrichments/nli_contradiction': CONTRA,
    })
    const subject = useSubjectStore()
    subject.focusTopic('topic:ai')
    const w = mount(EnrichmentEdgesPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    // The row has TWO buttons (left + right endpoint). Clicking the right
    // endpoint focuses topic:ml.
    const row = w.get('[data-testid="enrichment-edges-sim-topic:ai--topic:ml"]')
    const buttons = row.findAll('button')
    await buttons[1].trigger('click')
    expect(subject.topicId).toBe('topic:ml')
  })

  it('clicking a contradiction row pivots focus to a person', async () => {
    stubFetch({
      '/api/corpus/enrichments/topic_similarity': SIM,
      '/api/corpus/enrichments/nli_contradiction': CONTRA,
    })
    const subject = useSubjectStore()
    const w = mount(EnrichmentEdgesPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    const row = w.get('[data-testid="enrichment-edges-contra-i1--i2"]')
    const buttons = row.findAll('button')
    // Buttons in order: person_a, person_b, topic — click person_b.
    await buttons[1].trigger('click')
    expect(subject.personId).toBe('person:bob')
  })
})
