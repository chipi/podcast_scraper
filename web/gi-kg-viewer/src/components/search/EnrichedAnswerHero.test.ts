// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import EnrichedAnswerHero from './EnrichedAnswerHero.vue'
import { useSearchStore } from '../../stores/search'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

/**
 * EnrichedAnswerHero — Search v3 §S5 (RFC-107, UXS-016 + UXS-008). The
 * hero surfaces the QueryEnricher chain's ``related_topics`` output as an
 * aggregated summary above the hit cards. This spec pins the state
 * machine (hidden / skeleton / error / rendered) and the aggregation +
 * ranking rules.
 */
function mountHero() {
  return mount(EnrichedAnswerHero, { attachTo: document.body })
}

function makeHitWithRelatedTopics(rt: Array<{ topic_id: string; topic_label?: string; similarity?: number }>, docId = 'd:x') {
  return {
    doc_id: docId,
    score: 0.5,
    text: 'body',
    metadata: {
      doc_type: 'insight',
      query_enrichments: { related_topics: rt },
    },
  } as unknown as import('../../api/searchApi').SearchHit
}

describe('EnrichedAnswerHero (Search v3 §S5)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('is hidden when enrichment is off (no capability + no explicit opt-in)', () => {
    const w = mountHero()
    // capability off, filter null → hidden
    expect(w.find('[data-testid="enriched-answer-hero"]').exists()).toBe(false)
    w.unmount()
  })

  it('is hidden when enrichment is on but no results have enrichment output', () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = true
    const search = useSearchStore()
    search.results = [{ doc_id: 'd', score: 0.5, text: 'x', metadata: {} }]
    const w = mountHero()
    expect(w.find('[data-testid="enriched-answer-hero"]').exists()).toBe(false)
    w.unmount()
  })

  it('renders the error state when the server reports enrichmentCallFailed', () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = true
    const search = useSearchStore()
    search.enrichmentCallFailed = true
    const w = mountHero()
    expect(w.find('[data-testid="enriched-answer-hero"]').exists()).toBe(true)
    expect(w.find('[data-testid="enriched-answer-error"]').exists()).toBe(true)
    expect(w.find('[data-testid="enriched-answer-error"]').text()).toContain(
      'Enrichment failed',
    )
    w.unmount()
  })

  it('renders the skeleton when enrichment is on + search is loading', () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = true
    const search = useSearchStore()
    search.loading = true
    const w = mountHero()
    expect(w.find('[data-testid="enriched-answer-hero"]').exists()).toBe(true)
    expect(w.find('[data-testid="enriched-answer-skeleton"]').exists()).toBe(true)
    w.unmount()
  })

  it('renders topic chips aggregated + ranked across hits', () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = true
    const search = useSearchStore()
    // Two hits surface topic:llm (both with similarity 0.9), one hit surfaces
    // topic:bio with 0.7. Order should be llm (1.8, 2 hits) then bio (0.7, 1 hit).
    search.results = [
      makeHitWithRelatedTopics(
        [
          { topic_id: 'topic:llm', topic_label: 'LLMs', similarity: 0.9 },
          { topic_id: 'topic:bio', topic_label: 'Biology', similarity: 0.7 },
        ],
        'd:1',
      ),
      makeHitWithRelatedTopics(
        [{ topic_id: 'topic:llm', topic_label: 'LLMs', similarity: 0.9 }],
        'd:2',
      ),
    ]
    const w = mountHero()
    expect(w.find('[data-testid="enriched-answer-topics"]').exists()).toBe(true)
    const chips = w.findAll('[data-testid^="enriched-answer-topic-"]')
    expect(chips).toHaveLength(2)
    // First chip is topic:llm (summed 1.8, 2 hits); second is topic:bio.
    expect(chips[0].attributes('data-testid')).toBe('enriched-answer-topic-topic:llm')
    expect(chips[0].text()).toContain('LLMs')
    expect(chips[0].text()).toContain('2') // hit count
    expect(chips[1].attributes('data-testid')).toBe('enriched-answer-topic-topic:bio')
    expect(chips[1].text()).toContain('Biology')
    expect(chips[1].text()).toContain('1')
    w.unmount()
  })

  it('caps the visible chips at 6 and shows an overflow count when more exist', () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = true
    const search = useSearchStore()
    // 8 distinct topics from one hit; expect 6 chips + "+2 more" overflow.
    search.results = [
      makeHitWithRelatedTopics(
        Array.from({ length: 8 }, (_, i) => ({
          topic_id: `topic:${String.fromCharCode(97 + i)}`,
          topic_label: `Topic ${i}`,
          similarity: 1 - i * 0.05,
        })),
      ),
    ]
    const w = mountHero()
    expect(w.findAll('[data-testid^="enriched-answer-topic-"]')).toHaveLength(6)
    const overflow = w.get('[data-testid="enriched-answer-overflow"]')
    expect(overflow.text()).toContain('+2 more')
    w.unmount()
  })

  it('clicking a topic chip calls subject.focusTopic with the topic id', async () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = true
    const search = useSearchStore()
    search.results = [
      makeHitWithRelatedTopics([
        { topic_id: 'topic:climate', topic_label: 'Climate', similarity: 0.8 },
      ]),
    ]
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusTopic')
    const w = mountHero()
    await w.get('[data-testid="enriched-answer-topic-topic:climate"]').trigger('click')
    expect(spy).toHaveBeenCalledWith('topic:climate')
    w.unmount()
  })

  it('respects an explicit filter=false override even when capability is on', () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = true
    const search = useSearchStore()
    search.filters.enrichResults = false
    // Even with hits that carry related_topics, hero stays hidden when the
    // user explicitly turned enrichment off.
    search.results = [
      makeHitWithRelatedTopics([
        { topic_id: 'topic:x', topic_label: 'X', similarity: 0.5 },
      ]),
    ]
    const w = mountHero()
    expect(w.find('[data-testid="enriched-answer-hero"]').exists()).toBe(false)
    w.unmount()
  })

  it('renders when enrichment is opted-in even if capability signal is off', () => {
    const shell = useShellStore()
    shell.enrichedSearchAvailable = false
    const search = useSearchStore()
    search.filters.enrichResults = true
    search.results = [
      makeHitWithRelatedTopics([
        { topic_id: 'topic:x', topic_label: 'X', similarity: 0.5 },
      ]),
    ]
    const w = mountHero()
    expect(w.find('[data-testid="enriched-answer-hero"]').exists()).toBe(true)
    w.unmount()
  })
})
