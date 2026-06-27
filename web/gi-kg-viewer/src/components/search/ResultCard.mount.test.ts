// @vitest-environment happy-dom
import { beforeEach, describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import ResultCard from './ResultCard.vue'
import type { SearchHit } from '../../api/searchApi'
import { useSubjectStore } from '../../stores/subject'

/**
 * RFC-088 chunk-9 follow-up — mount tests for the search result card's
 * related-topics chip strip.
 */

function makeHit(extras: Partial<SearchHit> = {}): SearchHit {
  return {
    doc_id: 'h1',
    score: 0.9,
    metadata: { doc_type: 'kg_topic', topic_id: 'topic:ai' },
    text: 'AI',
    source_tier: 'aux',
    supporting_quotes: null,
    lifted: null,
    ...extras,
  } as SearchHit
}

beforeEach(() => {
  setActivePinia(createPinia())
})


describe('ResultCard — related_topics chip strip', () => {
  it('hides the chip strip when no related_topics on the hit', () => {
    const w = mount(ResultCard, {
      props: { hit: makeHit(), libraryOpensEnabled: false },
    })
    expect(w.find('[data-testid="search-result-related-topics"]').exists()).toBe(false)
  })

  it('renders one chip per related topic when present', () => {
    const w = mount(ResultCard, {
      props: {
        hit: makeHit({
          metadata: {
            doc_type: 'kg_topic',
            topic_id: 'topic:ai',
            query_enrichments: {
              related_topics: [
                { topic_id: 'topic:ml', topic_label: 'ML', similarity: 0.91 },
                { topic_id: 'topic:safety', topic_label: 'Safety', similarity: 0.83 },
              ],
            },
          },
        }),
        libraryOpensEnabled: false,
      },
    })
    expect(w.find('[data-testid="search-result-related-topics"]').exists()).toBe(true)
    expect(w.find('[data-testid="search-result-related-topic-topic:ml"]').exists()).toBe(true)
    expect(w.find('[data-testid="search-result-related-topic-topic:safety"]').exists()).toBe(true)
  })

  it('clicking a chip pivots subject focus to the partner topic', async () => {
    const subject = useSubjectStore()
    const w = mount(ResultCard, {
      props: {
        hit: makeHit({
          metadata: {
            doc_type: 'kg_topic',
            topic_id: 'topic:ai',
            query_enrichments: {
              related_topics: [{ topic_id: 'topic:ml', similarity: 0.91 }],
            },
          },
        }),
        libraryOpensEnabled: false,
      },
    })
    await w.get('[data-testid="search-result-related-topic-topic:ml"]').trigger('click')
    expect(subject.topicId).toBe('topic:ml')
  })

  it('skips chips with empty or non-string topic_id', () => {
    const w = mount(ResultCard, {
      props: {
        hit: makeHit({
          metadata: {
            doc_type: 'kg_topic',
            topic_id: 'topic:ai',
            query_enrichments: {
              related_topics: [
                { topic_id: 'topic:ml' },
                { topic_id: '' },
                { topic_id: null },
              ],
            },
          },
        }),
        libraryOpensEnabled: false,
      },
    })
    const chips = w.findAll('[data-testid^="search-result-related-topic-"]')
    expect(chips.length).toBe(1)
  })
})
