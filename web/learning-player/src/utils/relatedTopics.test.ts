import { describe, expect, it } from 'vitest'
import { aggregateRelatedTopics } from './relatedTopics'
import type { SearchHit } from '../services/types'

function hit(related: Array<{ topic_id: string; topic_label?: string; similarity: number }>): SearchHit {
  return {
    doc_id: 'd',
    score: 0.5,
    text: '',
    source_tier: 'insight',
    metadata: { query_enrichments: { related_topics: related } },
  }
}

describe('aggregateRelatedTopics', () => {
  it('returns an empty list when no hit carries query_enrichments', () => {
    const bare: SearchHit = {
      doc_id: 'd', score: 0.5, text: '', source_tier: 'insight', metadata: {},
    }
    expect(aggregateRelatedTopics([bare])).toEqual([])
  })

  it('takes the max similarity across hits and counts membership', () => {
    const chips = aggregateRelatedTopics([
      hit([{ topic_id: 'topic:ml', topic_label: 'ML', similarity: 0.6 }]),
      hit([{ topic_id: 'topic:ml', topic_label: 'ML', similarity: 0.9 }]),
    ])
    expect(chips).toEqual([{ topicId: 'topic:ml', label: 'ML', score: 0.9, count: 2 }])
  })

  it('sorts by score desc, then count desc, then id asc', () => {
    const chips = aggregateRelatedTopics([
      hit([
        { topic_id: 'topic:b', topic_label: 'B', similarity: 0.7 },
        { topic_id: 'topic:a', topic_label: 'A', similarity: 0.7 }, // tie -> id asc
      ]),
      hit([{ topic_id: 'topic:a', topic_label: 'A', similarity: 0.7 }]), // higher count
      hit([{ topic_id: 'topic:c', topic_label: 'C', similarity: 0.9 }]),
    ])
    expect(chips.map((c) => c.topicId)).toEqual(['topic:c', 'topic:a', 'topic:b'])
  })

  it('falls back to the topic id when no server-side label is present', () => {
    const [chip] = aggregateRelatedTopics([
      hit([{ topic_id: 'topic:xyz', similarity: 0.5 }]),
    ])
    expect(chip.label).toBe('topic:xyz')
  })

  it('a real label arriving later replaces a raw-id fallback', () => {
    const [chip] = aggregateRelatedTopics([
      hit([{ topic_id: 'topic:xyz', similarity: 0.5 }]),
      hit([{ topic_id: 'topic:xyz', topic_label: 'Grand Theory', similarity: 0.4 }]),
    ])
    expect(chip.label).toBe('Grand Theory')
  })

  it('drops entries with a blank topic_id', () => {
    const chips = aggregateRelatedTopics([
      hit([
        { topic_id: '', similarity: 0.9 },
        { topic_id: 'topic:ok', topic_label: 'Ok', similarity: 0.5 },
      ]),
    ])
    expect(chips.map((c) => c.topicId)).toEqual(['topic:ok'])
  })

  it('respects the limit', () => {
    const chips = aggregateRelatedTopics(
      [hit([
        { topic_id: 'topic:a', similarity: 0.9 },
        { topic_id: 'topic:b', similarity: 0.8 },
        { topic_id: 'topic:c', similarity: 0.7 },
      ])],
      2,
    )
    expect(chips.map((c) => c.topicId)).toEqual(['topic:a', 'topic:b'])
  })

  it('coerces a non-numeric similarity to 0 so the chip still surfaces', () => {
    const [chip] = aggregateRelatedTopics([
      hit([{ topic_id: 'topic:x', similarity: Number.NaN as unknown as number }]),
    ])
    expect(chip.score).toBe(0)
  })
})
