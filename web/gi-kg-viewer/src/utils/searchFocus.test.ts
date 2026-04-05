import { describe, expect, it } from 'vitest'
import { graphNodeIdFromSearchHit } from './searchFocus'
import type { SearchHit } from '../api/searchApi'

function hit(docType: string, sourceId?: string): SearchHit {
  return {
    doc_id: 'd1',
    score: 0.9,
    text: 'text',
    metadata: { doc_type: docType, source_id: sourceId },
  }
}

describe('graphNodeIdFromSearchHit', () => {
  it('returns source_id for focusable doc_types', () => {
    for (const dt of ['insight', 'quote', 'kg_topic', 'kg_entity']) {
      expect(graphNodeIdFromSearchHit(hit(dt, 'abc'))).toBe('abc')
    }
  })

  it('trims whitespace from source_id', () => {
    expect(graphNodeIdFromSearchHit(hit('insight', '  x  '))).toBe('x')
  })

  it('returns null for non-focusable doc_type', () => {
    expect(graphNodeIdFromSearchHit(hit('episode', 'abc'))).toBeNull()
  })

  it('returns null when source_id is missing', () => {
    expect(graphNodeIdFromSearchHit(hit('insight'))).toBeNull()
  })

  it('returns null when source_id is blank', () => {
    expect(graphNodeIdFromSearchHit(hit('insight', '  '))).toBeNull()
  })
})
