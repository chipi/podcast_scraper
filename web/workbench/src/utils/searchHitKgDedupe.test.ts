import { describe, expect, it } from 'vitest'
import type { SearchHit } from '../api/searchApi'
import { isKgSurfaceMultiEpisodeDedupe } from './searchHitKgDedupe'

function hit(meta: Record<string, unknown>): SearchHit {
  return {
    doc_id: 'x',
    score: 0.5,
    text: 't',
    metadata: meta,
  }
}

describe('isKgSurfaceMultiEpisodeDedupe', () => {
  it('is false for non-kg doc types', () => {
    expect(isKgSurfaceMultiEpisodeDedupe(hit({ doc_type: 'insight', kg_surface_match_count: 3 }))).toBe(
      false,
    )
  })

  it('is false when count missing or 1', () => {
    expect(isKgSurfaceMultiEpisodeDedupe(hit({ doc_type: 'kg_entity' }))).toBe(false)
    expect(isKgSurfaceMultiEpisodeDedupe(hit({ doc_type: 'kg_entity', kg_surface_match_count: 1 }))).toBe(
      false,
    )
  })

  it('is true for kg_entity / kg_topic with count > 1', () => {
    expect(isKgSurfaceMultiEpisodeDedupe(hit({ doc_type: 'kg_entity', kg_surface_match_count: 2 }))).toBe(
      true,
    )
    expect(isKgSurfaceMultiEpisodeDedupe(hit({ doc_type: 'kg_topic', kg_surface_match_count: 5 }))).toBe(
      true,
    )
  })
})
