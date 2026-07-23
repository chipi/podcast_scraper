import { describe, expect, it } from 'vitest'
import { matchedFieldLabel, summarizeMatchedFields } from './matchedFields'
import type { SearchHit } from '../services/types'

function hit(overrides: {
  matched_field?: string
  doc_type?: string
  score?: number
}): SearchHit {
  return {
    doc_id: 'd',
    score: overrides.score ?? 0.5,
    text: '',
    source_tier: 'segment',
    metadata: {
      ...(overrides.matched_field ? { matched_field: overrides.matched_field } : {}),
      ...(overrides.doc_type ? { doc_type: overrides.doc_type } : {}),
    },
  }
}

describe('matchedFieldLabel', () => {
  it('prefers matched_field when the indexer set it', () => {
    expect(matchedFieldLabel(hit({ matched_field: 'title' }))).toBe('Title')
    expect(matchedFieldLabel(hit({ matched_field: 'description' }))).toBe('Description')
    expect(matchedFieldLabel(hit({ matched_field: 'summary' }))).toBe('Summary')
    expect(matchedFieldLabel(hit({ matched_field: 'summary_bullet' }))).toBe('Summary bullet')
    expect(matchedFieldLabel(hit({ matched_field: 'transcript' }))).toBe('Transcript')
  })

  it('falls back to doc_type when matched_field is absent', () => {
    expect(matchedFieldLabel(hit({ doc_type: 'transcript' }))).toBe('Transcript')
    expect(matchedFieldLabel(hit({ doc_type: 'episode_title' }))).toBe('Title')
    expect(matchedFieldLabel(hit({ doc_type: 'episode_description' }))).toBe('Description')
    expect(matchedFieldLabel(hit({ doc_type: 'summary_short' }))).toBe('Summary')
    expect(matchedFieldLabel(hit({ doc_type: 'summary' }))).toBe('Summary bullet')
    expect(matchedFieldLabel(hit({ doc_type: 'insight' }))).toBe('Insight')
  })

  it('returns null for hits that are not episode-level metadata surfaces', () => {
    expect(matchedFieldLabel(hit({ doc_type: 'kg_topic' }))).toBeNull()
    expect(matchedFieldLabel(hit({ doc_type: 'kg_entity' }))).toBeNull()
    expect(matchedFieldLabel(hit({ doc_type: 'quote' }))).toBeNull()
    expect(matchedFieldLabel(hit({}))).toBeNull()
  })
})

describe('summarizeMatchedFields', () => {
  it('counts per label and emits results in the fixed display order', () => {
    const out = summarizeMatchedFields([
      hit({ doc_type: 'transcript' }),
      hit({ doc_type: 'transcript' }),
      hit({ doc_type: 'episode_title' }),
      hit({ doc_type: 'insight' }),
      hit({ doc_type: 'summary_short' }),
    ])
    expect(out).toEqual([
      { label: 'Title', count: 1 },
      { label: 'Summary', count: 1 },
      { label: 'Transcript', count: 2 },
      { label: 'Insight', count: 1 },
    ])
  })

  it('drops non-episode-surface hits from the summary silently', () => {
    const out = summarizeMatchedFields([
      hit({ doc_type: 'kg_topic' }),
      hit({ doc_type: 'kg_entity' }),
      hit({ doc_type: 'transcript' }),
    ])
    expect(out).toEqual([{ label: 'Transcript', count: 1 }])
  })

  it('returns an empty array when nothing matches an episode field', () => {
    const out = summarizeMatchedFields([
      hit({ doc_type: 'kg_topic' }),
      hit({ doc_type: 'kg_entity' }),
    ])
    expect(out).toEqual([])
  })
})
