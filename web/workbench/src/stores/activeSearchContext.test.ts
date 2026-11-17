import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { useActiveSearchContextStore } from './activeSearchContext'
import type { SearchHit } from '../api/searchApi'

function hit(episodeId: string, score: number, text = 'snippet text', docType = 'insight'): SearchHit {
  return {
    doc_id: `${docType}:${episodeId}:${score}`,
    score,
    metadata: { episode_id: episodeId, doc_type: docType },
    text,
  }
}

describe('useActiveSearchContextStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('is inactive before any context is set', () => {
    const s = useActiveSearchContextStore()
    expect(s.active).toBe(false)
    expect(s.relevanceFor('ep1')).toBeNull()
  })

  it('projects hits onto episodes keeping the best score per episode', () => {
    const s = useActiveSearchContextStore()
    s.setContext('climate', [
      hit('ep1', 0.4, 'weaker'),
      hit('ep1', 0.9, 'stronger'),
      hit('ep2', 0.7, 'other'),
    ])
    expect(s.active).toBe(true)
    expect(s.query).toBe('climate')
    expect(s.relevanceFor('ep1')?.score).toBe(0.9)
    expect(s.relevanceFor('ep1')?.snippet).toBe('stronger')
    expect(s.relevanceFor('ep2')?.score).toBe(0.7)
  })

  it('skips hits with no episode_id', () => {
    const s = useActiveSearchContextStore()
    const orphan: SearchHit = { doc_id: 'x', score: 1, metadata: {}, text: 't' }
    s.setContext('q', [orphan, hit('ep1', 0.5)])
    expect(s.byEpisode.size).toBe(1)
    expect(s.relevanceFor('ep1')).not.toBeNull()
  })

  it('is inactive when the query is empty even with hits', () => {
    const s = useActiveSearchContextStore()
    s.setContext('   ', [hit('ep1', 0.5)])
    expect(s.active).toBe(false)
    expect(s.relevanceFor('ep1')).toBeNull()
  })

  it('is inactive when no hits map to an episode', () => {
    const s = useActiveSearchContextStore()
    s.setContext('q', [])
    expect(s.active).toBe(false)
  })

  it('truncates long snippets with an ellipsis', () => {
    const s = useActiveSearchContextStore()
    s.setContext('q', [hit('ep1', 0.5, 'x'.repeat(400))])
    const snippet = s.relevanceFor('ep1')?.snippet ?? ''
    expect(snippet.length).toBeLessThanOrEqual(220)
    expect(snippet.endsWith('…')).toBe(true)
  })

  it('clear() drops the context', () => {
    const s = useActiveSearchContextStore()
    s.setContext('q', [hit('ep1', 0.5)])
    s.clear()
    expect(s.active).toBe(false)
    expect(s.query).toBe('')
    expect(s.byEpisode.size).toBe(0)
  })
})
