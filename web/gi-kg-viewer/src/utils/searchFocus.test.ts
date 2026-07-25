import { describe, expect, it } from 'vitest'
import type { Core } from 'cytoscape'
import {
  episodeFallbackForSearchHit,
  graphNodeIdFromSearchHit,
  primaryCompareSubjectFromHit,
  resolveCyNodeId,
} from './searchFocus'
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

describe('episodeFallbackForSearchHit', () => {
  function hitWithEpisode(docType: string, episodeId?: unknown): SearchHit {
    return {
      doc_id: 'd1',
      score: 0.9,
      text: 'text',
      metadata: { doc_type: docType, source_id: 'quote:abc', episode_id: episodeId },
    } as SearchHit
  }

  it('returns the episode_id as a fallback', () => {
    expect(episodeFallbackForSearchHit(hitWithEpisode('quote', 'ep-123'))).toBe('ep-123')
  })

  it('trims and rejects blank / non-string episode_id', () => {
    expect(episodeFallbackForSearchHit(hitWithEpisode('quote', '  ep-9  '))).toBe('ep-9')
    expect(episodeFallbackForSearchHit(hitWithEpisode('quote', '   '))).toBeNull()
    expect(episodeFallbackForSearchHit(hitWithEpisode('quote', undefined))).toBeNull()
    expect(episodeFallbackForSearchHit(hitWithEpisode('quote', 123))).toBeNull()
  })

  it('a quote hit has no graph node but its episode fallback resolves (stuck-timeout fix)', () => {
    // Quotes are not rendered as graph nodes → primary id never resolves.
    const quoteHit = hitWithEpisode('quote', '6162ba8a')
    const primary = graphNodeIdFromSearchHit(quoteHit) // 'quote:abc'
    const fallback = episodeFallbackForSearchHit(quoteHit) // '6162ba8a'
    // Graph has the unified episode node but no quote node.
    const core = mockCore(['__unified_ep__:6162ba8a'])
    expect(primary && resolveCyNodeId(core, primary)).toBeNull()
    expect(fallback && resolveCyNodeId(core, fallback)).toBe('__unified_ep__:6162ba8a')
  })
})

describe('primaryCompareSubjectFromHit (Search v3 §S8 pin-to-compare)', () => {
  function md(metadata: Record<string, unknown>): SearchHit {
    return { doc_id: 'd1', score: 0.9, text: 't', metadata } as SearchHit
  }

  it('maps a kg_topic hit to a Topic subject', () => {
    expect(
      primaryCompareSubjectFromHit(
        md({ doc_type: 'kg_topic', source_id: 'topic:compute', topic_label: 'Compute' }),
      ),
    ).toEqual({ kind: 'topic', id: 'topic:compute', label: 'Compute' })
  })

  it('maps a kg_entity hit to a Person subject', () => {
    expect(
      primaryCompareSubjectFromHit(
        md({ doc_type: 'kg_entity', source_id: 'person:alice', entity_label: 'Alice' }),
      ),
    ).toEqual({ kind: 'person', id: 'person:alice', label: 'Alice' })
  })

  it('maps an insight hit with a speaker to a Person subject', () => {
    expect(
      primaryCompareSubjectFromHit(md({ doc_type: 'insight', speaker_name: 'Bob' })),
    ).toEqual({ kind: 'person', id: 'Bob', label: 'Bob' })
  })

  it('falls back to the Episode when no speaker is present', () => {
    expect(
      primaryCompareSubjectFromHit(
        md({ doc_type: 'transcript', episode_id: 'ep-9', episode_title: 'Ep Nine' }),
      ),
    ).toEqual({ kind: 'episode', id: 'ep-9', label: 'Ep Nine' })
  })

  it('prefers the speaker over the episode for an insight hit that has both', () => {
    expect(
      primaryCompareSubjectFromHit(
        md({ doc_type: 'insight', speaker: 'Carol', episode_id: 'ep-1' }),
      ),
    ).toEqual({ kind: 'person', id: 'Carol', label: 'Carol' })
  })

  it('returns null when nothing usable is present', () => {
    expect(primaryCompareSubjectFromHit(md({ doc_type: 'insight' }))).toBeNull()
  })

  it('returns null for a kg_topic hit with a blank source_id', () => {
    expect(
      primaryCompareSubjectFromHit(md({ doc_type: 'kg_topic', source_id: '   ' })),
    ).toBeNull()
  })
})

// ── resolveCyNodeId ──

function mockCore(existingIds: string[]): Core {
  const idSet = new Set(existingIds)
  return {
    $id: (id: string) => ({ empty: () => !idSet.has(id) }),
  } as unknown as Core
}

describe('resolveCyNodeId', () => {
  it('returns bare id when it exists in the graph', () => {
    const core = mockCore(['topic:foo'])
    expect(resolveCyNodeId(core, 'topic:foo')).toBe('topic:foo')
  })

  it('resolves g: prefix for GI nodes', () => {
    const core = mockCore(['g:insight:abc123'])
    expect(resolveCyNodeId(core, 'insight:abc123')).toBe('g:insight:abc123')
  })

  it('resolves k: prefix for KG nodes', () => {
    const core = mockCore(['k:topic:cuba-s-economic-crisis'])
    expect(resolveCyNodeId(core, 'topic:cuba-s-economic-crisis')).toBe(
      'k:topic:cuba-s-economic-crisis',
    )
  })

  it('resolves k:kg: prefix for double-prefixed KG nodes', () => {
    const core = mockCore(['k:kg:entity:org:cuba'])
    expect(resolveCyNodeId(core, 'entity:org:cuba')).toBe('k:kg:entity:org:cuba')
  })

  it('resolves g:gi: prefix', () => {
    const core = mockCore(['g:gi:quote:abc'])
    expect(resolveCyNodeId(core, 'quote:abc')).toBe('g:gi:quote:abc')
  })

  it('prefers bare id over prefixed when both exist', () => {
    const core = mockCore(['insight:x', 'g:insight:x'])
    expect(resolveCyNodeId(core, 'insight:x')).toBe('insight:x')
  })

  it('prefers g: over k: when both exist', () => {
    const core = mockCore(['g:topic:x', 'k:topic:x'])
    expect(resolveCyNodeId(core, 'topic:x')).toBe('g:topic:x')
  })

  it('returns null when no variant matches', () => {
    const core = mockCore(['unrelated:node'])
    expect(resolveCyNodeId(core, 'topic:missing')).toBeNull()
  })

  it('returns null for empty rawId', () => {
    const core = mockCore(['g:anything'])
    expect(resolveCyNodeId(core, '')).toBeNull()
  })

  it('resolves __unified_ep__ for library episode focus', () => {
    const core = mockCore(['__unified_ep__:226cc6d2-2178-11f1-bfcd-c76500e5b946'])
    expect(
      resolveCyNodeId(core, '226cc6d2-2178-11f1-bfcd-c76500e5b946'),
    ).toBe('__unified_ep__:226cc6d2-2178-11f1-bfcd-c76500e5b946')
  })
})
