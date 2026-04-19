import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { useSubjectStore } from './subject'

describe('useSubjectStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('focusEpisode sets kind and clears graph fields', () => {
    const s = useSubjectStore()
    s.focusGraphNode('topic:x')
    s.focusEpisode('metadata/a.json')
    expect(s.kind).toBe('episode')
    expect(s.episodeMetadataPath).toBe('metadata/a.json')
    expect(s.graphNodeCyId).toBeNull()
  })

  it('focusGraphNode sets kind and clears episode path', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/b.json')
    s.focusGraphNode('insight:y')
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('insight:y')
    expect(s.episodeMetadataPath).toBeNull()
  })

  it('clearSubject resets all', () => {
    const s = useSubjectStore()
    s.focusEpisode('m.json', { graphConnectionsCyId: 'topic:z' })
    s.clearSubject()
    expect(s.kind).toBeNull()
    expect(s.episodeMetadataPath).toBeNull()
    expect(s.graphConnectionsCyId).toBeNull()
  })

  it('rejects blank paths as clear', () => {
    const s = useSubjectStore()
    s.focusEpisode('ok.json')
    s.focusEpisode('  ')
    expect(s.kind).toBeNull()
  })
})
