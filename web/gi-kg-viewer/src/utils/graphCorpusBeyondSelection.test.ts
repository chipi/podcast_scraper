import { afterEach, describe, expect, it } from 'vitest'
import { episodeArtifactPaths } from '../extensions'
import {
  normalizeArtifactRelPath,
  wouldCrossEpisodeExpandAppendNewArtifacts,
} from './graphCorpusBeyondSelection'

const ep = (kg: string, gi = '') => ({ gi_relative_path: gi, kg_relative_path: kg, bridge_relative_path: '' })

describe('normalizeArtifactRelPath', () => {
  it('normalizes slashes and leading ./', () => {
    expect(normalizeArtifactRelPath('.\\metadata\\a.kg.json')).toBe('metadata/a.kg.json')
  })
})

describe('wouldCrossEpisodeExpandAppendNewArtifacts', () => {
  afterEach(() => episodeArtifactPaths.reset())

  it('is false when episodes list is empty', () => {
    expect(wouldCrossEpisodeExpandAppendNewArtifacts([], ['metadata/a.kg.json'])).toBe(false)
  })

  it('is true when a KG path is not in selection', () => {
    expect(wouldCrossEpisodeExpandAppendNewArtifacts([ep('metadata/b.kg.json')], ['metadata/a.kg.json'])).toBe(true)
  })

  it('is false for a new GI path alone: the expand does not load it here', () => {
    expect(
      wouldCrossEpisodeExpandAppendNewArtifacts([ep('', 'metadata/b.gi.json')], ['metadata/a.kg.json']),
    ).toBe(false)
  })

  it('counts a path a downstream build adds (episodeArtifactPaths), as the expand loads it', () => {
    episodeArtifactPaths.register({
      id: 'probe',
      paths: (e) => (typeof e.gi_relative_path === 'string' && e.gi_relative_path ? [e.gi_relative_path] : []),
    })
    expect(
      wouldCrossEpisodeExpandAppendNewArtifacts([ep('', 'metadata/b.gi.json')], ['metadata/a.kg.json']),
    ).toBe(true)
  })

  it('is false when all episode paths are already selected', () => {
    expect(wouldCrossEpisodeExpandAppendNewArtifacts([ep('metadata/a.kg.json')], ['metadata/a.kg.json'])).toBe(false)
  })

  it('matches normalized selection keys', () => {
    expect(wouldCrossEpisodeExpandAppendNewArtifacts([ep('metadata/x.kg.json')], ['.\\metadata\\x.kg.json'])).toBe(false)
  })
})
