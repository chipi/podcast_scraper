import { describe, expect, it } from 'vitest'
import {
  normalizeArtifactRelPath,
  wouldCrossEpisodeExpandAppendNewArtifacts,
} from './graphCorpusBeyondSelection'

describe('normalizeArtifactRelPath', () => {
  it('normalizes slashes and leading ./', () => {
    expect(normalizeArtifactRelPath('.\\metadata\\a.gi.json')).toBe('metadata/a.gi.json')
  })
})

describe('wouldCrossEpisodeExpandAppendNewArtifacts', () => {
  it('is false when episodes list is empty', () => {
    expect(wouldCrossEpisodeExpandAppendNewArtifacts([], ['metadata/a.gi.json'])).toBe(false)
  })

  it('is true when a GI path is not in selection', () => {
    expect(
      wouldCrossEpisodeExpandAppendNewArtifacts(
        [{ gi_relative_path: 'metadata/b.gi.json', kg_relative_path: null }],
        ['metadata/a.gi.json'],
      ),
    ).toBe(true)
  })

  it('is false when all episode paths are already selected', () => {
    expect(
      wouldCrossEpisodeExpandAppendNewArtifacts(
        [{ gi_relative_path: 'metadata/a.gi.json', kg_relative_path: 'metadata/a.kg.json' }],
        ['metadata/a.gi.json', 'metadata/a.kg.json'],
      ),
    ).toBe(false)
  })

  it('matches normalized selection keys', () => {
    expect(
      wouldCrossEpisodeExpandAppendNewArtifacts(
        [{ gi_relative_path: 'metadata/x.gi.json', kg_relative_path: null }],
        ['.\\metadata\\x.gi.json'],
      ),
    ).toBe(false)
  })
})
