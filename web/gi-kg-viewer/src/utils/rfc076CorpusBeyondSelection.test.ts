import { describe, expect, it } from 'vitest'
import {
  normalizeArtifactRelPath,
  wouldRfc076AppendNewArtifacts,
} from './rfc076CorpusBeyondSelection'

describe('normalizeArtifactRelPath', () => {
  it('trims and flips backslashes', () => {
    expect(normalizeArtifactRelPath('  metadata\\a.gi.json  ')).toBe('metadata/a.gi.json')
  })
})

describe('wouldRfc076AppendNewArtifacts', () => {
  it('is false when episodes empty', () => {
    expect(wouldRfc076AppendNewArtifacts([], ['metadata/a.gi.json'])).toBe(false)
  })

  it('is false when every gi/kg path is already selected', () => {
    expect(
      wouldRfc076AppendNewArtifacts(
        [
          {
            gi_relative_path: 'metadata/a.gi.json',
            kg_relative_path: 'metadata/a.kg.json',
            bridge_relative_path: 'metadata/a.bridge.json',
          },
        ],
        ['metadata/a.gi.json', 'metadata/a.kg.json'],
      ),
    ).toBe(false)
  })

  it('is true when a gi path is not in selection', () => {
    expect(
      wouldRfc076AppendNewArtifacts(
        [
          {
            gi_relative_path: 'metadata/other.gi.json',
            kg_relative_path: 'metadata/other.kg.json',
            bridge_relative_path: 'metadata/other.bridge.json',
          },
        ],
        ['metadata/a.gi.json'],
      ),
    ).toBe(true)
  })

  it('is true when gi matches but kg is missing from selection', () => {
    expect(
      wouldRfc076AppendNewArtifacts(
        [
          {
            gi_relative_path: 'metadata/a.gi.json',
            kg_relative_path: 'metadata/a.kg.json',
            bridge_relative_path: 'metadata/a.bridge.json',
          },
        ],
        ['metadata/a.gi.json'],
      ),
    ).toBe(true)
  })
})
