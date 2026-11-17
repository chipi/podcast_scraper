import { afterEach, describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import {
  artifactRelPathsForResolvedRow,
  sortResolvedArtifactsNewestFirst,
} from './resolvedArtifacts'
import { episodeArtifactPaths } from '../extensions'

describe('sortResolvedArtifactsNewestFirst', () => {
  it('sorts by publish_date descending', () => {
    const rows = sortResolvedArtifactsNewestFirst([
      {
        episode_id: 'a',
        publish_date: '2024-01-01',
        gi_relative_path: 'a.gi.json',
        kg_relative_path: null,
        bridge_relative_path: null,
      },
      {
        episode_id: 'b',
        publish_date: '2024-06-01',
        gi_relative_path: 'b.gi.json',
        kg_relative_path: null,
        bridge_relative_path: null,
      },
    ])
    expect(rows.map((r) => r.episode_id)).toEqual(['b', 'a'])
  })
})

describe('artifactRelPathsForResolvedRow', () => {
  afterEach(() => episodeArtifactPaths.reset())

  it('returns the KG path, trimmed by the row builder; GI and bridge are a build\'s to add', () => {
    const r = { episode_id: 'a', gi_relative_path: 'm/a.gi.json', kg_relative_path: 'm/a.kg.json', bridge_relative_path: 'm/a.bridge.json' }
    expect(artifactRelPathsForResolvedRow(r as never)).toEqual(['m/a.kg.json'])
  })

  it('appends the paths an extension adds for the row', () => {
    episodeArtifactPaths.register({ id: 'x', paths: (ep) => [String(ep.bridge_relative_path ?? '')] })
    const r = { episode_id: 'a', kg_relative_path: 'm/a.kg.json', bridge_relative_path: 'm/a.bridge.json' }
    expect(artifactRelPathsForResolvedRow(r as never)).toEqual(['m/a.kg.json', 'm/a.bridge.json'])
  })

  it('omits a blank KG path', () => {
    expect(artifactRelPathsForResolvedRow({ episode_id: 'a', kg_relative_path: '  ' } as never)).toEqual([])
  })
})
