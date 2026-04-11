import { describe, expect, it } from 'vitest'
import type { ArtifactData } from '../types/artifact'
import {
  guessMetadataRelPathFromArtifactRelPath,
  logicalEpisodeIdFromGraphNodeId,
  logicalEpisodeIdsForLibraryGraphSync,
  logicalEpisodeIdsMatchingMetadataPath,
  resolveEpisodeMetadataFromLoadedArtifacts,
} from './graphEpisodeMetadata'
import { parseArtifact } from './parsing'

describe('graphEpisodeMetadata', () => {
  it('logicalEpisodeIdFromGraphNodeId strips known prefixes', () => {
    expect(logicalEpisodeIdFromGraphNodeId('episode:abc')).toBe('abc')
    expect(logicalEpisodeIdFromGraphNodeId('g:episode:abc')).toBe('abc')
    expect(logicalEpisodeIdFromGraphNodeId('k:episode:abc')).toBe('abc')
    expect(logicalEpisodeIdFromGraphNodeId('__unified_ep__:abc')).toBe('abc')
  })

  it('guessMetadataRelPathFromArtifactRelPath maps gi/kg to metadata', () => {
    expect(guessMetadataRelPathFromArtifactRelPath('metadata/x.gi.json')).toBe(
      'metadata/x.metadata.json',
    )
    expect(guessMetadataRelPathFromArtifactRelPath('x.kg.json')).toBe('x.metadata.json')
  })

  it('resolveEpisodeMetadataFromLoadedArtifacts matches loaded artifact episode id', () => {
    const data: ArtifactData = { episode_id: 'e1', nodes: [], edges: [] }
    const parsed = parseArtifact('foo.gi.json', data)
    const meta = resolveEpisodeMetadataFromLoadedArtifacts('e1', [parsed], [
      'metadata/foo.gi.json',
    ])
    expect(meta).toBe('metadata/foo.metadata.json')
  })

  it('logicalEpisodeIdsMatchingMetadataPath finds Episode by metadata path', () => {
    const data: ArtifactData = {
      nodes: [
        {
          id: 'g:episode:ep99',
          type: 'Episode',
          properties: { metadata_relative_path: 'metadata/foo.metadata.json' },
        },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(
      logicalEpisodeIdsMatchingMetadataPath(parsed, 'metadata/foo.metadata.json'),
    ).toEqual(['ep99'])
  })

  it('logicalEpisodeIdsForLibraryGraphSync prefers graph connections cy id', () => {
    const data: ArtifactData = { nodes: [], edges: [] }
    const parsed = parseArtifact('x.gi.json', data)
    expect(
      logicalEpisodeIdsForLibraryGraphSync(
        parsed,
        'metadata/foo.metadata.json',
        'g:episode:abc',
      ),
    ).toEqual(['abc'])
  })
})
