import { describe, expect, it } from 'vitest'
import type { ArtifactData } from '../types/artifact'
import {
  findEpisodeGraphNodeIdForMetadataPath,
  findEpisodeGraphNodeIdForMetadataPathOrEpisodeId,
  graphCyIdRepresentsEpisodeNode,
  guessMetadataRelPathFromArtifactRelPath,
  logicalEpisodeIdFromGraphNodeId,
  logicalEpisodeIdsForLibraryGraphSync,
  logicalEpisodeIdsMatchingMetadataPath,
  resolveEpisodeMetadataFromLoadedArtifacts,
} from './graphEpisodeMetadata'
import { parseArtifact } from './parsing'

describe('graphEpisodeMetadata', () => {
  it('graphCyIdRepresentsEpisodeNode uses id when raw node is missing', () => {
    expect(graphCyIdRepresentsEpisodeNode('episode:abc', null)).toBe(true)
    expect(graphCyIdRepresentsEpisodeNode('g:topic:x', null)).toBe(false)
  })

  it('graphCyIdRepresentsEpisodeNode trusts raw type when present', () => {
    expect(
      graphCyIdRepresentsEpisodeNode('x', {
        id: 'x',
        type: 'Episode',
        properties: {},
      }),
    ).toBe(true)
    expect(
      graphCyIdRepresentsEpisodeNode('episode:ignored', {
        id: 'g:topic:t',
        type: 'Topic',
        properties: {},
      }),
    ).toBe(false)
  })

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

  it('findEpisodeGraphNodeIdForMetadataPath returns Episode id for merged graph', () => {
    const want = 'metadata/foo.metadata.json'
    const data: ArtifactData = {
      nodes: [
        {
          id: 'g:episode:ep99',
          type: 'Episode',
          properties: { metadata_relative_path: want },
        },
        {
          id: 'g:episode:ep100',
          type: 'Episode',
          properties: { metadata_relative_path: 'other/metadata.json' },
        },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(findEpisodeGraphNodeIdForMetadataPath(parsed, want)).toBe('g:episode:ep99')
  })

  it('findEpisodeGraphNodeIdForMetadataPath picks lexicographically first when several match', () => {
    const want = 'metadata/foo.metadata.json'
    const data: ArtifactData = {
      nodes: [
        {
          id: 'z:episode:a',
          type: 'Episode',
          properties: { metadata_relative_path: want },
        },
        {
          id: 'a:episode:b',
          type: 'Episode',
          properties: { metadata_relative_path: want },
        },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(findEpisodeGraphNodeIdForMetadataPath(parsed, want)).toBe('a:episode:b')
  })

  it('findEpisodeGraphNodeIdForMetadataPathOrEpisodeId falls back to episode_id when metadata mismatches', () => {
    const wrongMeta = 'metadata/wrong.metadata.json'
    const data: ArtifactData = {
      nodes: [
        {
          id: '__unified_ep__:25560386-3aa0-11f1-bad6-37d650dbf5f0',
          type: 'Episode',
          properties: {
            metadata_relative_path: 'metadata/correct.metadata.json',
            episode_id: '25560386-3aa0-11f1-bad6-37d650dbf5f0',
          },
        },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(findEpisodeGraphNodeIdForMetadataPath(parsed, wrongMeta)).toBeNull()
    expect(
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
        parsed,
        wrongMeta,
        '25560386-3aa0-11f1-bad6-37d650dbf5f0',
      ),
    ).toBe('__unified_ep__:25560386-3aa0-11f1-bad6-37d650dbf5f0')
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
