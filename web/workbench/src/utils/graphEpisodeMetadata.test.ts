import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ArtifactData, ParsedArtifact } from '../types/artifact'
import {
  findEpisodeGraphNodeIdForMetadataPath,
  findEpisodeGraphNodeIdForMetadataPathOrEpisodeId,
  graphCyIdRepresentsEpisodeNode,
  guessMetadataRelPathFromArtifactRelPath,
  logicalEpisodeIdFromGraphNodeId,
  logicalEpisodeIdsForLibraryGraphSync,
  logicalEpisodeIdsMatchingMetadataPath,
  metadataPathFromEpisodeProperties,
  normalizeCorpusMetadataPath,
  resolveEpisodeMetadataFromLoadedArtifacts,
  resolveEpisodeMetadataViaCorpusCatalog,
} from './graphEpisodeMetadata'
import { parseArtifact } from './parsing'
import * as corpusLibraryApi from '../api/corpusLibraryApi'

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

  it('resolveEpisodeMetadataFromLoadedArtifacts prefers sourceCorpusRelPath when parallel indices are wrong', () => {
    const data: ArtifactData = { episode_id: 'e2', nodes: [], edges: [] }
    const parsed = parseArtifact('bar.gi.json', data, 'metadata/bar.gi.json')
    const meta = resolveEpisodeMetadataFromLoadedArtifacts('e2', [parsed], [
      'metadata/wrong.gi.json',
    ])
    expect(meta).toBe('metadata/bar.metadata.json')
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

  // --- logicalEpisodeIdFromGraphNodeId edge cases ---

  it('logicalEpisodeIdFromGraphNodeId returns null for empty / whitespace input', () => {
    expect(logicalEpisodeIdFromGraphNodeId('')).toBeNull()
    expect(logicalEpisodeIdFromGraphNodeId('   ')).toBeNull()
  })

  it('logicalEpisodeIdFromGraphNodeId returns null for unified prefix with empty rest', () => {
    expect(logicalEpisodeIdFromGraphNodeId('__unified_ep__:')).toBeNull()
    expect(logicalEpisodeIdFromGraphNodeId('__unified_ep__:   ')).toBeNull()
  })

  it('logicalEpisodeIdFromGraphNodeId trims surrounding whitespace before matching', () => {
    expect(logicalEpisodeIdFromGraphNodeId('  episode:abc  ')).toBe('abc')
    expect(logicalEpisodeIdFromGraphNodeId('__unified_ep__:  abc  ')).toBe('abc')
  })

  it('logicalEpisodeIdFromGraphNodeId handles the ep / kg:ep variants', () => {
    expect(logicalEpisodeIdFromGraphNodeId('g:ep:abc')).toBe('abc')
    expect(logicalEpisodeIdFromGraphNodeId('k:ep:abc')).toBe('abc')
    expect(logicalEpisodeIdFromGraphNodeId('g:kg:ep:abc')).toBe('abc')
    expect(logicalEpisodeIdFromGraphNodeId('k:kg:episode:abc')).toBe('abc')
  })

  it('logicalEpisodeIdFromGraphNodeId returns null when no prefix matches', () => {
    expect(logicalEpisodeIdFromGraphNodeId('g:topic:x')).toBeNull()
    expect(logicalEpisodeIdFromGraphNodeId('plain-id')).toBeNull()
  })

  // --- graphCyIdRepresentsEpisodeNode branches ---

  it('graphCyIdRepresentsEpisodeNode returns false for a non-Episode raw node even if id looks like an episode', () => {
    expect(
      graphCyIdRepresentsEpisodeNode('episode:abc', {
        id: 'episode:abc',
        type: 'Insight',
        properties: {},
      }),
    ).toBe(false)
  })

  it('graphCyIdRepresentsEpisodeNode returns false when raw node missing and id is not episode-shaped', () => {
    expect(graphCyIdRepresentsEpisodeNode('not-an-episode', null)).toBe(false)
  })

  // --- metadataPathFromEpisodeProperties branches ---

  it('metadataPathFromEpisodeProperties returns null for null node', () => {
    expect(metadataPathFromEpisodeProperties(null)).toBeNull()
  })

  it('metadataPathFromEpisodeProperties returns null when properties absent or not an object', () => {
    expect(
      metadataPathFromEpisodeProperties({ id: 'x', type: 'Episode' }),
    ).toBeNull()
    expect(
      metadataPathFromEpisodeProperties({
        id: 'x',
        type: 'Episode',
        properties: 'nope' as unknown as Record<string, unknown>,
      }),
    ).toBeNull()
  })

  it('metadataPathFromEpisodeProperties reads metadata_relative_path and trims it', () => {
    expect(
      metadataPathFromEpisodeProperties({
        id: 'x',
        type: 'Episode',
        properties: { metadata_relative_path: '  metadata/a.metadata.json  ' },
      }),
    ).toBe('metadata/a.metadata.json')
  })

  it('metadataPathFromEpisodeProperties falls back to source_metadata_relative_path', () => {
    expect(
      metadataPathFromEpisodeProperties({
        id: 'x',
        type: 'Episode',
        properties: {
          source_metadata_relative_path: 'metadata/b.metadata.json',
        },
      }),
    ).toBe('metadata/b.metadata.json')
  })

  it('metadataPathFromEpisodeProperties ignores non-string / blank values', () => {
    expect(
      metadataPathFromEpisodeProperties({
        id: 'x',
        type: 'Episode',
        properties: {
          metadata_relative_path: 42 as unknown as string,
          source_metadata_relative_path: '   ',
        },
      }),
    ).toBeNull()
  })

  // --- guessMetadataRelPathFromArtifactRelPath branches ---

  it('guessMetadataRelPathFromArtifactRelPath returns null for empty / whitespace input', () => {
    expect(guessMetadataRelPathFromArtifactRelPath('')).toBeNull()
    expect(guessMetadataRelPathFromArtifactRelPath('   ')).toBeNull()
  })

  it('guessMetadataRelPathFromArtifactRelPath returns null for unrelated extensions', () => {
    expect(guessMetadataRelPathFromArtifactRelPath('metadata/x.json')).toBeNull()
    expect(guessMetadataRelPathFromArtifactRelPath('x.bridge.json')).toBeNull()
  })

  it('guessMetadataRelPathFromArtifactRelPath normalizes backslashes and is case-insensitive on extension', () => {
    expect(
      guessMetadataRelPathFromArtifactRelPath('metadata\\sub\\x.GI.JSON'),
    ).toBe('metadata/sub/x.metadata.json')
    expect(guessMetadataRelPathFromArtifactRelPath('a\\b\\x.KG.json')).toBe(
      'a/b/x.metadata.json',
    )
  })

  // --- resolveEpisodeMetadataFromLoadedArtifacts branches ---

  it('resolveEpisodeMetadataFromLoadedArtifacts returns null for blank logical id', () => {
    const data: ArtifactData = { episode_id: 'e1', nodes: [], edges: [] }
    const parsed = parseArtifact('foo.gi.json', data)
    expect(
      resolveEpisodeMetadataFromLoadedArtifacts('  ', [parsed], [
        'metadata/foo.gi.json',
      ]),
    ).toBeNull()
  })

  it('resolveEpisodeMetadataFromLoadedArtifacts skips artifacts without a usable episode id', () => {
    const noEp = parseArtifact('foo.gi.json', { nodes: [], edges: [] })
    const merged = parseArtifact('m.gi.json', {
      episode_id: 'merged:abc',
      nodes: [],
      edges: [],
    })
    expect(
      resolveEpisodeMetadataFromLoadedArtifacts('want', [noEp, merged], [
        'metadata/a.gi.json',
        'metadata/b.gi.json',
      ]),
    ).toBeNull()
  })

  it('resolveEpisodeMetadataFromLoadedArtifacts skips artifacts whose episode id does not match', () => {
    const other = parseArtifact('other.gi.json', {
      episode_id: 'other',
      nodes: [],
      edges: [],
    })
    expect(
      resolveEpisodeMetadataFromLoadedArtifacts('want', [other], [
        'metadata/other.gi.json',
      ]),
    ).toBeNull()
  })

  it('resolveEpisodeMetadataFromLoadedArtifacts returns null when the rel path resolves to empty', () => {
    const data: ArtifactData = { episode_id: 'e1', nodes: [], edges: [] }
    const parsed = parseArtifact('foo.gi.json', data)
    // No sourceCorpusRelPath and a blank selectedRelPaths entry → rel is '' → null.
    expect(
      resolveEpisodeMetadataFromLoadedArtifacts('e1', [parsed], ['   ']),
    ).toBeNull()
  })

  it('resolveEpisodeMetadataFromLoadedArtifacts returns null when the matching index has no rel path at all', () => {
    // When sourceCorpusRelPath is absent AND selectedRelPaths[i] is undefined
    // (missing parallel entry), rel coalesces to '' and the episode is skipped
    // gracefully (previously this threw a TypeError — fixed alongside #914 tests).
    const data: ArtifactData = { episode_id: 'e1', nodes: [], edges: [] }
    const parsed = parseArtifact('foo.gi.json', data)
    expect(resolveEpisodeMetadataFromLoadedArtifacts('e1', [parsed], [])).toBeNull()
  })

  it('resolveEpisodeMetadataFromLoadedArtifacts continues when the rel path is not guessable', () => {
    const data: ArtifactData = { episode_id: 'e1', nodes: [], edges: [] }
    const parsed = parseArtifact('foo.json', data)
    // selectedRelPaths has a non-gi/kg path → guess returns null → no map → null.
    expect(
      resolveEpisodeMetadataFromLoadedArtifacts('e1', [parsed], [
        'metadata/foo.json',
      ]),
    ).toBeNull()
  })

  it('resolveEpisodeMetadataFromLoadedArtifacts uses sourceCorpusRelPathByEpisodeId map when primary rel is not guessable', () => {
    const data: ArtifactData = { episode_id: 'e1', nodes: [], edges: [] }
    const parsed = parseArtifact('foo.json', data, 'metadata/foo.json')
    const withMap: ParsedArtifact = {
      ...parsed,
      sourceCorpusRelPathByEpisodeId: { e1: 'metadata/mapped.gi.json' },
    }
    expect(
      resolveEpisodeMetadataFromLoadedArtifacts('e1', [withMap], [
        'metadata/foo.json',
      ]),
    ).toBe('metadata/mapped.metadata.json')
  })

  it('resolveEpisodeMetadataFromLoadedArtifacts returns null when map entry is also not guessable', () => {
    const data: ArtifactData = { episode_id: 'e1', nodes: [], edges: [] }
    const parsed = parseArtifact('foo.json', data, 'metadata/foo.json')
    const withMap: ParsedArtifact = {
      ...parsed,
      sourceCorpusRelPathByEpisodeId: { e1: 'metadata/mapped.json' },
    }
    expect(
      resolveEpisodeMetadataFromLoadedArtifacts('e1', [withMap], [
        'metadata/foo.json',
      ]),
    ).toBeNull()
  })

  it('resolveEpisodeMetadataFromLoadedArtifacts ignores a map missing the wanted episode id', () => {
    const data: ArtifactData = { episode_id: 'e1', nodes: [], edges: [] }
    const parsed = parseArtifact('foo.json', data, 'metadata/foo.json')
    const withMap: ParsedArtifact = {
      ...parsed,
      sourceCorpusRelPathByEpisodeId: { someoneElse: 'metadata/x.gi.json' },
    }
    expect(
      resolveEpisodeMetadataFromLoadedArtifacts('e1', [withMap], [
        'metadata/foo.json',
      ]),
    ).toBeNull()
  })

  // --- normalizeCorpusMetadataPath ---

  it('normalizeCorpusMetadataPath trims and converts backslashes', () => {
    expect(normalizeCorpusMetadataPath('  a\\b\\c.json  ')).toBe('a/b/c.json')
    expect(normalizeCorpusMetadataPath('')).toBe('')
  })

  // --- findEpisodeGraphNodeIdForMetadataPath branches ---

  it('findEpisodeGraphNodeIdForMetadataPath returns null for blank want path', () => {
    const parsed = parseArtifact('x.gi.json', {
      nodes: [
        {
          id: 'g:episode:ep1',
          type: 'Episode',
          properties: { metadata_relative_path: 'metadata/a.metadata.json' },
        },
      ],
      edges: [],
    })
    expect(findEpisodeGraphNodeIdForMetadataPath(parsed, '   ')).toBeNull()
  })

  it('findEpisodeGraphNodeIdForMetadataPath returns null for null artifact', () => {
    expect(
      findEpisodeGraphNodeIdForMetadataPath(null, 'metadata/a.metadata.json'),
    ).toBeNull()
  })

  it('findEpisodeGraphNodeIdForMetadataPath skips null nodes, missing ids, non-Episodes, and metadata-less rows', () => {
    const want = 'metadata/foo.metadata.json'
    const data: ArtifactData = {
      nodes: [
        null as unknown as RawNodeLike,
        { id: null, type: 'Episode', properties: { metadata_relative_path: want } },
        { id: 'g:topic:t', type: 'Topic', properties: { metadata_relative_path: want } },
        { id: 'g:episode:noMeta', type: 'Episode', properties: {} },
        { id: 'g:episode:wrong', type: 'Episode', properties: { metadata_relative_path: 'metadata/other.metadata.json' } },
        { id: 'g:episode:hit', type: 'Episode', properties: { metadata_relative_path: want } },
      ] as unknown as ArtifactData['nodes'],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(findEpisodeGraphNodeIdForMetadataPath(parsed, want)).toBe(
      'g:episode:hit',
    )
  })

  it('findEpisodeGraphNodeIdForMetadataPath matches across path-separator drift', () => {
    const data: ArtifactData = {
      nodes: [
        {
          id: 'g:episode:ep1',
          type: 'Episode',
          properties: { metadata_relative_path: 'metadata\\foo.metadata.json' },
        },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(
      findEpisodeGraphNodeIdForMetadataPath(parsed, 'metadata/foo.metadata.json'),
    ).toBe('g:episode:ep1')
  })

  // --- findEpisodeGraphNodeIdForMetadataPathOrEpisodeId branches ---

  it('findEpisodeGraphNodeIdForMetadataPathOrEpisodeId returns the metadata hit without consulting fallback', () => {
    const want = 'metadata/foo.metadata.json'
    const data: ArtifactData = {
      nodes: [
        { id: 'g:episode:hit', type: 'Episode', properties: { metadata_relative_path: want } },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(parsed, want, 'whatever'),
    ).toBe('g:episode:hit')
  })

  it('findEpisodeGraphNodeIdForMetadataPathOrEpisodeId returns null when fallback id is blank/missing', () => {
    const data: ArtifactData = {
      nodes: [
        { id: 'g:episode:x', type: 'Episode', properties: { episode_id: 'eid' } },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    const wrong = 'metadata/none.metadata.json'
    expect(
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(parsed, wrong, '  '),
    ).toBeNull()
    expect(
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(parsed, wrong, null),
    ).toBeNull()
    expect(
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(parsed, wrong, undefined),
    ).toBeNull()
  })

  it('findEpisodeGraphNodeIdForMetadataPathOrEpisodeId returns null when artifact has no nodes', () => {
    const parsed = parseArtifact('x.gi.json', { nodes: [], edges: [] })
    expect(
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
        parsed,
        'metadata/none.metadata.json',
        'eid',
      ),
    ).toBeNull()
    expect(
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
        null,
        'metadata/none.metadata.json',
        'eid',
      ),
    ).toBeNull()
  })

  it('findEpisodeGraphNodeIdForMetadataPathOrEpisodeId matches fallback via node id', () => {
    const data: ArtifactData = {
      nodes: [
        { id: 'g:episode:eid', type: 'Episode', properties: {} },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
        parsed,
        'metadata/none.metadata.json',
        'eid',
      ),
    ).toBe('g:episode:eid')
  })

  it('findEpisodeGraphNodeIdForMetadataPathOrEpisodeId matches fallback via episode_id property when id does not encode it', () => {
    const data: ArtifactData = {
      nodes: [
        { id: 'opaque-node-1', type: 'Episode', properties: { episode_id: 'eid' } },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
        parsed,
        'metadata/none.metadata.json',
        'eid',
      ),
    ).toBe('opaque-node-1')
  })

  it('findEpisodeGraphNodeIdForMetadataPathOrEpisodeId skips bad nodes and returns null when nothing matches', () => {
    const data: ArtifactData = {
      nodes: [
        null as unknown as RawNodeLike,
        { id: 'g:topic:t', type: 'Topic', properties: { episode_id: 'eid' } },
        { id: null, type: 'Episode', properties: { episode_id: 'eid' } },
        { id: 'g:episode:nope', type: 'Episode', properties: { episode_id: 'other' } },
      ] as unknown as ArtifactData['nodes'],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
        parsed,
        'metadata/none.metadata.json',
        'eid',
      ),
    ).toBeNull()
  })

  // --- logicalEpisodeIdsMatchingMetadataPath branches ---

  it('logicalEpisodeIdsMatchingMetadataPath returns [] for blank path or null artifact', () => {
    const parsed = parseArtifact('x.gi.json', { nodes: [], edges: [] })
    expect(logicalEpisodeIdsMatchingMetadataPath(parsed, '   ')).toEqual([])
    expect(
      logicalEpisodeIdsMatchingMetadataPath(null, 'metadata/a.metadata.json'),
    ).toEqual([])
  })

  it('logicalEpisodeIdsMatchingMetadataPath falls back to episode_id property and de-duplicates', () => {
    const want = 'metadata/foo.metadata.json'
    const data: ArtifactData = {
      nodes: [
        null as unknown as RawNodeLike,
        { id: 'g:topic:t', type: 'Topic', properties: { metadata_relative_path: want } },
        { id: 'g:episode:noMeta', type: 'Episode', properties: {} },
        { id: 'g:episode:wrong', type: 'Episode', properties: { metadata_relative_path: 'metadata/other.metadata.json' } },
        // opaque id → no logical from id → falls back to episode_id property
        { id: 'opaque-1', type: 'Episode', properties: { metadata_relative_path: want, episode_id: 'epX' } },
        // duplicate logical id should be collapsed
        { id: 'g:episode:epX', type: 'Episode', properties: { metadata_relative_path: want } },
        { id: 'g:episode:epY', type: 'Episode', properties: { metadata_relative_path: want } },
      ] as unknown as ArtifactData['nodes'],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(logicalEpisodeIdsMatchingMetadataPath(parsed, want)).toEqual([
      'epX',
      'epY',
    ])
  })

  it('logicalEpisodeIdsMatchingMetadataPath skips rows with no resolvable logical id', () => {
    const want = 'metadata/foo.metadata.json'
    const data: ArtifactData = {
      nodes: [
        { id: 'opaque-no-eid', type: 'Episode', properties: { metadata_relative_path: want } },
        { id: null, type: 'Episode', properties: { metadata_relative_path: want } },
      ] as unknown as ArtifactData['nodes'],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(logicalEpisodeIdsMatchingMetadataPath(parsed, want)).toEqual([])
  })

  // --- logicalEpisodeIdsForLibraryGraphSync branches ---

  it('logicalEpisodeIdsForLibraryGraphSync falls back to metadata scan when cy id is blank', () => {
    const want = 'metadata/foo.metadata.json'
    const data: ArtifactData = {
      nodes: [
        { id: 'g:episode:ep99', type: 'Episode', properties: { metadata_relative_path: want } },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(
      logicalEpisodeIdsForLibraryGraphSync(parsed, want, '   '),
    ).toEqual(['ep99'])
    expect(logicalEpisodeIdsForLibraryGraphSync(parsed, want, null)).toEqual([
      'ep99',
    ])
  })

  it('logicalEpisodeIdsForLibraryGraphSync scans metadata when cy id is not episode-shaped', () => {
    const want = 'metadata/foo.metadata.json'
    const data: ArtifactData = {
      nodes: [
        { id: 'g:episode:ep99', type: 'Episode', properties: { metadata_relative_path: want } },
      ],
      edges: [],
    }
    const parsed = parseArtifact('x.gi.json', data)
    expect(
      logicalEpisodeIdsForLibraryGraphSync(parsed, want, 'g:topic:not-episode'),
    ).toEqual(['ep99'])
  })

  // --- resolveEpisodeMetadataViaCorpusCatalog branches ---

  describe('resolveEpisodeMetadataViaCorpusCatalog', () => {
    beforeEach(() => {
      vi.restoreAllMocks()
    })
    afterEach(() => {
      vi.restoreAllMocks()
    })

    function mockEpisodes(
      pages: Array<{
        items: Array<{ episode_id: string | null; metadata_relative_path: string }>
        next_cursor: string | null
      }>,
    ) {
      const spy = vi.spyOn(corpusLibraryApi, 'fetchCorpusEpisodes')
      let call = 0
      spy.mockImplementation(async () => {
        const page = pages[Math.min(call, pages.length - 1)]
        call++
        return {
          path: 'corpus',
          feed_id: null,
          items: page.items as unknown as corpusLibraryApi.CorpusEpisodeListItem[],
          next_cursor: page.next_cursor,
        }
      })
      return spy
    }

    it('returns null for blank episode id or corpus path', async () => {
      const spy = vi.spyOn(corpusLibraryApi, 'fetchCorpusEpisodes')
      expect(
        await resolveEpisodeMetadataViaCorpusCatalog('corpus', '   '),
      ).toBeNull()
      expect(
        await resolveEpisodeMetadataViaCorpusCatalog('   ', 'eid'),
      ).toBeNull()
      expect(spy).not.toHaveBeenCalled()
    })

    it('finds the metadata path on the first page', async () => {
      const spy = mockEpisodes([
        {
          items: [
            { episode_id: 'other', metadata_relative_path: 'metadata/o.metadata.json' },
            { episode_id: 'eid', metadata_relative_path: '  metadata/hit.metadata.json  ' },
          ],
          next_cursor: 'next',
        },
      ])
      expect(
        await resolveEpisodeMetadataViaCorpusCatalog('corpus', 'eid'),
      ).toBe('metadata/hit.metadata.json')
      expect(spy).toHaveBeenCalledTimes(1)
    })

    it('paginates using next_cursor until a hit is found', async () => {
      const spy = mockEpisodes([
        { items: [{ episode_id: 'a', metadata_relative_path: 'metadata/a.metadata.json' }], next_cursor: 'c1' },
        { items: [{ episode_id: 'eid', metadata_relative_path: 'metadata/hit.metadata.json' }], next_cursor: 'c2' },
      ])
      expect(
        await resolveEpisodeMetadataViaCorpusCatalog('corpus', 'eid'),
      ).toBe('metadata/hit.metadata.json')
      expect(spy).toHaveBeenCalledTimes(2)
    })

    it('stops paging when next_cursor is null and returns null on no match', async () => {
      const spy = mockEpisodes([
        { items: [{ episode_id: 'a', metadata_relative_path: 'metadata/a.metadata.json' }], next_cursor: null },
      ])
      expect(
        await resolveEpisodeMetadataViaCorpusCatalog('corpus', 'eid'),
      ).toBeNull()
      expect(spy).toHaveBeenCalledTimes(1)
    })

    it('respects maxPages and returns null when exhausted', async () => {
      const spy = mockEpisodes([
        { items: [{ episode_id: 'x', metadata_relative_path: 'metadata/x.metadata.json' }], next_cursor: 'always' },
      ])
      expect(
        await resolveEpisodeMetadataViaCorpusCatalog('corpus', 'eid', 3),
      ).toBeNull()
      expect(spy).toHaveBeenCalledTimes(3)
    })

    it('ignores items whose metadata_relative_path is blank', async () => {
      const spy = mockEpisodes([
        { items: [{ episode_id: 'eid', metadata_relative_path: '   ' }], next_cursor: null },
      ])
      expect(
        await resolveEpisodeMetadataViaCorpusCatalog('corpus', 'eid'),
      ).toBeNull()
      expect(spy).toHaveBeenCalledTimes(1)
    })

    it('cancels before the first fetch when shouldCancel is true', async () => {
      const spy = vi.spyOn(corpusLibraryApi, 'fetchCorpusEpisodes')
      expect(
        await resolveEpisodeMetadataViaCorpusCatalog(
          'corpus',
          'eid',
          8,
          () => true,
        ),
      ).toBeNull()
      expect(spy).not.toHaveBeenCalled()
    })

    it('cancels after the fetch resolves (second cancel check)', async () => {
      const spy = mockEpisodes([
        { items: [{ episode_id: 'eid', metadata_relative_path: 'metadata/hit.metadata.json' }], next_cursor: null },
      ])
      let calls = 0
      const shouldCancel = () => {
        // false on the pre-fetch check, true on the post-fetch check
        calls++
        return calls > 1
      }
      expect(
        await resolveEpisodeMetadataViaCorpusCatalog(
          'corpus',
          'eid',
          8,
          shouldCancel,
        ),
      ).toBeNull()
      expect(spy).toHaveBeenCalledTimes(1)
    })

    it('treats a missing next_cursor field as the end of pagination', async () => {
      const spy = vi.spyOn(corpusLibraryApi, 'fetchCorpusEpisodes')
      spy.mockResolvedValue({
        path: 'corpus',
        feed_id: null,
        items: [],
        next_cursor: null,
      } as unknown as corpusLibraryApi.CorpusEpisodesResponse)
      expect(
        await resolveEpisodeMetadataViaCorpusCatalog('corpus', 'eid'),
      ).toBeNull()
      expect(spy).toHaveBeenCalledTimes(1)
    })
  })
})

type RawNodeLike = { id: unknown; type: string; properties?: Record<string, unknown> }
