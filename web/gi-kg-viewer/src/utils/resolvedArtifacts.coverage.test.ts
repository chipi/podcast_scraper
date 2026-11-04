import { describe, expect, it } from 'vitest'

import type { CorpusResolvedEpisodeArtifact } from '../api/corpusLibraryApi'
import type { ParsedArtifact } from '../types/artifact'
import {
  artifactRelPathsForResolvedRow,
  episodeIdsFromParsedArtifacts,
  sortResolvedArtifactsNewestFirst,
} from './resolvedArtifacts'

function parsed(kind: 'gi' | 'kg', data: unknown): ParsedArtifact {
  return {
    name: 'x',
    kind,
    episodeId: null,
    nodes: 0,
    edges: 0,
    nodeTypes: {},
    data,
  } as ParsedArtifact
}

function row(
  episode_id: string,
  publish_date: string | null,
  extra: Partial<CorpusResolvedEpisodeArtifact> = {},
): CorpusResolvedEpisodeArtifact {
  return {
    episode_id,
    publish_date,
    gi_relative_path: null,
    kg_relative_path: null,
    bridge_relative_path: null,
    ...extra,
  }
}

describe('episodeIdsFromParsedArtifacts', () => {
  it('collects trimmed episode ids from gi artifacts only', () => {
    const ids = episodeIdsFromParsedArtifacts([
      parsed('gi', { episode_id: '  ep:1 ' }),
      parsed('kg', { episode_id: 'ep:kg' }),
      parsed('gi', { episode_id: 'ep:2' }),
    ])
    expect([...ids].sort()).toEqual(['ep:1', 'ep:2'])
  })

  it('skips missing, blank and non-string ids and dedupes', () => {
    const ids = episodeIdsFromParsedArtifacts([
      parsed('gi', {}),
      parsed('gi', { episode_id: '   ' }),
      parsed('gi', { episode_id: 42 }),
      parsed('gi', { episode_id: 'ep:1' }),
      parsed('gi', { episode_id: 'ep:1' }),
    ])
    expect([...ids]).toEqual(['ep:1'])
  })

  it('returns an empty set for no artifacts', () => {
    expect(episodeIdsFromParsedArtifacts([]).size).toBe(0)
  })
})

describe('sortResolvedArtifactsNewestFirst', () => {
  it('orders by publish date descending', () => {
    const out = sortResolvedArtifactsNewestFirst([
      row('a', '2026-01-01'),
      row('b', '2026-03-01T10:00:00Z'),
      row('c', '2026-02-01'),
    ])
    expect(out.map((r) => r.episode_id)).toEqual(['b', 'c', 'a'])
  })

  it('breaks date ties by episode_id ascending', () => {
    const out = sortResolvedArtifactsNewestFirst([
      row('z', '2026-01-01'),
      row('a', '2026-01-01T23:00:00Z'),
    ])
    expect(out.map((r) => r.episode_id)).toEqual(['a', 'z'])
  })

  it('sinks null, blank and unparseable dates to the end', () => {
    const out = sortResolvedArtifactsNewestFirst([
      row('n', null),
      row('blank', '  '),
      row('bad', 'garbage-date'),
      row('d', '2020-01-01'),
    ])
    expect(out[0].episode_id).toBe('d')
    expect(out.slice(1).map((r) => r.episode_id)).toEqual(['bad', 'blank', 'n'])
  })

  it('does not mutate its input', () => {
    const input = [row('a', '2026-01-01'), row('b', '2026-02-01')]
    sortResolvedArtifactsNewestFirst(input)
    expect(input.map((r) => r.episode_id)).toEqual(['a', 'b'])
  })
})

describe('artifactRelPathsForResolvedRow', () => {
  it('returns gi, kg, bridge in order, trimmed', () => {
    const out = artifactRelPathsForResolvedRow(
      row('a', null, {
        gi_relative_path: ' m/a.gi.json ',
        kg_relative_path: 'm/a.kg.json',
        bridge_relative_path: 'm/a.bridge.json',
      }),
    )
    expect(out).toEqual(['m/a.gi.json', 'm/a.kg.json', 'm/a.bridge.json'])
  })

  it('omits null and blank paths', () => {
    expect(
      artifactRelPathsForResolvedRow(row('a', null, { kg_relative_path: '  ' })),
    ).toEqual([])
    expect(
      artifactRelPathsForResolvedRow(row('a', null, { bridge_relative_path: 'b.json' })),
    ).toEqual(['b.json'])
  })
})
