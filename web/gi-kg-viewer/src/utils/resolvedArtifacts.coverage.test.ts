import { afterEach, describe, expect, it } from 'vitest'

import type { CorpusResolvedEpisodeArtifact } from '../api/corpusLibraryApi'
import type { ParsedArtifact } from '../types/artifact'
import {
  artifactRelPathsForResolvedRow,
  sortResolvedArtifactsNewestFirst,
} from './resolvedArtifacts'
import { episodeArtifactPaths } from '../extensions'

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
