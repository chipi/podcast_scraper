import { describe, expect, it } from 'vitest'
import type { TopicClustersDocument } from '../api/corpusTopicClustersApi'
import type { ParsedArtifact } from '../types/artifact'
import { formatLocalYmd } from './localCalendarDate'
import {
  GRAPH_DEFAULT_EPISODE_CAP,
  GRAPH_SCORE_RECENCY_MIN,
  GRAPH_SCORE_RECENCY_MAX,
  GRAPH_SCORE_TOPIC_CLUSTER_BONUS,
  GRAPH_SCORE_ALL_TIME_DECAY_DAYS,
  GRAPH_SCORE_GI_DENSITY_MAX,
  calendarPublishYmdFromParsedArtifact,
  episodeIdsInTopicClustersForGraphScoring,
  episodeStemFromArtifactRelPath,
  isValidPublishYmd,
  selectParsedArtifactsForGraphLoad,
  selectRelPathsForGraphLoad,
  stemMatchesTopicClusterEpisodeId,
} from './graphEpisodeSelection'

describe('selectRelPathsForGraphLoad', () => {
  const rows = [
    { relative_path: 'm/a.gi.json', kind: 'gi', publish_date: '2024-01-10' },
    { relative_path: 'm/a.kg.json', kind: 'kg', publish_date: '2024-01-10' },
    { relative_path: 'm/a.bridge.json', kind: 'bridge', publish_date: '2024-01-10' },
    { relative_path: 'm/b.gi.json', kind: 'gi', publish_date: '2024-06-20' },
    { relative_path: 'm/b.kg.json', kind: 'kg', publish_date: '2024-06-20' },
  ]

  it('returns all paths for two episodes when cap is high', () => {
    const r = selectRelPathsForGraphLoad(rows, '', 10)
    expect(r.episodeCount).toBe(2)
    expect(r.wasCapped).toBe(false)
    expect(r.selectedRelPaths.length).toBe(5)
  })

  it('caps at N episodes for all-time lens (recency-ordered when no cluster signal)', () => {
    const many = [
      ...rows,
      { relative_path: 'm/c.gi.json', kind: 'gi', publish_date: '2024-03-01' },
      { relative_path: 'm/c.kg.json', kind: 'kg', publish_date: '2024-03-01' },
    ]
    const r = selectRelPathsForGraphLoad(many, '', 2)
    expect(r.episodeCount).toBe(2)
    expect(r.wasCapped).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/b.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/c.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/a.'))).toBe(false)
  })

  it('filters by sinceYmd', () => {
    const r = selectRelPathsForGraphLoad(rows, '2024-06-01', 10)
    expect(r.episodeCount).toBe(1)
    expect(r.selectedRelPaths.every((p) => p.includes('/b.'))).toBe(true)
  })

  it('prefers topic-cluster episode over slightly newer non-cluster when scores tie up', () => {
    const pool = [
      { relative_path: 'm/new.gi.json', kind: 'gi', publish_date: '2024-06-25' },
      { relative_path: 'm/new.kg.json', kind: 'kg', publish_date: '2024-06-25' },
      { relative_path: 'm/mid.gi.json', kind: 'gi', publish_date: '2024-06-10' },
      { relative_path: 'm/mid.kg.json', kind: 'kg', publish_date: '2024-06-10' },
      { relative_path: 'm/old.gi.json', kind: 'gi', publish_date: '2024-01-05' },
      { relative_path: 'm/old.kg.json', kind: 'kg', publish_date: '2024-01-05' },
    ]
    const doc: TopicClustersDocument = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:t',
          members: [{ topic_id: 'topic:x', episode_ids: ['mid'] }],
        },
      ],
    }
    const r = selectRelPathsForGraphLoad(pool, '', 2, doc)
    expect(r.episodeCount).toBe(2)
    expect(r.selectedRelPaths.some((p) => p.includes('/mid.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/new.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/old.'))).toBe(false)
  })

  it('uses default cap constant', () => {
    // Interim ceiling pending the large-graph layout work (#967); see the constant's
    // doc comment for the stress-test rationale (cose is ~O(n²)).
    expect(GRAPH_DEFAULT_EPISODE_CAP).toBe(25)
  })
})

describe('topic cluster scoring helpers', () => {
  it('collects episode ids from cluster members', () => {
    const doc: TopicClustersDocument = {
      clusters: [
        {
          members: [
            { topic_id: 'a', episode_ids: ['e1', 'e2'] },
            { topic_id: 'b', episode_ids: ['e3'] },
          ],
        },
      ],
    }
    expect(episodeIdsInTopicClustersForGraphScoring(doc)).toEqual(new Set(['e1', 'e2', 'e3']))
  })

  it('matches stem basename to cluster episode id', () => {
    const ids = new Set(['ep42'])
    expect(stemMatchesTopicClusterEpisodeId('feeds/x/ep42', ids)).toBe(true)
    expect(stemMatchesTopicClusterEpisodeId('ep42', ids)).toBe(true)
    expect(stemMatchesTopicClusterEpisodeId('feeds/x/other', ids)).toBe(false)
  })

  it('adds cluster bonus so clustered episode can rank above slightly newer non-cluster', () => {
    const pool = [
      { relative_path: 'pods/z.gi.json', kind: 'gi', publish_date: '2024-06-30' },
      { relative_path: 'pods/z.kg.json', kind: 'kg', publish_date: '2024-06-30' },
      { relative_path: 'pods/y.gi.json', kind: 'gi', publish_date: '2024-06-15' },
      { relative_path: 'pods/y.kg.json', kind: 'kg', publish_date: '2024-06-15' },
      { relative_path: 'pods/x.gi.json', kind: 'gi', publish_date: '2024-06-10' },
      { relative_path: 'pods/x.kg.json', kind: 'kg', publish_date: '2024-06-10' },
    ]
    const doc: TopicClustersDocument = {
      clusters: [
        {
          members: [{ topic_id: 'topic:t', episode_ids: ['x'] }],
        },
      ],
    }
    const r = selectRelPathsForGraphLoad(pool, '', 2, doc)
    expect(r.selectedRelPaths.some((p) => p.includes('/z.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/x.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/y.'))).toBe(false)
  })
})

describe('GRAPH_SCORE_TOPIC_CLUSTER_BONUS', () => {
  it('is the documented default cluster bonus', () => {
    expect(GRAPH_SCORE_TOPIC_CLUSTER_BONUS).toBe(0.4)
  })
})

describe('scoring tunable constants', () => {
  it('exposes the documented recency floor/ceiling, decay window, and density max', () => {
    expect(GRAPH_SCORE_RECENCY_MIN).toBe(0.2)
    expect(GRAPH_SCORE_RECENCY_MAX).toBe(1.0)
    expect(GRAPH_SCORE_ALL_TIME_DECAY_DAYS).toBe(90)
    expect(GRAPH_SCORE_GI_DENSITY_MAX).toBe(0.4)
  })
})

describe('isValidPublishYmd', () => {
  it('accepts a well-formed YYYY-MM-DD (with surrounding whitespace trimmed)', () => {
    expect(isValidPublishYmd('2024-01-10')).toBe(true)
    expect(isValidPublishYmd('  2024-01-10  ')).toBe(true)
  })

  it('rejects empty, partial, datetime, or malformed strings', () => {
    expect(isValidPublishYmd('')).toBe(false)
    expect(isValidPublishYmd('2024-1-1')).toBe(false)
    expect(isValidPublishYmd('2024-01-10T12:00:00')).toBe(false)
    expect(isValidPublishYmd('not-a-date')).toBe(false)
    expect(isValidPublishYmd('20240110')).toBe(false)
  })
})

describe('episodeStemFromArtifactRelPath', () => {
  it('strips known artifact suffixes', () => {
    expect(episodeStemFromArtifactRelPath('feeds/x/ep.gi.json')).toBe('feeds/x/ep')
    expect(episodeStemFromArtifactRelPath('feeds/x/ep.kg.json')).toBe('feeds/x/ep')
    expect(episodeStemFromArtifactRelPath('feeds/x/ep.bridge.json')).toBe('feeds/x/ep')
  })

  it('normalizes backslashes to forward slashes and trims', () => {
    expect(episodeStemFromArtifactRelPath('  feeds\\x\\ep.gi.json ')).toBe('feeds/x/ep')
  })

  it('returns the normalized path unchanged when no known suffix matches', () => {
    expect(episodeStemFromArtifactRelPath('feeds/x/ep.other.json')).toBe('feeds/x/ep.other.json')
    expect(episodeStemFromArtifactRelPath('feeds/x/ep')).toBe('feeds/x/ep')
  })
})

describe('episodeIdsInTopicClustersForGraphScoring edge cases', () => {
  it('returns an empty set for null/undefined docs', () => {
    expect(episodeIdsInTopicClustersForGraphScoring(null)).toEqual(new Set())
    expect(episodeIdsInTopicClustersForGraphScoring(undefined)).toEqual(new Set())
  })

  it('returns an empty set when clusters is not an array', () => {
    const doc = { clusters: 'nope' } as unknown as TopicClustersDocument
    expect(episodeIdsInTopicClustersForGraphScoring(doc)).toEqual(new Set())
  })

  it('skips members that are not arrays and member episode_ids that are not arrays', () => {
    const doc = {
      clusters: [
        { members: 'nope' },
        { members: [{ topic_id: 'a', episode_ids: 'nope' }, { topic_id: 'b', episode_ids: ['e1'] }] },
        {},
      ],
    } as unknown as TopicClustersDocument
    expect(episodeIdsInTopicClustersForGraphScoring(doc)).toEqual(new Set(['e1']))
  })

  it('trims ids and ignores non-string and blank entries', () => {
    const doc = {
      clusters: [
        { members: [{ topic_id: 'a', episode_ids: ['  e1  ', '', '   ', 7, null] }] },
      ],
    } as unknown as TopicClustersDocument
    expect(episodeIdsInTopicClustersForGraphScoring(doc)).toEqual(new Set(['e1']))
  })
})

describe('stemMatchesTopicClusterEpisodeId edge cases', () => {
  it('returns false for an empty stem or empty cluster id set', () => {
    expect(stemMatchesTopicClusterEpisodeId('', new Set(['ep1']))).toBe(false)
    expect(stemMatchesTopicClusterEpisodeId('  ', new Set(['ep1']))).toBe(false)
    expect(stemMatchesTopicClusterEpisodeId('feeds/x/ep1', new Set())).toBe(false)
  })

  it('matches on the full-stem equality path', () => {
    expect(stemMatchesTopicClusterEpisodeId('feeds/x/ep1', new Set(['feeds/x/ep1']))).toBe(true)
  })

  it('normalizes backslashes before matching', () => {
    expect(stemMatchesTopicClusterEpisodeId('feeds\\x\\ep1', new Set(['ep1']))).toBe(true)
  })

  it('skips blank cluster ids while still finding a later valid match', () => {
    expect(stemMatchesTopicClusterEpisodeId('ep1', new Set(['  ', 'ep1']))).toBe(true)
  })

  it('returns false when no cluster id matches', () => {
    expect(stemMatchesTopicClusterEpisodeId('feeds/x/ep1', new Set(['  ', 'ep9']))).toBe(false)
  })

  it('matches a basename when the stem has no leading slash', () => {
    expect(stemMatchesTopicClusterEpisodeId('ep1', new Set(['ep1']))).toBe(true)
  })

  it('matches on the "…/id" suffix path without a full-stem or basename equality', () => {
    // id contains a slash, so basename comparison can never match — only the
    // endsWith(`/${e}`) branch can succeed here.
    expect(stemMatchesTopicClusterEpisodeId('feeds/x/sub/ep1', new Set(['sub/ep1']))).toBe(true)
  })
})

describe('selectRelPathsForGraphLoad additional branches', () => {
  it('returns the empty result when no rows have a graph-relevant kind', () => {
    const r = selectRelPathsForGraphLoad(
      [{ relative_path: 'm/a.txt', kind: 'transcript', publish_date: '2024-01-10' }],
      '',
      10,
    )
    expect(r).toEqual({ selectedRelPaths: [], wasCapped: false, episodeCount: 0 })
  })

  it('returns the empty result when the dated window filters everything out', () => {
    const r = selectRelPathsForGraphLoad(
      [{ relative_path: 'm/a.gi.json', kind: 'gi', publish_date: '2024-01-10' }],
      '2030-01-01',
      10,
    )
    expect(r).toEqual({ selectedRelPaths: [], wasCapped: false, episodeCount: 0 })
  })

  it('clamps a non-positive cap up to 1', () => {
    const rows = [
      { relative_path: 'm/a.gi.json', kind: 'gi', publish_date: '2024-01-10' },
      { relative_path: 'm/b.gi.json', kind: 'gi', publish_date: '2024-06-20' },
    ]
    const r = selectRelPathsForGraphLoad(rows, '', 0)
    expect(r.episodeCount).toBe(1)
    expect(r.wasCapped).toBe(true)
    expect(r.selectedRelPaths).toEqual(['m/b.gi.json'])
  })

  it('skips blank relative paths and rows whose stem normalizes to nothing-but-suffix', () => {
    const rows = [
      { relative_path: '   ', kind: 'gi', publish_date: '2024-01-10' },
      { relative_path: 'm/a.gi.json', kind: 'gi', publish_date: '2024-06-20' },
    ]
    const r = selectRelPathsForGraphLoad(rows, '', 10)
    expect(r.episodeCount).toBe(1)
    expect(r.selectedRelPaths).toEqual(['m/a.gi.json'])
  })

  it('treats invalid/blank publish dates as undated (kept in all-time, recency floor)', () => {
    const rows = [
      { relative_path: 'm/dated.gi.json', kind: 'gi', publish_date: '2024-06-20' },
      { relative_path: 'm/undated.gi.json', kind: 'gi', publish_date: 'garbage' },
    ]
    const r = selectRelPathsForGraphLoad(rows, '', 10)
    expect(r.episodeCount).toBe(2)
    // dated episode (higher recency) sorts before the undated floor-scored one
    expect(r.selectedRelPaths).toContain('m/dated.gi.json')
    expect(r.selectedRelPaths).toContain('m/undated.gi.json')
  })

  it('keeps a later valid date when the same stem appears across kinds out of order', () => {
    const rows = [
      { relative_path: 'm/a.kg.json', kind: 'kg', publish_date: '2024-06-20' },
      { relative_path: 'm/a.gi.json', kind: 'gi', publish_date: '2024-01-10' },
    ]
    const r = selectRelPathsForGraphLoad(rows, '2024-06-01', 10)
    // newest date (2024-06-20) wins for the stem, so it survives the since filter
    expect(r.episodeCount).toBe(1)
    expect(r.selectedRelPaths).toEqual(['m/a.gi.json', 'm/a.kg.json'])
  })

  it('ramps dated-lens recency linearly across a non-zero span (newest > oldest)', () => {
    const rows = [
      { relative_path: 'm/oldest.gi.json', kind: 'gi', publish_date: '2024-01-01' },
      { relative_path: 'm/mid.gi.json', kind: 'gi', publish_date: '2024-03-01' },
      { relative_path: 'm/newest.gi.json', kind: 'gi', publish_date: '2024-06-01' },
    ]
    // dated lens with a real span -> linear interpolation; newest ranks first
    const r = selectRelPathsForGraphLoad(rows, '2024-01-01', 1)
    expect(r.episodeCount).toBe(1)
    expect(r.selectedRelPaths).toEqual(['m/newest.gi.json'])
  })

  it('handles a single dated episode (dated span <= 0 -> recency ceiling)', () => {
    const rows = [
      { relative_path: 'm/a.gi.json', kind: 'gi', publish_date: '2024-06-20' },
      { relative_path: 'm/b.gi.json', kind: 'gi', publish_date: '2024-06-20' },
    ]
    const r = selectRelPathsForGraphLoad(rows, '2024-06-01', 10)
    expect(r.episodeCount).toBe(2)
    expect(r.wasCapped).toBe(false)
  })

  it('falls back to the all-time decay floor for episodes older than the decay window', () => {
    const rows = [
      { relative_path: 'm/recent.gi.json', kind: 'gi', publish_date: '2024-06-20' },
      // well over GRAPH_SCORE_ALL_TIME_DECAY_DAYS before the newest
      { relative_path: 'm/ancient.gi.json', kind: 'gi', publish_date: '2020-01-01' },
    ]
    const r = selectRelPathsForGraphLoad(rows, '', 1)
    expect(r.episodeCount).toBe(1)
    expect(r.selectedRelPaths).toEqual(['m/recent.gi.json'])
  })

  it('tie-breaks two undated all-time episodes by stem ascending', () => {
    const rows = [
      { relative_path: 'm/zeta.gi.json', kind: 'gi', publish_date: '' },
      { relative_path: 'm/alpha.gi.json', kind: 'gi', publish_date: '' },
    ]
    const r = selectRelPathsForGraphLoad(rows, '', 1)
    expect(r.episodeCount).toBe(1)
    // equal score + equal (placeholder) publish key -> stem ascending picks 'alpha'
    expect(r.selectedRelPaths).toEqual(['m/alpha.gi.json'])
  })
})

describe('calendarPublishYmdFromParsedArtifact', () => {
  const makeArtifact = (nodes: ParsedArtifact['data']['nodes']): ParsedArtifact => ({
    name: 'x.gi.json',
    kind: 'gi',
    episodeId: null,
    nodes: nodes?.length ?? 0,
    edges: 0,
    nodeTypes: {},
    data: { nodes },
  })

  it('uses an Episode node publish_date when present and parseable', () => {
    const art = makeArtifact([
      { type: 'Episode', properties: { publish_date: '2024-03-15T00:00:00Z' } },
    ])
    expect(calendarPublishYmdFromParsedArtifact(art, 0)).toBe(
      formatLocalYmd(new Date(Date.parse('2024-03-15T00:00:00Z'))),
    )
  })

  it('falls back to file mtime when there is no Episode node', () => {
    const art = makeArtifact([{ type: 'Topic', properties: {} }])
    const mtime = Date.parse('2024-09-01T12:00:00')
    expect(calendarPublishYmdFromParsedArtifact(art, mtime)).toBe(formatLocalYmd(new Date(mtime)))
  })

  it('falls back to mtime when nodes is missing entirely', () => {
    const art = { ...makeArtifact([]), data: {} } as ParsedArtifact
    const mtime = Date.parse('2024-09-01T12:00:00')
    expect(calendarPublishYmdFromParsedArtifact(art, mtime)).toBe(formatLocalYmd(new Date(mtime)))
  })

  it('skips null nodes, non-Episode nodes, and missing/unparseable publish_date', () => {
    const mtime = Date.parse('2024-09-01T12:00:00')
    const artNullPd = makeArtifact([
      null,
      { type: 'Topic', properties: { publish_date: '2024-01-01' } },
      { type: 'Episode', properties: {} },
      { type: 'Episode', properties: { publish_date: 'not-a-date' } },
    ] as unknown as ParsedArtifact['data']['nodes'])
    expect(calendarPublishYmdFromParsedArtifact(artNullPd, mtime)).toBe(formatLocalYmd(new Date(mtime)))
  })

  it('uses the first Episode node with a parseable date', () => {
    const art = makeArtifact([
      { type: 'Episode', properties: { publish_date: 'bad' } },
      { type: 'Episode', properties: { publish_date: '2024-05-05T00:00:00Z' } },
    ])
    expect(calendarPublishYmdFromParsedArtifact(art, 0)).toBe(
      formatLocalYmd(new Date(Date.parse('2024-05-05T00:00:00Z'))),
    )
  })
})

describe('selectParsedArtifactsForGraphLoad', () => {
  const makeCandidate = (
    name: string,
    publishYmd: string,
    fileLastModifiedMs = Date.parse('2024-09-01T12:00:00'),
  ) => ({
    art: {
      name,
      kind: 'gi' as const,
      episodeId: null,
      nodes: 0,
      edges: 0,
      nodeTypes: {},
      data: {},
    } as ParsedArtifact,
    relKey: name,
    publishYmd,
    fileLastModifiedMs,
  })

  it('keeps all artifacts for two episodes under an all-time lens with a high cap', () => {
    const r = selectParsedArtifactsForGraphLoad(
      [
        makeCandidate('m/a.gi.json', '2024-01-10'),
        makeCandidate('m/a.kg.json', '2024-01-10'),
        makeCandidate('m/b.gi.json', '2024-06-20'),
      ],
      '',
      10,
    )
    expect(r.episodeCount).toBe(2)
    expect(r.wasCapped).toBe(false)
    expect(r.kept.length).toBe(3)
  })

  it('caps episodes recency-first under an all-time lens', () => {
    const r = selectParsedArtifactsForGraphLoad(
      [
        makeCandidate('m/old.gi.json', '2024-01-10'),
        makeCandidate('m/new.gi.json', '2024-06-20'),
      ],
      '',
      1,
    )
    expect(r.episodeCount).toBe(1)
    expect(r.wasCapped).toBe(true)
    expect(r.kept.map((a) => a.name)).toEqual(['m/new.gi.json'])
  })

  it('filters by sinceYmd (dated lens)', () => {
    const r = selectParsedArtifactsForGraphLoad(
      [
        makeCandidate('m/old.gi.json', '2024-01-10'),
        makeCandidate('m/new.gi.json', '2024-06-20'),
      ],
      '2024-06-01',
      10,
    )
    expect(r.episodeCount).toBe(1)
    expect(r.kept.map((a) => a.name)).toEqual(['m/new.gi.json'])
  })

  it('returns the empty result when the dated window filters everything out', () => {
    const r = selectParsedArtifactsForGraphLoad(
      [makeCandidate('m/a.gi.json', '2024-01-10')],
      '2030-01-01',
      10,
    )
    expect(r).toEqual({ kept: [], wasCapped: false, episodeCount: 0 })
  })

  it('returns the empty result for no candidates', () => {
    expect(selectParsedArtifactsForGraphLoad([], '', 10)).toEqual({
      kept: [],
      wasCapped: false,
      episodeCount: 0,
    })
  })

  it('clamps a non-positive cap up to 1', () => {
    const r = selectParsedArtifactsForGraphLoad(
      [
        makeCandidate('m/a.gi.json', '2024-01-10'),
        makeCandidate('m/b.gi.json', '2024-06-20'),
      ],
      '',
      0,
    )
    expect(r.episodeCount).toBe(1)
    expect(r.wasCapped).toBe(true)
  })

  it('derives a publish date from file mtime when the candidate has none', () => {
    const mtime = Date.parse('2024-06-20T12:00:00')
    const datedMtime = Date.parse('2024-01-10T12:00:00')
    const r = selectParsedArtifactsForGraphLoad(
      [
        // undated -> mtime in June survives the June since-filter
        makeCandidate('m/from-mtime.gi.json', '', mtime),
        makeCandidate('m/too-old.gi.json', '', datedMtime),
      ],
      '2024-06-01',
      10,
    )
    expect(r.episodeCount).toBe(1)
    expect(r.kept.map((a) => a.name)).toEqual(['m/from-mtime.gi.json'])
  })

  it('groups candidates by stem and tracks the newest mtime + latest valid date', () => {
    const r = selectParsedArtifactsForGraphLoad(
      [
        makeCandidate('m/a.kg.json', '2024-01-10', Date.parse('2024-01-10T12:00:00')),
        makeCandidate('m/a.gi.json', '2024-06-20', Date.parse('2024-08-01T12:00:00')),
      ],
      '2024-06-01',
      10,
    )
    // both kinds share stem 'm/a'; newest valid date (June) survives the filter
    expect(r.episodeCount).toBe(1)
    expect(r.kept.map((a) => a.name).sort()).toEqual(['m/a.gi.json', 'm/a.kg.json'])
  })

  it('falls back to the raw name as the stem when it has no known suffix', () => {
    const r = selectParsedArtifactsForGraphLoad([makeCandidate('m/loose', '2024-06-20')], '', 10)
    expect(r.episodeCount).toBe(1)
    expect(r.kept.map((a) => a.name)).toEqual(['m/loose'])
  })

  it('applies the topic-cluster bonus to outrank a slightly newer non-cluster episode', () => {
    const doc: TopicClustersDocument = {
      clusters: [{ members: [{ topic_id: 't', episode_ids: ['x'] }] }],
    }
    const r = selectParsedArtifactsForGraphLoad(
      [
        makeCandidate('pods/z.gi.json', '2024-06-30'),
        makeCandidate('pods/y.gi.json', '2024-06-15'),
        makeCandidate('pods/x.gi.json', '2024-06-10'),
      ],
      '',
      2,
      doc,
    )
    const names = r.kept.map((a) => a.name)
    expect(names).toContain('pods/z.gi.json')
    expect(names).toContain('pods/x.gi.json')
    expect(names).not.toContain('pods/y.gi.json')
  })

  it('tie-breaks equal-score episodes by newer publish date', () => {
    // a fresh anchor sets poolMax; both older episodes sit beyond the decay
    // window so they share the recency floor (equal score) but differ by date.
    const r = selectParsedArtifactsForGraphLoad(
      [
        makeCandidate('m/anchor.gi.json', '2024-06-20'),
        makeCandidate('m/older.gi.json', '2020-01-01'),
        makeCandidate('m/newer-of-old.gi.json', '2021-01-01'),
      ],
      '',
      2,
    )
    expect(r.episodeCount).toBe(2)
    const names = r.kept.map((a) => a.name)
    expect(names).toContain('m/anchor.gi.json')
    // between the two floor-scored episodes, the newer date wins the second slot
    expect(names).toContain('m/newer-of-old.gi.json')
    expect(names).not.toContain('m/older.gi.json')
  })

  it('tie-breaks undated all-time episodes by stem ascending', () => {
    const mtime = Date.parse('2024-06-20T12:00:00')
    const r = selectParsedArtifactsForGraphLoad(
      [
        makeCandidate('m/zeta.gi.json', '', mtime),
        makeCandidate('m/alpha.gi.json', '', mtime),
      ],
      '',
      1,
    )
    expect(r.episodeCount).toBe(1)
    expect(r.kept.map((a) => a.name)).toEqual(['m/alpha.gi.json'])
  })
})
