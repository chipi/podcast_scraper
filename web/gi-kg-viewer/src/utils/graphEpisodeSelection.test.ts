import { describe, expect, it } from 'vitest'
import {
  GRAPH_DEFAULT_EPISODE_CAP,
  selectRelPathsForGraphLoad,
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

  it('caps at N newest for all-time lens', () => {
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

  it('uses default cap constant', () => {
    expect(GRAPH_DEFAULT_EPISODE_CAP).toBe(15)
  })
})
