import { describe, expect, it } from 'vitest'
import { groupEpisodesByYear } from './yearGrouping'

interface G {
  id: string
  date: string | null
}

const getDate = (g: G) => g.date

describe('groupEpisodesByYear', () => {
  it('buckets groups by year and sorts years descending', () => {
    const sections = groupEpisodesByYear<G>(
      [
        { id: 'a', date: '2024-01-15' },
        { id: 'b', date: '2023-06-01' },
        { id: 'c', date: '2024-12-05' },
        { id: 'd', date: '2022-07-20' },
      ],
      getDate,
    )
    expect(sections.map((s) => s.year)).toEqual([2024, 2023, 2022])
    expect(sections[0].groups.map((g) => g.id)).toEqual(['a', 'c'])
    expect(sections[1].groups.map((g) => g.id)).toEqual(['b'])
    expect(sections[2].groups.map((g) => g.id)).toEqual(['d'])
  })

  it('preserves within-year order (already score-sorted upstream)', () => {
    const sections = groupEpisodesByYear<G>(
      [
        { id: '3', date: '2024-03-05' },
        { id: '2', date: '2024-06-05' },
        { id: '1', date: '2024-11-05' },
      ],
      getDate,
    )
    expect(sections).toHaveLength(1)
    expect(sections[0].groups.map((g) => g.id)).toEqual(['3', '2', '1'])
  })

  it('places rows with a null / unparseable date in an "unknown" bucket at the bottom', () => {
    const sections = groupEpisodesByYear<G>(
      [
        { id: 'a', date: '2024-03-01' },
        { id: 'b', date: null },
        { id: 'c', date: 'not-a-date' },
      ],
      getDate,
    )
    expect(sections.map((s) => s.year)).toEqual([2024, 'unknown'])
    expect(sections[1].groups.map((g) => g.id)).toEqual(['b', 'c'])
  })

  it('returns an empty array for an empty input', () => {
    expect(groupEpisodesByYear<G>([], getDate)).toEqual([])
  })

  it('handles a single-year corpus without adding a fake bucket', () => {
    const sections = groupEpisodesByYear<G>(
      [
        { id: 'a', date: '2024-03-01' },
        { id: 'b', date: '2024-04-01' },
      ],
      getDate,
    )
    expect(sections).toHaveLength(1)
    expect(sections[0].year).toBe(2024)
  })
})
