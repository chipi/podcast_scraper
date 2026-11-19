import { describe, expect, it } from 'vitest'
import { buildLibrarySearchHandoffQuery } from './corpusSearchHandoff'

describe('buildLibrarySearchHandoffQuery', () => {
  it('ignores prose summary_text; uses title and bullets like build_similarity_query', () => {
    const q = buildLibrarySearchHandoffQuery({
      summary_title: 'T',
      summary_bullets: ['alpha'],
      episode_title: 'ignored',
    })
    expect(q).toContain('T')
    expect(q).toContain('alpha')
    expect(q).not.toContain('ignored')
  })

  it('joins title and multiple bullets', () => {
    const q = buildLibrarySearchHandoffQuery({
      summary_title: 'T',
      summary_bullets: ['alpha', 'bravo'],
      episode_title: 'ignored',
    })
    expect(q).toContain('T')
    expect(q).toContain('alpha')
    expect(q).toContain('bravo')
    expect(q).not.toContain('ignored')
  })

  it('falls back to episode title when no title or bullets', () => {
    expect(
      buildLibrarySearchHandoffQuery({
        summary_title: null,
        summary_bullets: [],
        episode_title: '  only  ',
      }),
    ).toBe('only')
  })

  it('truncates at word boundary when over maxChars', () => {
    const long = 'word '.repeat(500)
    const q = buildLibrarySearchHandoffQuery(
      {
        summary_title: null,
        summary_bullets: [long],
        episode_title: 'x',
      },
      40,
    )
    expect(q.length).toBeLessThanOrEqual(40)
    expect(q.endsWith(' ')).toBe(false)
  })

  it('clips a paragraph stuffed into summary_title', () => {
    const prose = 'A New Experiment in Private Equity ' + 'word '.repeat(80)
    const q = buildLibrarySearchHandoffQuery({
      summary_title: prose,
      summary_bullets: [],
      episode_title: 'Ep',
    })
    expect(q.length).toBeLessThanOrEqual(480)
    expect(q.startsWith('A New')).toBe(true)
    expect(q.length).toBeLessThanOrEqual(145)
  })

  it('clips huge bullets and caps total length by default', () => {
    const bullet = 'B'.repeat(2000)
    const q = buildLibrarySearchHandoffQuery({
      summary_title: 'Short title',
      summary_bullets: [bullet],
      episode_title: 'Ep',
    })
    expect(q.length).toBeLessThanOrEqual(480)
    expect(q).toContain('Short title')
    expect(q.length).toBeLessThan(400)
  })
})
