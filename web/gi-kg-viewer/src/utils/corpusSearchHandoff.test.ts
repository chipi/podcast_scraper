import { describe, expect, it } from 'vitest'
import { buildLibrarySearchHandoffQuery } from './corpusSearchHandoff'

describe('buildLibrarySearchHandoffQuery', () => {
  it('prefers summary_text when set', () => {
    const q = buildLibrarySearchHandoffQuery({
      summary_text: '  full prose summary  ',
      summary_title: 'T',
      summary_bullets: ['alpha'],
      episode_title: 'ignored',
    })
    expect(q).toContain('full prose summary')
    expect(q).not.toContain('T')
    expect(q).not.toContain('alpha')
  })

  it('prefers summary title and bullets over episode title', () => {
    const q = buildLibrarySearchHandoffQuery({
      summary_title: 'T',
      summary_bullets: ['alpha', 'bravo'],
      episode_title: 'ignored',
    })
    expect(q).toContain('T')
    expect(q).toContain('alpha')
    expect(q).not.toContain('ignored')
  })

  it('falls back to episode title when no summary', () => {
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
})
