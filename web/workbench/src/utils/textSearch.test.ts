import { describe, expect, it } from 'vitest'
import { textSearchSegments } from './textSearch'

describe('textSearchSegments', () => {
  it('returns the whole text as one non-match segment for an empty query', () => {
    expect(textSearchSegments('hello world', '')).toEqual({
      segments: [{ text: 'hello world', match: false }],
      matchCount: 0,
    })
  })

  it('splits around case-insensitive matches with running matchIndex', () => {
    const { segments, matchCount } = textSearchSegments('Error: error ERROR', 'error')
    expect(matchCount).toBe(3)
    expect(segments.filter((s) => s.match).map((s) => s.matchIndex)).toEqual([0, 1, 2])
    // Reconstructing the segments reproduces the original text.
    expect(segments.map((s) => s.text).join('')).toBe('Error: error ERROR')
  })

  it('handles no matches', () => {
    const { segments, matchCount } = textSearchSegments('nothing here', 'zzz')
    expect(matchCount).toBe(0)
    expect(segments).toEqual([{ text: 'nothing here', match: false }])
  })

  it('handles a match at the very start and end', () => {
    const { segments, matchCount } = textSearchSegments('aXa', 'a')
    expect(matchCount).toBe(2)
    expect(segments.map((s) => s.text).join('')).toBe('aXa')
  })
})
