import { describe, expect, it } from 'vitest'
import { segmentsForSubstringNeedle } from './exploreHighlight'

describe('segmentsForSubstringNeedle', () => {
  it('returns one segment when needle is empty', () => {
    expect(segmentsForSubstringNeedle('Hello world', '', 100)).toEqual([
      { text: 'Hello world', mark: false },
    ])
  })

  it('highlights case-insensitive matches', () => {
    expect(segmentsForSubstringNeedle('aa Climate bb climate cc', 'climate', 200)).toEqual([
      { text: 'aa ', mark: false },
      { text: 'Climate', mark: true },
      { text: ' bb ', mark: false },
      { text: 'climate', mark: true },
      { text: ' cc', mark: false },
    ])
  })

  it('truncates before splitting', () => {
    const long = 'x'.repeat(100)
    const out = segmentsForSubstringNeedle(long, 'xx', 10)
    const joined = out.map((s) => s.text).join('')
    expect(joined.endsWith('…')).toBe(true)
    expect(joined.length).toBeLessThanOrEqual(11)
    expect(out.some((s) => s.mark)).toBe(true)
  })
})
