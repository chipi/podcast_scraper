import { describe, expect, it } from 'vitest'
import { normalizeFeedIdForViewer } from './feedId'

describe('normalizeFeedIdForViewer', () => {
  it('trims whitespace', () => {
    expect(normalizeFeedIdForViewer('  abc  ')).toBe('abc')
  })

  it('handles null and undefined', () => {
    expect(normalizeFeedIdForViewer(null)).toBe('')
    expect(normalizeFeedIdForViewer(undefined)).toBe('')
  })
})
