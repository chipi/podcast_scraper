import { describe, expect, it } from 'vitest'
import { formatInsightPositionHintLine } from './insightPositionHint'

describe('formatInsightPositionHintLine', () => {
  it('uses tiers when no duration', () => {
    expect(formatInsightPositionHintLine(0.1)).toBe(
      'Position in episode: early segment',
    )
    expect(formatInsightPositionHintLine(0.5)).toBe(
      'Position in episode: middle segment',
    )
    expect(formatInsightPositionHintLine(0.9)).toBe(
      'Position in episode: late segment',
    )
  })

  it('clamps hint to 0–1', () => {
    expect(formatInsightPositionHintLine(-1)).toBe(
      'Position in episode: early segment',
    )
    expect(formatInsightPositionHintLine(2)).toBe(
      'Position in episode: late segment',
    )
  })

  it('formats time when duration is known', () => {
    const ms = 10 * 60 * 1000 // 10 minutes
    expect(formatInsightPositionHintLine(0.5, ms)).toBe(
      'Position in episode: about 5m into the episode',
    )
    expect(formatInsightPositionHintLine(0.05, ms)).toBe(
      'Position in episode: about 30s into the episode',
    )
  })
})
