import { describe, expect, it } from 'vitest'
import { StaleGeneration } from './staleGeneration'

describe('StaleGeneration', () => {
  it('bump returns monotonic sequences and isCurrent matches latest', () => {
    const g = new StaleGeneration()
    const a = g.bump()
    const b = g.bump()
    expect(a).toBeLessThan(b)
    expect(g.isCurrent(b)).toBe(true)
    expect(g.isCurrent(a)).toBe(false)
    expect(g.isStale(a)).toBe(true)
  })

  it('invalidate advances without returning a seq', () => {
    const g = new StaleGeneration()
    const seq = g.bump()
    g.invalidate()
    expect(g.isStale(seq)).toBe(true)
  })
})
