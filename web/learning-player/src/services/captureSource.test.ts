import { afterEach, describe, expect, it } from 'vitest'
import { captureSource, setLearnOrigin, setShortListenProbe } from './captureSource'

describe('captureSource (operator 2026-10-12)', () => {
  afterEach(() => {
    setShortListenProbe(() => false)
    setLearnOrigin(null)
  })

  it('is an episode by default', () => {
    expect(captureSource('ep1')).toBe('episode')
  })

  it('is a learning day only on the episode opened from one', () => {
    setLearnOrigin('ep1')
    expect(captureSource('ep1')).toBe('learn')
    expect(captureSource('ep2')).toBe('episode')
  })

  it('is a short listen while one plays, whatever page opened the episode', () => {
    setLearnOrigin('ep1')
    setShortListenProbe(() => true)
    expect(captureSource('ep1')).toBe('moment')
  })
})
