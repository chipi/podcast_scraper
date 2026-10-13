import { describe, expect, it } from 'vitest'
import { reelMinutes } from './reelLength'

describe('reelMinutes', () => {
  it('rounds to whole minutes, at least one, at most the budget', () => {
    expect(reelMinutes(144)).toBe(2) // 2:24
    expect(reelMinutes(264)).toBe(4) // 4:24
    expect(reelMinutes(20)).toBe(1)
    expect(reelMinutes(0)).toBe(1)
    expect(reelMinutes(17 * 60)).toBe(15)
    expect(reelMinutes(17 * 60, 30)).toBe(17)
  })
})
