// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'
import {
  THEME_REGION_PALETTE,
  THEME_REGION_PALETTE_SIZE,
  themeRegionColor,
  storylineRegionIndex,
} from './themeRegionPalette'

describe('themeRegionPalette', () => {
  it('palette has THEME_REGION_PALETTE_SIZE entries and matches constant', () => {
    expect(THEME_REGION_PALETTE.length).toBe(THEME_REGION_PALETTE_SIZE)
    expect(THEME_REGION_PALETTE_SIZE).toBe(8)
  })

  it('storylineRegionIndex is deterministic — same id → same slot across calls', () => {
    expect(storylineRegionIndex('thc:interest-rates')).toBe(storylineRegionIndex('thc:interest-rates'))
    expect(storylineRegionIndex('thc:ai-agents')).toBe(storylineRegionIndex('thc:ai-agents'))
  })

  it('storylineRegionIndex returns an integer in [0, PALETTE_SIZE)', () => {
    const ids = ['thc:a', 'thc:b', 'thc:c', 'thc:interest-rates', 'thc:ai-agents']
    for (const id of ids) {
      const idx = storylineRegionIndex(id)
      expect(Number.isInteger(idx)).toBe(true)
      expect(idx).toBeGreaterThanOrEqual(0)
      expect(idx).toBeLessThan(THEME_REGION_PALETTE_SIZE)
    }
  })

  it('themeRegionColor resolves to a palette hex', () => {
    for (const id of ['thc:x', 'thc:y', 'thc:z']) {
      expect(THEME_REGION_PALETTE).toContain(themeRegionColor(id))
    }
  })

  it('themeRegionColor falls back gracefully on empty id', () => {
    expect(themeRegionColor('')).toBe(THEME_REGION_PALETTE[0])
  })
})
