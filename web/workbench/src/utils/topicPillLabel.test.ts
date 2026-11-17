import { describe, expect, it } from 'vitest'

import { renderPillLabel } from './topicPillLabel'

describe('renderPillLabel', () => {
  describe('ellipsis strategy (legacy default)', () => {
    it('returns short label verbatim', () => {
      expect(renderPillLabel('oil prices', 24, 'ellipsis')).toBe('oil prices')
    })

    it('returns label at exact cap verbatim', () => {
      const label = 'a'.repeat(24)
      expect(renderPillLabel(label, 24, 'ellipsis')).toBe(label)
    })

    it('truncates with ellipsis past the cap', () => {
      const out = renderPillLabel('a'.repeat(40), 24, 'ellipsis')
      // 23 chars of payload + 1-char ellipsis = 24-char display
      expect(out).toHaveLength(24)
      expect(out.endsWith('…')).toBe(true)
    })

    it('trims leading/trailing whitespace before measuring', () => {
      // Pre-#656 foundation: #653 labels sometimes arrive with trailing
      // whitespace from the canonical slug joiner — don't let that
      // bloat past the cap.
      const out = renderPillLabel('   oil prices   ', 24, 'ellipsis')
      expect(out).toBe('oil prices')
    })
  })

  describe('wrap strategy', () => {
    it('returns full label regardless of length', () => {
      const long = 'disruptions in the strait of hormuz are causing an energy crisis'
      expect(renderPillLabel(long, 24, 'wrap')).toBe(long)
    })

    it('still trims whitespace', () => {
      expect(renderPillLabel('  ai spending  ', 24, 'wrap')).toBe('ai spending')
    })
  })

  describe('none strategy', () => {
    it('returns full label regardless of length', () => {
      const long = 'a'.repeat(100)
      expect(renderPillLabel(long, 5, 'none')).toBe(long)
    })
  })

  describe('#653-era short labels (pre-#656 foundation target)', () => {
    it.each([
      ['oil prices', 10],
      ['ai spending', 11],
      ['shadow fleet', 12],
      ['naval blockade', 14],
      ['sanctions evasion', 17],
    ])('renders %p (%d chars) without truncation under default cap', (label, expectedLength) => {
      const out = renderPillLabel(label, 24, 'ellipsis')
      expect(out).toBe(label)
      expect(out).toHaveLength(expectedLength)
      expect(out.endsWith('…')).toBe(false)
    })
  })
})
