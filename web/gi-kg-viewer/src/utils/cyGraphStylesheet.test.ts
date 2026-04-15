import { describe, expect, it } from 'vitest'
import {
  buildGiKgCyStylesheet,
  estimateLabelHalfWidthPx,
  sideLabelTextMarginX,
} from './cyGraphStylesheet'

function nodeRuleStyle(sheet: ReturnType<typeof buildGiKgCyStylesheet>): Record<string, unknown> {
  return (sheet.find((r) => (r as { selector?: string }).selector === 'node') as { style: Record<string, unknown> })
    .style
}

describe('estimateLabelHalfWidthPx', () => {
  it('grows with length up to half max wrap', () => {
    expect(estimateLabelHalfWidthPx('a', 140, 5.5)).toBe(10)
    const long = 'x'.repeat(200)
    expect(estimateLabelHalfWidthPx(long, 140, 5.5)).toBe(70)
  })

  it('treats multi-line-ish labels as full wrap width', () => {
    const wrapped = 'x'.repeat(120)
    expect(estimateLabelHalfWidthPx(wrapped, 140, 5.5)).toBe(70)
  })
})

describe('sideLabelTextMarginX', () => {
  it('is zero for empty label', () => {
    expect(sideLabelTextMarginX(18, '  ', 140, 3, 5.5)).toBe(0)
  })

  it('increases with label length then caps', () => {
    const short = sideLabelTextMarginX(18, 'hi', 140, 3, 5.5)
    const long = sideLabelTextMarginX(18, 'x'.repeat(200), 140, 3, 5.5)
    expect(long).toBeGreaterThan(short)
  })

  it('compact graph uses smaller body so margin can be smaller for same text', () => {
    const label = 'episode title here'
    const full = sideLabelTextMarginX(18, label, 140, 3, 5.5)
    const mini = sideLabelTextMarginX(14, label, 72, 2, 4.75)
    expect(mini).toBeLessThan(full)
  })
})

describe('buildGiKgCyStylesheet', () => {
  it('default side: center halign, static margin-x 0 (dynamic callback supplies offset)', () => {
    const st = nodeRuleStyle(buildGiKgCyStylesheet({ compact: false }))
    expect(st['text-valign']).toBe('center')
    expect(st['text-halign']).toBe('center')
    expect(st['text-margin-y']).toBe(0)
    expect(st['text-margin-x']).toBe(0)
    expect(st['text-background-shape']).toBe('roundrectangle')
    expect(Number(st['text-outline-width'])).toBeGreaterThan(0)
  })

  it('search-hit widens body but does not pin text-margin-x (callback uses ele.width())', () => {
    const hitRule = buildGiKgCyStylesheet({ includeSearchHit: true, compact: false }).find(
      (r) => (r as { selector?: string }).selector === 'node.search-hit',
    ) as { style: Record<string, unknown> }
    expect(hitRule.style['text-margin-x']).toBeUndefined()
    expect(hitRule.style.width).toBe(24)
  })

  it('above placement: top center with negative margin-y', () => {
    const st = nodeRuleStyle(buildGiKgCyStylesheet({ nodeLabelPlacement: 'above', compact: false }))
    expect(st['text-valign']).toBe('top')
    expect(st['text-halign']).toBe('center')
    expect(st['text-margin-x']).toBe(0)
    expect(Number(st['text-margin-y'])).toBeLessThan(0)
  })

  it('below placement: bottom center with positive margin-y', () => {
    const st = nodeRuleStyle(buildGiKgCyStylesheet({ nodeLabelPlacement: 'below', compact: false }))
    expect(st['text-valign']).toBe('bottom')
    expect(st['text-halign']).toBe('center')
    expect(st['text-margin-x']).toBe(0)
    expect(Number(st['text-margin-y'])).toBeGreaterThan(0)
  })

  it('includes TopicCluster compound styling rule', () => {
    const sheet = buildGiKgCyStylesheet({ compact: false })
    const rule = sheet.find(
      (r) => (r as { selector?: string }).selector === 'node[type = "TopicCluster"]',
    ) as { style: Record<string, unknown> }
    expect(rule).toBeTruthy()
    expect(rule.style['background-opacity']).toBeDefined()
    expect(rule.style['border-style']).toBe('dashed')
  })
})
