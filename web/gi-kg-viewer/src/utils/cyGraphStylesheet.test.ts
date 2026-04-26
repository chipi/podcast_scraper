import { describe, expect, it } from 'vitest'
import {
  buildGiKgCyStylesheet,
  estimateLabelHalfWidthPx,
  sideLabelTextMarginX,
  topicDegreeHeatBorderWidthPx,
} from './cyGraphStylesheet'

function nodeRuleStyle(sheet: ReturnType<typeof buildGiKgCyStylesheet>): Record<string, unknown> {
  return (sheet.find((r) => (r as { selector?: string }).selector === 'node') as { style: Record<string, unknown> })
    .style
}

describe('topicDegreeHeatBorderWidthPx', () => {
  it('returns 0 when heat is missing, zero, or non-finite', () => {
    expect(topicDegreeHeatBorderWidthPx(undefined, false)).toBe(0)
    expect(topicDegreeHeatBorderWidthPx(0, false)).toBe(0)
    expect(topicDegreeHeatBorderWidthPx(Number.NaN, false)).toBe(0)
  })

  it('ramps from 1px once heat is positive (main profile)', () => {
    expect(topicDegreeHeatBorderWidthPx(0.25, false)).toBeCloseTo(1.75, 5)
    expect(topicDegreeHeatBorderWidthPx(1, false)).toBe(4)
  })

  it('uses compact scaling with a 1px floor when heat is positive', () => {
    expect(topicDegreeHeatBorderWidthPx(0.01, true)).toBeGreaterThanOrEqual(1)
    expect(topicDegreeHeatBorderWidthPx(0, true)).toBe(0)
  })
})

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

  it('search-hit adds ring only (tier sizes stay literal; margin comes from callback + width)', () => {
    const hitRule = buildGiKgCyStylesheet({ includeSearchHit: true, compact: false }).find(
      (r) => (r as { selector?: string }).selector === 'node.search-hit',
    ) as { style: Record<string, unknown> }
    expect(hitRule.style['text-margin-x']).toBeUndefined()
    expect(hitRule.style.width).toBeUndefined()
    expect(hitRule.style['border-color']).toBe('#fab005')
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
    expect(rule.style['background-opacity']).toBe(0.06)
    expect(rule.style['border-style']).toBe('dashed')
    expect(rule.style['background-color']).toBe('var(--ps-kg)')
  })

  it('includes cross-episode expandable and expanded-seed ring rules (full graph)', () => {
    const sheet = buildGiKgCyStylesheet({ compact: false })
    const expandable = sheet.find(
      (r) => (r as { selector?: string }).selector === 'node.graph-expand-eligible',
    ) as { style: Record<string, unknown> }
    const seed = sheet.find(
      (r) => (r as { selector?: string }).selector === 'node.graph-expand-seed',
    ) as { style: Record<string, unknown> }
    expect(expandable.style['border-color']).toBe('#14b8a6')
    expect(expandable.style['border-width']).toBe(2)
    expect(seed.style['border-color']).toBe('#748ffc')
    expect(seed.style['border-width']).toBe(3)
  })

  it('includes cross-episode expand ring rules with thinner borders for compact/minimap', () => {
    const sheet = buildGiKgCyStylesheet({ compact: true })
    const expandable = sheet.find(
      (r) => (r as { selector?: string }).selector === 'node.graph-expand-eligible',
    ) as { style: Record<string, unknown> }
    const seed = sheet.find(
      (r) => (r as { selector?: string }).selector === 'node.graph-expand-seed',
    ) as { style: Record<string, unknown> }
    expect(expandable.style['border-width']).toBe(1.5)
    expect(seed.style['border-width']).toBe(2.25)
  })

  it('includes explicit edge fallback for (unknown) edgeType', () => {
    const sheet = buildGiKgCyStylesheet({ compact: false })
    const rule = sheet.find(
      (r) => (r as { selector?: string }).selector === 'edge[edgeType = "(unknown)"]',
    ) as { style: Record<string, unknown> }
    expect(rule).toBeTruthy()
    expect(rule.style['line-color']).toBe('var(--ps-muted)')
  })

  // RFC-080 V2 — Insight grounding selector lives in the stylesheet
  // unconditionally; only fires when the class is assigned at element-
  // build time (toCytoElements). Search-hit border specificity is
  // covered by source order — `.search-hit` is appended last in the
  // sheet so its border-color/width override the dashed warning when
  // both apply.
  it('V2: includes insight-ungrounded dashed warning border', () => {
    const sheet = buildGiKgCyStylesheet({ compact: false })
    const rule = sheet.find(
      (r) =>
        (r as { selector?: string }).selector ===
        'node[type = "Insight"].insight-ungrounded',
    ) as { style: Record<string, unknown> }
    expect(rule).toBeTruthy()
    expect(rule.style['border-style']).toBe('dashed')
    expect(rule.style['border-width']).toBe(1.5)
    expect(rule.style['border-opacity']).toBe(0.7)
  })

  it('V2: search-hit selector is appended after insight-ungrounded so its border wins', () => {
    const sheet = buildGiKgCyStylesheet({ compact: false, includeSearchHit: true })
    const ungroundedIdx = sheet.findIndex(
      (r) =>
        (r as { selector?: string }).selector ===
        'node[type = "Insight"].insight-ungrounded',
    )
    const searchHitIdx = sheet.findIndex(
      (r) => (r as { selector?: string }).selector === 'node.search-hit',
    )
    expect(ungroundedIdx).toBeGreaterThanOrEqual(0)
    expect(searchHitIdx).toBeGreaterThan(ungroundedIdx)
  })

  // RFC-080 V5 — node-size-by-degree is opt-in via
  // enableNodeSizeByDegree. Default keeps fixed widths so the lens can
  // be staged on real corpora before promoting to default-on (per RFC
  // rollout).
  it('V5: width/height stay fixed by default for Topic + Episode', () => {
    const sheet = buildGiKgCyStylesheet({ compact: false })
    const topic = sheet.find(
      (r) => (r as { selector?: string }).selector === 'node[type = "Topic"]',
    ) as { style: Record<string, unknown> }
    const episode = sheet.find(
      (r) => (r as { selector?: string }).selector === 'node[type = "Episode"]',
    ) as { style: Record<string, unknown> }
    expect(typeof topic.style.width).toBe('number')
    expect(typeof episode.style.width).toBe('number')
  })

  it('V5: enableNodeSizeByDegree maps Topic + Episode width/height through degreeHeat', () => {
    const sheet = buildGiKgCyStylesheet({ compact: false, enableNodeSizeByDegree: true })
    const topic = sheet.find(
      (r) => (r as { selector?: string }).selector === 'node[type = "Topic"]',
    ) as { style: Record<string, unknown> }
    const episode = sheet.find(
      (r) => (r as { selector?: string }).selector === 'node[type = "Episode"]',
    ) as { style: Record<string, unknown> }
    expect(String(topic.style.width)).toMatch(/^mapData\(degreeHeat, 0, 1, \d+, \d+\)$/)
    expect(String(topic.style.height)).toMatch(/^mapData\(degreeHeat, 0, 1, \d+, \d+\)$/)
    expect(String(episode.style.width)).toMatch(/^mapData\(degreeHeat, 0, 1, \d+, \d+\)$/)
  })

  it('V5: other types (Quote, Speaker, …) keep fixed sizes even when V5 is on', () => {
    const sheet = buildGiKgCyStylesheet({ compact: false, enableNodeSizeByDegree: true })
    const quote = sheet.find(
      (r) => (r as { selector?: string }).selector === 'node[type = "Quote"]',
    ) as { style: Record<string, unknown> }
    expect(typeof quote.style.width).toBe('number')
  })

  // RFC-080 V1 — aggregated-edge selectors are always present in the
  // sheet but only fire when toCytoElements emits an edge with the
  // matching class. Width is `mapData(weight, ...)` so the chunkiest
  // edges read clearly without overwhelming the canvas. Disjoint from
  // the per-Insight ABOUT selector (which uses edgeType, not class).
  it('V1: includes graph-edge-about-agg width mapping by data(weight)', () => {
    const sheet = buildGiKgCyStylesheet({ compact: false })
    const rule = sheet.find(
      (r) => (r as { selector?: string }).selector === 'edge.graph-edge-about-agg',
    ) as { style: Record<string, unknown> }
    expect(rule).toBeTruthy()
    expect(String(rule.style.width)).toMatch(/^mapData\(weight, 1, \d+, [\d.]+, [\d.]+\)$/)
    expect(rule.style['line-color']).toBe('var(--ps-gi)')
  })

  it('V1: includes graph-edge-spoke-in-agg width mapping by data(weight)', () => {
    const sheet = buildGiKgCyStylesheet({ compact: false })
    const rule = sheet.find(
      (r) => (r as { selector?: string }).selector === 'edge.graph-edge-spoke-in-agg',
    ) as { style: Record<string, unknown> }
    expect(rule).toBeTruthy()
    expect(String(rule.style.width)).toMatch(/^mapData\(weight, 1, \d+, [\d.]+, [\d.]+\)$/)
    expect(rule.style['line-color']).toBe('var(--ps-primary)')
    expect(rule.style['target-arrow-shape']).toBe('triangle')
  })
})
