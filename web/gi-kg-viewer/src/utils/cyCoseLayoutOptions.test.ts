import { describe, expect, it } from 'vitest'
import type { EdgeSingular } from 'cytoscape'
import {
  giKgCoseEdgeElasticity,
  giKgCoseIdealEdgeLength,
  giKgCoseLayoutOptionsCompact,
  giKgCoseLayoutOptionsMain,
  giKgCoseLayoutOptionsMainFallback,
  giKgCoseNodeRepulsionFromData,
  isIntraTopicClusterEdgeParents,
  isTopicClusterParentId,
} from './cyCoseLayoutOptions'

describe('isTopicClusterParentId', () => {
  it('is true for tc: ids', () => {
    expect(isTopicClusterParentId('tc:foo')).toBe(true)
    expect(isTopicClusterParentId('  tc:x  ')).toBe(true)
  })

  it('is false for other parents', () => {
    expect(isTopicClusterParentId('')).toBe(false)
    expect(isTopicClusterParentId('g:topic')).toBe(false)
    expect(isTopicClusterParentId(undefined)).toBe(false)
  })
})

describe('isIntraTopicClusterEdgeParents', () => {
  it('is true when both parents match the same tc:', () => {
    expect(isIntraTopicClusterEdgeParents('tc:a', 'tc:a')).toBe(true)
  })

  it('is false when parents differ or are not tc:', () => {
    expect(isIntraTopicClusterEdgeParents('tc:a', 'tc:b')).toBe(false)
    expect(isIntraTopicClusterEdgeParents(undefined, 'tc:a')).toBe(false)
    expect(isIntraTopicClusterEdgeParents('g:x', 'g:x')).toBe(false)
  })
})

describe('giKgCoseNodeRepulsionFromData', () => {
  it('uses higher repulsion for TopicCluster compounds', () => {
    const main = giKgCoseNodeRepulsionFromData('TopicCluster', undefined, 'main')
    const base = giKgCoseNodeRepulsionFromData('Topic', undefined, 'main')
    expect(main).toBeGreaterThan(base)
  })

  it('uses lower repulsion for members under tc:', () => {
    const member = giKgCoseNodeRepulsionFromData('Topic', 'tc:cl', 'main')
    const base = giKgCoseNodeRepulsionFromData('Topic', undefined, 'main')
    expect(member).toBeLessThan(base)
  })
})

describe('giKgCoseLayoutOptionsMain', () => {
  it('includes nestingFactor for cross-graph edges', () => {
    const o = giKgCoseLayoutOptionsMain()
    expect(o.name).toBe('cose')
    expect(o.nestingFactor).toBe(1.52)
    expect(o.numIter).toBe(2500)
    expect(typeof o.nodeRepulsion).toBe('function')
    expect(typeof o.idealEdgeLength).toBe('function')
    expect(typeof o.edgeElasticity).toBe('function')
  })
})

function mockEdge(
  sourceParent: string | null,
  targetParent: string | null,
  edgeType: string,
): EdgeSingular {
  return {
    data: (k: string) => (k === 'edgeType' ? edgeType : undefined),
    source: () => ({ data: (k: string) => (k === 'parent' ? sourceParent : undefined) }),
    target: () => ({ data: (k: string) => (k === 'parent' ? targetParent : undefined) }),
  } as unknown as EdgeSingular
}

describe('giKgCoseIdealEdgeLength', () => {
  it('prioritises intra-topic-cluster length over semantic ABOUT', () => {
    const edge = mockEdge('tc:x', 'tc:x', 'ABOUT')
    expect(giKgCoseIdealEdgeLength(edge, 'main')).toBe(36)
  })

  it('uses semantic ABOUT length outside tc: clusters', () => {
    const edge = mockEdge(null, null, 'ABOUT')
    expect(giKgCoseIdealEdgeLength(edge, 'main')).toBe(80)
  })
})

describe('giKgCoseEdgeElasticity', () => {
  it('uses default elasticity inside tc: clusters', () => {
    const edge = mockEdge('tc:x', 'tc:x', 'ABOUT')
    expect(giKgCoseEdgeElasticity(edge, 'main')).toBe(100)
  })

  it('uses semantic elasticity for ABOUT outside tc:', () => {
    const edge = mockEdge(null, null, 'ABOUT')
    expect(giKgCoseEdgeElasticity(edge, 'main')).toBe(200)
  })
})

describe('giKgCoseLayoutOptionsMainFallback', () => {
  it('uses flat repulsion/edge length (no per-node callbacks)', () => {
    const o = giKgCoseLayoutOptionsMainFallback()
    expect(o.name).toBe('cose')
    expect(o.nestingFactor).toBe(1.52)
    expect(o.numIter).toBe(2500)
    expect(typeof o.nodeRepulsion).toBe('function')
    expect(typeof o.idealEdgeLength).toBe('function')
    expect((o.nodeRepulsion as () => number)()).toBe(880_000)
    expect((o.idealEdgeLength as () => number)()).toBe(96)
  })
})

describe('giKgCoseLayoutOptionsCompact', () => {
  it('matches minimap gravity and disables animation', () => {
    const o = giKgCoseLayoutOptionsCompact()
    expect(o.gravity).toBe(0.32)
    expect(o.animate).toBe(false)
    expect(o.nestingFactor).toBe(1.52)
  })
})
