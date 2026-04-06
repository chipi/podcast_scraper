import { describe, expect, it } from 'vitest'
import {
  GRAPH_NODE_UNKNOWN_FILL,
  graphNodeFill,
  graphNodeLegendLabel,
  graphNodeTypeStyles,
  semanticTypeForLegendVisual,
} from './colors'

describe('graphNodeLegendLabel', () => {
  it('humanises Entity_person', () => {
    expect(graphNodeLegendLabel('Entity_person')).toBe('Entity (person)')
  })

  it('humanises Entity_organization', () => {
    expect(graphNodeLegendLabel('Entity_organization')).toBe('Entity (organization)')
  })

  it('passes through other keys', () => {
    expect(graphNodeLegendLabel('Topic')).toBe('Topic')
    expect(graphNodeLegendLabel('Episode')).toBe('Episode')
  })
})

describe('semanticTypeForLegendVisual', () => {
  it('maps Entity_person to Entity', () => {
    expect(semanticTypeForLegendVisual('Entity_person')).toBe('Entity')
  })

  it('maps Entity_organization to Entity', () => {
    expect(semanticTypeForLegendVisual('Entity_organization')).toBe('Entity')
  })

  it('passes through non-entity keys', () => {
    expect(semanticTypeForLegendVisual('Insight')).toBe('Insight')
  })
})

describe('graphNodeFill', () => {
  it('returns correct fill for known types', () => {
    expect(graphNodeFill('Episode')).toBe(graphNodeTypeStyles.Episode.background)
    expect(graphNodeFill('Insight')).toBe(graphNodeTypeStyles.Insight.background)
  })

  it('falls back to Entity_person for bare Entity', () => {
    expect(graphNodeFill('Entity')).toBe(graphNodeTypeStyles.Entity_person.background)
  })

  it('returns unknown fill for unrecognised type', () => {
    expect(graphNodeFill('Alien')).toBe(GRAPH_NODE_UNKNOWN_FILL)
  })
})
