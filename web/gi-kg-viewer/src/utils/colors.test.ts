import { describe, expect, it } from 'vitest'
import {
  GRAPH_NODE_UNKNOWN_FILL,
  graphNodeFill,
  graphNodeLegendLabel,
  graphNodeTypeStyles,
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

/* graph-v3 L palette collision guard (harden follow-up).
 * Before tier 3 L both Entity_person and TopicCluster rendered #9775fa
 * purple, so Persons visually merged into semantic-cluster boxes on the
 * graph. L split them: Entity_person → amber #c49a28, TopicCluster →
 * violet #8574c7. A future palette edit could reintroduce the collision
 * silently — this test locks in the invariant so it will fail loudly. */
describe('graphNodeTypeStyles — brand-palette collision guards', () => {
  it('Entity_person and TopicCluster paint distinct backgrounds', () => {
    expect(graphNodeTypeStyles.Entity_person.background).not.toBe(
      graphNodeTypeStyles.TopicCluster.background,
    )
  })

  it('every node type has a unique background hex — no silent palette collisions', () => {
    const seen = new Map<string, string[]>()
    for (const [type, style] of Object.entries(graphNodeTypeStyles)) {
      const hex = style.background.toLowerCase()
      const bucket = seen.get(hex) ?? []
      bucket.push(type)
      seen.set(hex, bucket)
    }
    // graphNodeTypeStyles intentionally aliases the generic `Entity` fallback
    // to the same fill as `Entity_person` (see graphNodeFill). Allow that
    // one deliberate collision; every other pairing is a bug.
    const collisions = [...seen.entries()]
      .filter(([, types]) => types.length > 1)
      .filter(([, types]) => !(types.length === 2 && types.includes('Entity') && types.includes('Entity_person')))
    expect(collisions).toEqual([])
  })
})
