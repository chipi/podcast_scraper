import { describe, expect, it } from 'vitest'
import { visualGroupForNode, visualNodeTypeCounts } from './visualGroup'

describe('visualGroupForNode', () => {
  it('returns ? for null/undefined', () => {
    expect(visualGroupForNode(null)).toBe('?')
    expect(visualGroupForNode(undefined)).toBe('?')
  })

  it('passes through non-Entity types', () => {
    expect(visualGroupForNode({ type: 'Topic' })).toBe('Topic')
    expect(visualGroupForNode({ type: 'Insight' })).toBe('Insight')
  })

  it('maps GIL Person to Entity_person styling group', () => {
    expect(visualGroupForNode({ type: 'Person', properties: { name: 'Ada' } })).toBe(
      'Entity_person',
    )
  })

  it('defaults Entity to Entity_person when no entity_kind', () => {
    expect(visualGroupForNode({ type: 'Entity' })).toBe('Entity_person')
  })

  it('maps CIL kind person|org on Entity', () => {
    expect(visualGroupForNode({ type: 'Entity', properties: { kind: 'person' } })).toBe(
      'Entity_person',
    )
    expect(visualGroupForNode({ type: 'Entity', properties: { kind: 'org' } })).toBe(
      'Entity_organization',
    )
  })

  it('maps organization variants to Entity_organization', () => {
    for (const kind of ['organization', 'org', 'company', 'corporation', 'institution']) {
      expect(visualGroupForNode({ type: 'Entity', properties: { entity_kind: kind } })).toBe(
        'Entity_organization',
      )
    }
  })

  it('maps person to Entity_person', () => {
    expect(
      visualGroupForNode({ type: 'Entity', properties: { entity_kind: 'person' } }),
    ).toBe('Entity_person')
  })

  it('handles case-insensitive entity_kind', () => {
    expect(
      visualGroupForNode({ type: 'Entity', properties: { entity_kind: 'Organization' } }),
    ).toBe('Entity_organization')
  })

  it('treats blank entity_kind as person', () => {
    expect(
      visualGroupForNode({ type: 'Entity', properties: { entity_kind: '  ' } }),
    ).toBe('Entity_person')
  })
})

describe('visualNodeTypeCounts', () => {
  it('counts visual groups', () => {
    const nodes = [
      { type: 'Topic' },
      { type: 'Topic' },
      { type: 'Entity', properties: { entity_kind: 'org' } },
      { type: 'Entity' },
    ]
    expect(visualNodeTypeCounts(nodes)).toEqual({
      Topic: 2,
      Entity_organization: 1,
      Entity_person: 1,
    })
  })

  it('returns empty for empty array', () => {
    expect(visualNodeTypeCounts([])).toEqual({})
  })

  it('handles non-array gracefully', () => {
    expect(visualNodeTypeCounts(null as unknown as [])).toEqual({})
  })
})
