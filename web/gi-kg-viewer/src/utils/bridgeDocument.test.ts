import { describe, expect, it } from 'vitest'
import {
  bridgeIdentityForGraphNodeId,
  crossLayerPresenceLabel,
  normalizeCilIdForBridgeLookup,
  parseBridgeDocument,
} from './bridgeDocument'

describe('bridgeDocument', () => {
  it('normalizes g:/k:/kg: prefixes for lookup', () => {
    expect(normalizeCilIdForBridgeLookup('g:person:alice')).toBe('person:alice')
    expect(normalizeCilIdForBridgeLookup('k:person:alice')).toBe('person:alice')
    expect(normalizeCilIdForBridgeLookup('k:kg:topic:x')).toBe('topic:x')
  })

  it('parses identities from raw JSON', () => {
    const doc = parseBridgeDocument({
      schema_version: '1.0',
      identities: [
        {
          id: 'person:a',
          type: 'person',
          display_name: 'A',
          aliases: ['a1'],
          sources: { gi: true, kg: false },
        },
      ],
    })
    expect(doc?.identities).toHaveLength(1)
    expect(doc?.identities?.[0].sources).toEqual({ gi: true, kg: false })
  })

  it('crossLayerPresenceLabel covers gi, kg, both', () => {
    expect(crossLayerPresenceLabel({ gi: true, kg: true })).toBe(
      'Grounded Insights and Knowledge graph',
    )
    expect(crossLayerPresenceLabel({ gi: true, kg: false })).toBe('Grounded Insights only')
    expect(crossLayerPresenceLabel({ gi: false, kg: true })).toBe('Knowledge graph only')
  })

  it('bridgeIdentityForGraphNodeId matches prefixed cytoscape id', () => {
    const doc = parseBridgeDocument({
      identities: [
        { id: 'person:bob', type: 'person', display_name: '', aliases: [], sources: { gi: true, kg: true } },
      ],
    })
    const row = bridgeIdentityForGraphNodeId(doc, 'g:person:bob')
    expect(row?.sources.gi).toBe(true)
  })
})
