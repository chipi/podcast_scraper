import { afterEach, describe, expect, it } from 'vitest'

import { graphTypeTraits } from '../extensions'
import { GRAPH_NODE_UNKNOWN_FILL, graphNodeFill, graphNodeLegendLabel, graphNodeTypeChrome } from './colors'

afterEach(() => graphTypeTraits.reset())

describe('graphTypeTraits.chrome', () => {
  it('colours and labels a type the viewer does not style itself', () => {
    graphTypeTraits.register({
      id: 't',
      chrome: { Claim: { background: '#123456', border: '#654321', labelColor: '#ffffff', legendLabel: 'Claim (ext)' } },
    })
    expect(graphNodeFill('Claim')).toBe('#123456')
    expect(graphNodeTypeChrome('Claim')).toEqual({ background: '#123456', border: '#654321', labelColor: '#ffffff' })
    expect(graphNodeLegendLabel('Claim')).toBe('Claim (ext)')
  })

  it('never overrides a type the viewer styles, and an unknown type stays grey', () => {
    graphTypeTraits.register({ id: 't', chrome: { Topic: { background: '#000000', border: '#000000', labelColor: '#000000', legendLabel: 'X' } } })
    expect(graphNodeFill('Topic')).not.toBe('#000000')
    expect(graphNodeLegendLabel('Topic')).toBe('Topic')
    expect(graphNodeFill('Nope')).toBe(GRAPH_NODE_UNKNOWN_FILL)
  })
})
