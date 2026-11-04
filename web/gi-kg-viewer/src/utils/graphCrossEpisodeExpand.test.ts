import type { Core } from 'cytoscape'
import { describe, expect, it, vi } from 'vitest'

import type { RawGraphNode } from '../types/artifact'
import {
  graphNodeExpandableForCrossEpisodeExpand,
  syncCrossEpisodeExpandNodeClasses,
} from './graphCrossEpisodeExpand'

type FakeNode = {
  id: () => string
  isNode?: () => boolean
  empty?: () => boolean
  degree?: () => number
  classes: Set<string>
  addClass: (c: string) => void
  removeClass: (c: string) => void
}

function fakeNode(id: string, degree = 3): FakeNode {
  const classes = new Set<string>()
  return {
    id: () => id,
    isNode: () => true,
    empty: () => false,
    degree: () => degree,
    classes,
    addClass: (c) => void classes.add(c),
    removeClass: (c) => void classes.delete(c),
  }
}

function fakeCore(nodes: FakeNode[]): Core {
  const byId = new Map(nodes.map((n) => [n.id(), n]))
  const missing = { empty: () => true }
  return {
    $id: (id: string) => byId.get(id) ?? missing,
    batch: (fn: () => void) => fn(),
    nodes: () => ({ forEach: (cb: (n: FakeNode) => void) => nodes.forEach(cb) }),
  } as unknown as Core
}

const topic = (id: string): RawGraphNode => ({ id, type: 'Topic' }) as RawGraphNode

describe('graphNodeExpandableForCrossEpisodeExpand', () => {
  it('accepts a high-degree canonical Topic node', () => {
    const core = fakeCore([fakeNode('topic:ai', 3)])
    expect(graphNodeExpandableForCrossEpisodeExpand(core, 'topic:ai', topic('topic:ai'))).toBe(true)
  })

  it('accepts Person and Entity types with layer-prefixed ids', () => {
    const core = fakeCore([fakeNode('g:person:ann', 2), fakeNode('k:org:acme', 2)])
    expect(
      graphNodeExpandableForCrossEpisodeExpand(core, 'g:person:ann', {
        id: 'g:person:ann',
        type: 'Person',
      } as RawGraphNode),
    ).toBe(true)
    expect(
      graphNodeExpandableForCrossEpisodeExpand(core, 'k:org:acme', {
        id: 'k:org:acme',
        type: 'Entity',
      } as RawGraphNode),
    ).toBe(true)
  })

  it('rejects Episode nodes', () => {
    const core = fakeCore([fakeNode('episode:1', 5)])
    expect(
      graphNodeExpandableForCrossEpisodeExpand(core, 'episode:1', {
        id: 'episode:1',
        type: 'Episode',
      } as RawGraphNode),
    ).toBe(false)
  })

  it('rejects non-expandable node types and null raw nodes', () => {
    const core = fakeCore([fakeNode('topic:x', 5)])
    expect(
      graphNodeExpandableForCrossEpisodeExpand(core, 'topic:x', {
        id: 'topic:x',
        type: 'Insight',
      } as RawGraphNode),
    ).toBe(false)
    expect(graphNodeExpandableForCrossEpisodeExpand(core, 'topic:x', null)).toBe(false)
  })

  it('rejects ids that are not canonical person/org/topic CIL ids', () => {
    const core = fakeCore([fakeNode('quote:1', 5)])
    expect(
      graphNodeExpandableForCrossEpisodeExpand(core, 'quote:1', topic('quote:1')),
    ).toBe(false)
  })

  it('rejects when the node is absent from the core', () => {
    const core = fakeCore([])
    expect(graphNodeExpandableForCrossEpisodeExpand(core, 'topic:ai', topic('topic:ai'))).toBe(
      false,
    )
  })

  it('rejects an element that is not a node', () => {
    const n = fakeNode('topic:ai', 5)
    n.isNode = () => false
    expect(
      graphNodeExpandableForCrossEpisodeExpand(fakeCore([n]), 'topic:ai', topic('topic:ai')),
    ).toBe(false)
  })

  it('rejects degree 1 and 0 nodes', () => {
    const core = fakeCore([fakeNode('topic:a', 1), fakeNode('topic:b', 0)])
    expect(graphNodeExpandableForCrossEpisodeExpand(core, 'topic:a', topic('topic:a'))).toBe(false)
    expect(graphNodeExpandableForCrossEpisodeExpand(core, 'topic:b', topic('topic:b'))).toBe(false)
  })

  it('swallows cytoscape errors and reports not expandable', () => {
    const core = {
      $id: () => {
        throw new Error('boom')
      },
    } as unknown as Core
    expect(graphNodeExpandableForCrossEpisodeExpand(core, 'topic:ai', topic('topic:ai'))).toBe(
      false,
    )
  })
})

describe('syncCrossEpisodeExpandNodeClasses', () => {
  const rawFor = (id: string) => topic(id)

  it('marks an expanded seed and no eligible class on it', () => {
    const n = fakeNode('topic:ai')
    n.classes.add('graph-expand-eligible')
    syncCrossEpisodeExpandNodeClasses(fakeCore([n]), {
      isExpandedSeed: () => true,
      rawNode: rawFor,
    })
    expect([...n.classes]).toEqual(['graph-expand-seed'])
  })

  it('marks eligible nodes when no corpus probe is supplied', () => {
    const n = fakeNode('topic:ai')
    syncCrossEpisodeExpandNodeClasses(fakeCore([n]), {
      isExpandedSeed: () => false,
      rawNode: rawFor,
    })
    expect([...n.classes]).toEqual(['graph-expand-eligible'])
  })

  it('requires the corpus probe to return true when supplied', () => {
    const yes = fakeNode('topic:yes')
    const no = fakeNode('topic:no')
    const pending = fakeNode('topic:pending')
    const probe = (id: string) => (id === 'topic:yes' ? true : id === 'topic:no' ? false : undefined)
    syncCrossEpisodeExpandNodeClasses(fakeCore([yes, no, pending]), {
      isExpandedSeed: () => false,
      rawNode: rawFor,
      corpusWouldAppendOutsideGraph: probe,
    })
    expect(yes.classes.has('graph-expand-eligible')).toBe(true)
    expect(no.classes.size).toBe(0)
    expect(pending.classes.size).toBe(0)
  })

  it('clears stale classes from nodes that are no longer expandable', () => {
    const n = fakeNode('topic:ai', 1)
    n.classes.add('graph-expand-eligible')
    n.classes.add('graph-expand-seed')
    syncCrossEpisodeExpandNodeClasses(fakeCore([n]), {
      isExpandedSeed: () => false,
      rawNode: rawFor,
    })
    expect(n.classes.size).toBe(0)
  })

  it('skips non-node elements and survives a throwing callback', () => {
    const notNode = fakeNode('topic:x')
    notNode.isNode = () => false
    const bad = fakeNode('topic:bad')
    const good = fakeNode('topic:good')
    const rawNode = vi.fn((id: string) => {
      if (id === 'topic:bad') throw new Error('boom')
      return topic(id)
    })
    syncCrossEpisodeExpandNodeClasses(fakeCore([notNode, bad, good]), {
      isExpandedSeed: () => false,
      rawNode,
    })
    expect(notNode.classes.size).toBe(0)
    expect(good.classes.has('graph-expand-eligible')).toBe(true)
  })
})
