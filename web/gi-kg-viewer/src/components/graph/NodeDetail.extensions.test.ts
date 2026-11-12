// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, h } from 'vue'

vi.mock('../../api/userPreferencesApi', () => ({
  fetchUserPreferences: vi.fn().mockResolvedValue(null),
  patchUserPreferences: vi.fn().mockResolvedValue(null),
  replaceUserPreferences: vi.fn().mockResolvedValue(null),
}))
vi.mock('../../api/cilApi', async (orig) => ({
  ...(await orig<typeof import('../../api/cilApi')>()),
  fetchTopicTimeline: vi.fn().mockResolvedValue({ topic_id: 't', episodes: [] }),
  fetchTopicTimelineMerged: vi.fn().mockResolvedValue({ topic_ids: [], episodes: [] }),
}))

import { fetchTopicTimeline, fetchTopicTimelineMerged } from '../../api/cilApi'
import { nodeDetailExtensions, type NodeDetailContext } from '../../extensions'
import { useShellStore } from '../../stores/shell'
import type { ParsedArtifact, RawGraphNode } from '../../types/artifact'
import NodeDetail from './NodeDetail.vue'

const STUBS = {
  PersonLandingView: true,
  PodcastNodeView: true,
  TopicEntityView: true,
  TranscriptViewerDialog: true,
  PodcastCover: true,
  GraphConnectionsSection: {
    name: 'GraphConnectionsSection',
    props: ['viewArtifact', 'nodeId', 'aggregatedNeighborRows', 'connectionsEmptyHint', 'compoundNeighborhood', 'denseNeighborList'],
    template: '<div data-testid="connections-stub" />',
  },
  NodeEnrichmentSection: {
    name: 'NodeEnrichmentSection',
    props: ['nodeId', 'nodeType'],
    emits: ['has-content'],
    template: '<div />',
  },
  HelpTip: { name: 'HelpTip', template: '<div><slot /></div>' },
}

function artifactOf(nodes: RawGraphNode[]): ParsedArtifact {
  return { name: 'merged', kind: 'both', episodeId: null, nodes: nodes.length, edges: 0, nodeTypes: {}, data: { nodes, edges: [] } }
}

const topic = (id: string, label: string): RawGraphNode => ({ id, type: 'Topic', properties: { label } }) as RawGraphNode

/** A section that shows what it was given. */
const ctxSection = (tag: string) =>
  defineComponent({
    props: { ctx: { type: Object, required: true } },
    setup(props) {
      return () => h('p', { 'data-testid': `section-${tag}` }, `${(props.ctx as NodeDetailContext).nodeId}|${(props.ctx as NodeDetailContext).nodeType}`)
    },
  })

function mountDetail(props: Record<string, unknown>) {
  return mount(NodeDetail, { props, attachTo: document.body, global: { stubs: STUBS } })
}

function readyShell(): void {
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  shell.cilQueriesApiAvailable = true
}

describe('NodeDetail extensions', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(fetchTopicTimeline).mockClear()
    vi.mocked(fetchTopicTimelineMerged).mockClear()
  })
  afterEach(() => nodeDetailExtensions.reset())

  it('shows a node only an extension knows, with its title and avatar group', () => {
    nodeDetailExtensions.register({
      id: 'x',
      knowsNode: (id) => id === 'grp:1',
      title: (ctx) => (ctx.nodeId === 'grp:1' ? 'Group One' : null),
      avatarGroup: (ctx) => (ctx.nodeId === 'grp:1' ? 'Topic' : null),
    })
    const w = mountDetail({ viewArtifact: artifactOf([]), nodeId: 'grp:1' })
    expect(w.find('aside').exists()).toBe(true)
    expect(w.text()).toContain('Group One')
    expect(w.find('.text-xl').exists()).toBe(true)
    w.unmount()
  })

  it('renders sections in their regions with the node context', () => {
    nodeDetailExtensions.register({
      id: 'x',
      sections: [
        { region: 'header', component: ctxSection('header') },
        { region: 'overview', component: ctxSection('overview') },
        { region: 'after-aliases', component: ctxSection('after-aliases') },
      ],
    })
    const w = mountDetail({ viewArtifact: artifactOf([topic('topic:a', 'A')]), nodeId: 'topic:a' })
    for (const r of ['header', 'overview', 'after-aliases']) {
      expect(w.get(`[data-testid="section-${r}"]`).text()).toBe('topic:a|Topic')
    }
    w.unmount()
  })

  it('adds an extension tab for the nodes it shows for, with its panel and the ctx', async () => {
    readyShell()
    nodeDetailExtensions.register({
      id: 'x',
      tabs: [
        { id: 'ext-tab', label: 'Ext', shows: (ctx) => ctx.isTopicNode, component: ctxSection('tab') },
      ],
    })
    const w = mountDetail({ viewArtifact: artifactOf([topic('topic:a', 'A')]), nodeId: 'topic:a', embedInRail: true })
    await flushPromises()
    const tab = w.get('[data-testid="node-detail-rail-tab-ext:ext-tab"]')
    expect(tab.text()).toBe('Ext')
    await tab.trigger('click')
    expect(tab.attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="section-tab"]').text()).toBe('topic:a|Topic')
    w.unmount()
  })

  it('shows no extension tab for a node its test refuses', async () => {
    readyShell()
    nodeDetailExtensions.register({
      id: 'x',
      tabs: [{ id: 'ext-tab', label: 'Ext', shows: () => false, component: ctxSection('tab') }],
    })
    const w = mountDetail({ viewArtifact: artifactOf([topic('topic:a', 'A')]), nodeId: 'topic:a', embedInRail: true })
    await flushPromises()
    expect(w.find('[data-testid="node-detail-rail-tab-ext:ext-tab"]').exists()).toBe(false)
    w.unmount()
  })

  it('replaces the neighbour list and the minimap neighbourhood', async () => {
    const rows = [{ id: 'n1' }]
    nodeDetailExtensions.register({
      id: 'x',
      connections: () => ({ rows: rows as never, emptyHint: 'none here', compound: { compoundId: 'grp:1', memberIds: ['topic:a'] } }),
    })
    const w = mountDetail({ viewArtifact: artifactOf([topic('topic:a', 'A')]), nodeId: 'topic:a', embedInRail: true })
    await w.get('#node-detail-rail-tab-neighbourhood').trigger('click')
    const c = w.getComponent({ name: 'GraphConnectionsSection' })
    expect(c.props('aggregatedNeighborRows')).toEqual(rows)
    expect(c.props('connectionsEmptyHint')).toBe('none here')
    expect(c.props('compoundNeighborhood')).toEqual({ compoundId: 'grp:1', memberIds: ['topic:a'] })
    w.unmount()
  })

  it('names the kind of a node outside the slice, and renders its body section', () => {
    nodeDetailExtensions.register({
      id: 'x',
      kindOf: (id) => (id.startsWith('claim:') ? 'Claim' : null),
      sections: [{ region: 'body', component: ctxSection('body') }],
    })
    const w = mountDetail({ viewArtifact: artifactOf([]), nodeId: 'claim:1', embedInRail: true })
    // The kind alone shows the panel (no `knowsNode`), and the body section sees it.
    expect(w.find('aside').exists()).toBe(true)
    expect(w.get('[data-testid="section-body"]').text()).toBe('claim:1|Claim')
    w.unmount()
  })

  it('keeps the keys a section shows out of the generic property list', () => {
    const node = { id: 'n:1', type: 'Thing', properties: { label: 'N', weight: 3, colour: 'red' } } as RawGraphNode
    const mountWith = () => mountDetail({ viewArtifact: artifactOf([node]), nodeId: 'n:1' })
    const plain = mountWith()
    expect(plain.text()).toContain('Weight')
    plain.unmount()
    nodeDetailExtensions.register({ id: 'x', hiddenProperties: (ctx) => (ctx.nodeType === 'Thing' ? ['weight'] : []) })
    const w = mountWith()
    expect(w.text()).not.toContain('Weight')
    expect(w.text()).toContain('Colour')
    w.unmount()
  })

  it('gives sections the node, the full graph and the panel actions', async () => {
    readyShell()
    let seen: NodeDetailContext | null = null
    nodeDetailExtensions.register({
      id: 'x',
      sections: [
        {
          region: 'body',
          component: defineComponent({
            props: { ctx: { type: Object, required: true } },
            setup(props) {
              seen = props.ctx as NodeDetailContext
              return () => h('p')
            },
          }),
        },
      ],
    })
    const art = artifactOf([topic('topic:a', 'A')])
    const w = mountDetail({ viewArtifact: art, nodeId: 'topic:a' })
    await flushPromises()
    expect(seen!.node?.id).toBe('topic:a')
    expect(seen!.fullArtifact).toEqual(art)
    seen!.prefillSearch('  rates  ')
    seen!.prefillSearch('   ')
    seen!.openSearchFilters({ groundedOnly: true, minConfidence: 0.5 })
    seen!.focusOnGraph('topic:b')
    expect(w.emitted('prefill-semantic-search')).toEqual([[{ query: 'rates' }]])
    expect(w.emitted('open-search-insight-filters')).toEqual([[{ groundedOnly: true, minConfidence: 0.5 }]])
    expect(w.emitted('go-graph')).toHaveLength(1)
    w.unmount()
  })
})
