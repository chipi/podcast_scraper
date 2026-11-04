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
  InsightNodeView: true,
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

  it('merges an extension’s topics on the Timeline tab, and passes its toggle state', async () => {
    readyShell()
    const seen: boolean[] = []
    nodeDetailExtensions.register({
      id: 'x',
      timeline: (ctx, toggled) => {
        seen.push(toggled)
        return toggled && ctx.isTopicNode ? ['topic:a', 'topic:b'] : null
      },
      timelineToggle: (ctx) =>
        ctx.isTopicNode ? { label: 'Group', titleOn: 'on', titleOff: 'off', testid: 'tl-toggle' } : null,
    })
    const w = mountDetail({ viewArtifact: artifactOf([topic('topic:a', 'A')]), nodeId: 'topic:a', embedInRail: true })
    await flushPromises()
    expect(fetchTopicTimeline).toHaveBeenCalledTimes(1)
    await w.get('#node-detail-rail-tab-timeline').trigger('click')
    await flushPromises()
    const toggle = w.get('[data-testid="tl-toggle"]')
    expect(toggle.attributes('title')).toBe('off')
    await toggle.trigger('click')
    await flushPromises()
    expect(seen).toContain(true)
    expect(fetchTopicTimelineMerged).toHaveBeenCalledWith('/corpus', ['topic:a', 'topic:b'])
    expect(w.get('[data-testid="tl-toggle"]').attributes('aria-pressed')).toBe('true')
    w.unmount()
  })

  it("'hidden' removes the Timeline tab even for a topic", async () => {
    readyShell()
    nodeDetailExtensions.register({ id: 'x', timeline: () => 'hidden' })
    const w = mountDetail({ viewArtifact: artifactOf([topic('topic:a', 'A')]), nodeId: 'topic:a', embedInRail: true })
    await flushPromises()
    expect(w.find('#node-detail-rail-tab-timeline').exists()).toBe(false)
    expect(fetchTopicTimeline).not.toHaveBeenCalled()
    w.unmount()
  })

  it('a non-topic node with merged topics gets the Timeline tab and its body; an on toggle takes its colour', async () => {
    readyShell()
    nodeDetailExtensions.register({
      id: 'x',
      knowsNode: (id) => id === 'grp:1',
      timeline: (ctx) => (ctx.nodeId === 'grp:1' ? ['topic:a', 'topic:b'] : null),
      timelineToggle: (ctx) =>
        ctx.nodeId === 'grp:1' ? { label: 'G', titleOn: 'on', titleOff: 'off', testid: 'tl-toggle', color: 'rgb(1, 2, 3)' } : null,
    })
    const w = mountDetail({ viewArtifact: artifactOf([]), nodeId: 'grp:1', embedInRail: true })
    await flushPromises()
    expect(fetchTopicTimelineMerged).toHaveBeenCalledWith('/corpus', ['topic:a', 'topic:b'])
    await w.get('#node-detail-rail-tab-timeline').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="node-detail-inline-timeline"]').exists()).toBe(true)
    const toggle = w.get('[data-testid="tl-toggle"]')
    expect(toggle.attributes('style') ?? '').not.toContain('rgb(1, 2, 3)')
    await toggle.trigger('click')
    expect(w.get('[data-testid="tl-toggle"]').attributes('style')).toContain('rgb(1, 2, 3)')
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
})
