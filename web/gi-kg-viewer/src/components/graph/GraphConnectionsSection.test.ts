// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { ParsedArtifact } from '../../types/artifact'
import type { GraphNeighborRow } from '../../utils/graphNeighbors'
import { useGraphHandoffStore } from '../../stores/graphHandoff'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useShellStore } from '../../stores/shell'
import GraphConnectionsSection from './GraphConnectionsSection.vue'

const SECTION = '[data-testid="graph-connections-section"]'
const FOCUS_GRAPH = '[data-testid="graph-connection-focus-graph"]'
const OPEN_LIBRARY = '[data-testid="graph-connection-open-library"]'
const PREFILL = '[data-testid="graph-connection-prefill-search"]'

/** Minimal merged graph: a center Topic with two neighbours (an Episode + another Topic). */
function makeArtifact(): ParsedArtifact {
  return {
    name: 'merged',
    kind: 'both',
    episodeId: null,
    nodes: 3,
    edges: 2,
    nodeTypes: {},
    data: {
      nodes: [
        { id: 'topic:center', type: 'Topic', properties: { label: 'Center topic' } },
        {
          id: 'ep:1',
          type: 'Episode',
          properties: {
            title: 'Episode One',
            metadata_relative_path: 'feeds/x/ep1/metadata.json',
          },
        },
        { id: 'topic:other', type: 'Topic', properties: { label: 'Other topic' } },
      ],
      edges: [
        { from: 'topic:center', to: 'ep:1', type: 'mentions' },
        { from: 'topic:center', to: 'topic:other', type: 'related' },
      ],
    },
  }
}

/** Artifact whose center node has no incident edges (isolated). */
function makeIsolatedArtifact(): ParsedArtifact {
  return {
    name: 'merged',
    kind: 'both',
    episodeId: null,
    nodes: 1,
    edges: 0,
    nodeTypes: {},
    data: {
      nodes: [{ id: 'topic:center', type: 'Topic', properties: { label: 'Center topic' } }],
      edges: [],
    },
  }
}

const STUBS = { GraphNeighborhoodMiniMap: true }

function mountSection(props: Record<string, unknown>) {
  return mount(GraphConnectionsSection, {
    props,
    attachTo: document.body,
    global: { stubs: STUBS },
  })
}

describe('GraphConnectionsSection', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders nothing when the center node is not in the view artifact', () => {
    const w = mountSection({ viewArtifact: makeArtifact(), nodeId: 'topic:missing' })
    expect(w.find(SECTION).exists()).toBe(false)
  })

  it('renders nothing when viewArtifact or nodeId is null', () => {
    expect(mountSection({ viewArtifact: null, nodeId: 'topic:center' }).find(SECTION).exists()).toBe(
      false,
    )
    expect(mountSection({ viewArtifact: makeArtifact(), nodeId: null }).find(SECTION).exists()).toBe(
      false,
    )
  })

  it('renders the section, minimap stub, and connection count when center is in view', () => {
    const w = mountSection({ viewArtifact: makeArtifact(), nodeId: 'topic:center' })
    expect(w.find(SECTION).exists()).toBe(true)
    expect(w.findComponent({ name: 'GraphNeighborhoodMiniMap' }).exists()).toBe(true)
    // Two neighbours -> "Connections (2)"
    expect(w.text()).toContain('Connections (2)')
    expect(w.findAll('[data-connection-node-id]')).toHaveLength(2)
  })

  it('renders one row per neighbour with label, type, and edge type', () => {
    const w = mountSection({ viewArtifact: makeArtifact(), nodeId: 'topic:center' })
    const rows = w.findAll('[data-connection-node-id]')
    const epRow = rows.find((r) => r.attributes('data-connection-node-id') === 'ep:1')!
    expect(epRow.text()).toContain('Episode One')
    expect(epRow.text()).toContain('Episode')
    expect(epRow.text()).toContain('(mentions)')
  })

  it('shows the default empty-state copy for an isolated center node', () => {
    const w = mountSection({ viewArtifact: makeIsolatedArtifact(), nodeId: 'topic:center' })
    expect(w.find(SECTION).exists()).toBe(true)
    expect(w.find('ul').exists()).toBe(false)
    expect(w.text()).toContain('No edges in this view')
  })

  it('honours a custom connectionsEmptyHint over the default copy', () => {
    const w = mountSection({
      viewArtifact: makeIsolatedArtifact(),
      nodeId: 'topic:center',
      connectionsEmptyHint: 'Custom empty hint',
    })
    expect(w.text()).toContain('Custom empty hint')
    expect(w.text()).not.toContain('No edges in this view')
  })

  it('clicking the G button fires the handoff, nav focus, and emits go-graph', async () => {
    const w = mountSection({ viewArtifact: makeArtifact(), nodeId: 'topic:center' })
    const handoff = useGraphHandoffStore()
    const nav = useGraphNavigationStore()
    const genBefore = handoff.generation
    const rows = w.findAll('[data-connection-node-id]')
    const epRow = rows.find((r) => r.attributes('data-connection-node-id') === 'ep:1')!

    await epRow.get(FOCUS_GRAPH).trigger('click')

    expect(w.emitted('go-graph')).toHaveLength(1)
    expect(nav.pendingFocusNodeId).toBe('ep:1')
    // canvasTapped accepted a new handoff envelope -> generation advances.
    expect(handoff.generation).toBeGreaterThan(genBefore)
  })

  it('only Episode neighbours render the Library (L) button', () => {
    const w = mountSection({ viewArtifact: makeArtifact(), nodeId: 'topic:center' })
    const rows = w.findAll('[data-connection-node-id]')
    const epRow = rows.find((r) => r.attributes('data-connection-node-id') === 'ep:1')!
    const topicRow = rows.find((r) => r.attributes('data-connection-node-id') === 'topic:other')!
    expect(epRow.find(OPEN_LIBRARY).exists()).toBe(true)
    expect(topicRow.find(OPEN_LIBRARY).exists()).toBe(false)
  })

  it('disables the Library button until health + corpus library API are available', async () => {
    const w = mountSection({ viewArtifact: makeArtifact(), nodeId: 'topic:center' })
    const shell = useShellStore()
    const epRow = w
      .findAll('[data-connection-node-id]')
      .find((r) => r.attributes('data-connection-node-id') === 'ep:1')!

    // Default store state: not healthy -> disabled, no emit on click.
    expect(epRow.get(OPEN_LIBRARY).attributes('disabled')).toBeDefined()
    await epRow.get(OPEN_LIBRARY).trigger('click')
    expect(w.emitted('open-library-episode')).toBeUndefined()

    // Enable: emits the resolved metadata path.
    shell.healthStatus = 'ok'
    shell.corpusLibraryApiAvailable = true
    await w.vm.$nextTick()
    expect(epRow.get(OPEN_LIBRARY).attributes('disabled')).toBeUndefined()
    await epRow.get(OPEN_LIBRARY).trigger('click')
    expect(w.emitted('open-library-episode')).toHaveLength(1)
    expect(w.emitted('open-library-episode')![0]).toEqual([
      { metadata_relative_path: 'feeds/x/ep1/metadata.json' },
    ])
  })

  it('disables the prefill (S) button until health is set, then emits the query', async () => {
    const w = mountSection({ viewArtifact: makeArtifact(), nodeId: 'topic:center' })
    const shell = useShellStore()
    const epRow = w
      .findAll('[data-connection-node-id]')
      .find((r) => r.attributes('data-connection-node-id') === 'ep:1')!

    expect(epRow.get(PREFILL).attributes('disabled')).toBeDefined()
    await epRow.get(PREFILL).trigger('click')
    expect(w.emitted('prefill-semantic-search')).toBeUndefined()

    shell.healthStatus = 'ok'
    await w.vm.$nextTick()
    expect(epRow.get(PREFILL).attributes('disabled')).toBeUndefined()
    await epRow.get(PREFILL).trigger('click')
    const evs = w.emitted('prefill-semantic-search')
    expect(evs).toHaveLength(1)
    expect((evs![0][0] as { query: string }).query).toContain('Episode One')
  })

  it('aggregated mode renders merged-row copy, count, and Via lines', () => {
    const aggregatedNeighborRows: GraphNeighborRow[] = [
      {
        id: 'ep:1',
        label: 'Episode One',
        type: 'Episode',
        visualType: 'Episode',
        edgeType: 'mentions',
        direction: 'out',
        viaMemberTopicIds: ['topic:center'],
      },
    ]
    const w = mountSection({
      viewArtifact: makeArtifact(),
      nodeId: 'topic:center',
      aggregatedNeighborRows,
    })
    expect(w.text()).toContain('Connections to other nodes (1)')
    expect(w.text()).toContain('Edges from member topics')
    // Via line resolves the member topic id back to its label.
    expect(w.text()).toContain('Via: Center topic')
  })

  it('aggregated empty rows ([]) still render the section with the empty hint', () => {
    const w = mountSection({
      viewArtifact: makeArtifact(),
      nodeId: 'topic:center',
      aggregatedNeighborRows: [],
      connectionsEmptyHint: 'No member edges',
    })
    expect(w.find(SECTION).exists()).toBe(true)
    expect(w.text()).toContain('Connections to other nodes (0)')
    expect(w.text()).toContain('No member edges')
  })
})
