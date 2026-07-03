// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { ParsedArtifact, RawGraphNode } from '../../types/artifact'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import NodeDetail from './NodeDetail.vue'

// Heavy / API-driven children are stubbed: GraphConnectionsSection (cytoscape
// minimap), TranscriptViewerDialog, PodcastCover, and HelpTip (slot popover).
const STUBS = {
  PersonLandingView: { name: 'PersonLandingView', template: '<div data-testid="person-landing-view" />' },
  GraphConnectionsSection: true,
  TranscriptViewerDialog: true,
  PodcastCover: true,
  TopicEntityView: { name: 'TopicEntityView', template: '<div data-testid="topic-entity-view" />' },
  NodeEnrichmentSection: true,
  // Keep HelpTip rendering its default slot so trigger labels are assertable,
  // but stub the popover machinery to a passthrough.
  HelpTip: {
    name: 'HelpTip',
    props: ['buttonText', 'buttonAriaLabel', 'buttonClass', 'prefWidth'],
    template:
      '<div data-stub="help-tip"><button :aria-label="buttonAriaLabel">{{ buttonText }}</button><slot /></div>',
  },
}

function artifactOf(nodes: RawGraphNode[], edges: ParsedArtifact['data']['edges'] = []): ParsedArtifact {
  return {
    name: 'merged',
    kind: 'both',
    episodeId: null,
    nodes: nodes.length,
    edges: edges?.length ?? 0,
    nodeTypes: {},
    data: { nodes, edges },
  }
}

function mountDetail(props: Record<string, unknown>) {
  return mount(NodeDetail, {
    props,
    attachTo: document.body,
    global: { stubs: STUBS },
  })
}

describe('NodeDetail', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders nothing when nodeId is null', () => {
    const w = mountDetail({ viewArtifact: artifactOf([]), nodeId: null })
    expect(w.find('aside').exists()).toBe(false)
  })

  it('renders nothing when the node is absent and there is no topic cluster', () => {
    const w = mountDetail({ viewArtifact: artifactOf([]), nodeId: 'missing' })
    expect(w.find('aside').exists()).toBe(false)
  })

  it('renders the person view for an out-of-slice person id even when the node is absent', () => {
    // A co-speaker sourced from the full server-side relational graph isn't in
    // the current viewer slice → node lookup returns null. The rail must still
    // render the person view (which loads from server endpoints) instead of an
    // empty "Node" shell, with a name humanized from the id slug.
    const w = mountDetail({
      viewArtifact: artifactOf([]),
      nodeId: 'g:person:robert-armstrong',
      embedInRail: true,
    })
    expect(w.find('aside').exists()).toBe(true)
    expect(w.find('[data-testid="person-landing-view"]').exists()).toBe(true)
    expect(w.find('.node-detail-primary-title').text()).toBe('Robert Armstrong')
  })

  it('renders the panel with the node display name and type chip for a generic node', () => {
    const art = artifactOf([
      { id: 'ep:1', type: 'Episode', properties: { title: 'My Episode' } },
    ])
    const w = mountDetail({ viewArtifact: art, nodeId: 'ep:1' })
    expect(w.find('aside').exists()).toBe(true)
    expect(w.find('.node-detail-primary-title').text()).toBe('My Episode')
    // Non-embedInRail shows the node-kind row with the type chip.
    expect(w.find('[data-testid="node-detail-kind-row"]').text()).toContain('Episode')
  })

  it('emits close when the ✕ button is clicked (non-embedded mode)', async () => {
    const art = artifactOf([{ id: 'ep:1', type: 'Episode', properties: { title: 'E' } }])
    const w = mountDetail({ viewArtifact: art, nodeId: 'ep:1' })
    await w.get('button[aria-label="Close detail"]').trigger('click')
    expect(w.emitted('close')).toHaveLength(1)
  })

  it('hides the close button and shows rail tabs in embedInRail mode', () => {
    const art = artifactOf([{ id: 'ep:1', type: 'Episode', properties: { title: 'E' } }])
    const w = mountDetail({ viewArtifact: art, nodeId: 'ep:1', embedInRail: true })
    expect(w.find('button[aria-label="Close detail"]').exists()).toBe(false)
    expect(w.find('[data-testid="node-detail-rail-tab-details"]').exists()).toBe(true)
    expect(w.find('[data-testid="node-detail-rail-tab-neighbourhood"]').exists()).toBe(true)
  })

  it('switches between Details and Neighbourhood rail tabs', async () => {
    const art = artifactOf([{ id: 'ep:1', type: 'Episode', properties: { title: 'E' } }])
    const w = mountDetail({ viewArtifact: art, nodeId: 'ep:1', embedInRail: true })
    const detailsTab = w.get('[data-testid="node-detail-rail-tab-details"]')
    const neighTab = w.get('[data-testid="node-detail-rail-tab-neighbourhood"]')
    expect(detailsTab.attributes('aria-selected')).toBe('true')
    await neighTab.trigger('click')
    expect(neighTab.attributes('aria-selected')).toBe('true')
    expect(detailsTab.attributes('aria-selected')).toBe('false')
  })

  // --- Topic node branch -----------------------------------------------------

  it('renders Topic gateway buttons and aliases; emits topic prefill + explore', async () => {
    const art = artifactOf([
      {
        id: 'topic:ai',
        type: 'Topic',
        properties: { label: 'Artificial Intelligence', aliases: ['AI', 'ML'] },
      },
    ])
    const w = mountDetail({ viewArtifact: art, nodeId: 'topic:ai' })
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    await w.vm.$nextTick()

    expect(w.find('[data-testid="node-detail-topic-aliases"]').text()).toContain('AI, ML')
    await w.get('[data-testid="node-detail-topic-prefill-search"]').trigger('click')
    expect(w.emitted('prefill-semantic-search')![0]).toEqual([
      { query: 'Artificial Intelligence' },
    ])

    await w.get('[data-testid="node-detail-topic-explore-filter"]').trigger('click')
    expect(w.emitted('open-explore-topic-filter')![0]).toEqual([
      { topic: 'Artificial Intelligence' },
    ])
  })

  it('renders embedded TopicEntityView instead of "Open full Topic profile" button for topic nodes', () => {
    const art = artifactOf([
      { id: 'topic:ai', type: 'Topic', properties: { label: 'AI' } },
    ])
    const w = mountDetail({ viewArtifact: art, nodeId: 'topic:ai' })
    expect(w.find('[data-testid="node-detail-open-topic-profile"]').exists()).toBe(false)
    expect(w.find('[data-testid="topic-entity-view"]').exists()).toBe(true)
  })

  it('disables Topic gateway buttons until health is set', () => {
    const art = artifactOf([
      { id: 'topic:ai', type: 'Topic', properties: { label: 'AI' } },
    ])
    const w = mountDetail({ viewArtifact: art, nodeId: 'topic:ai' })
    expect(
      w.get('[data-testid="node-detail-topic-prefill-search"]').attributes('disabled'),
    ).toBeDefined()
  })

  // --- Person / Entity branch ------------------------------------------------

  it('renders Person panel: role summary, aliases, profile + gateway buttons', async () => {
    const art = artifactOf(
      [
        { id: 'person:ada', type: 'Person', properties: { name: 'Ada Lovelace', aliases: ['Ada'] } },
        { id: 'quote:1', type: 'Quote', properties: { text: 'Quote text' } },
        { id: 'ep:1', type: 'Episode', properties: { title: 'Ep' } },
      ],
      [
        { from: 'quote:1', to: 'person:ada', type: 'SPOKEN_BY' },
        { from: 'person:ada', to: 'ep:1', type: 'SPOKE_IN' },
      ],
    )
    const w = mountDetail({ viewArtifact: art, nodeId: 'person:ada' })
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    await w.vm.$nextTick()

    // NodeDetail now embeds PersonLandingView with the person details
    expect(w.find('[data-testid="person-landing-view"]').exists()).toBe(true)
    // The standalone "Open full Person profile" button is gone
    expect(w.find('[data-testid="node-detail-open-person-profile"]').exists()).toBe(false)
    // Gateway buttons still present
    await w.get('[data-testid="node-detail-person-entity-prefill-search"]').trigger('click')
    expect(w.emitted('prefill-semantic-search')![0]).toEqual([{ query: 'Ada Lovelace' }])

    // Person (not org) routes the second button to the speaker filter.
    await w.get('[data-testid="node-detail-person-entity-explore-filter"]').trigger('click')
    expect(w.emitted('open-explore-speaker-filter')![0]).toEqual([{ speaker: 'Ada Lovelace' }])
  })

  // --- Insight branch --------------------------------------------------------

  it('renders Insight details tip, gateway buttons, related topics and supporting quotes', async () => {
    const art = artifactOf(
      [
        {
          id: 'insight:1',
          type: 'Insight',
          properties: {
            text: 'A grounded insight about AI safety.',
            grounded: true,
            insight_type: 'claim',
          },
        },
        { id: 'topic:ai', type: 'Topic', properties: { label: 'AI Safety' } },
        { id: 'quote:1', type: 'Quote', properties: { text: 'Supporting quote one.' } },
      ],
      [
        { from: 'insight:1', to: 'topic:ai', type: 'ABOUT' },
        { from: 'insight:1', to: 'quote:1', type: 'SUPPORTED_BY' },
      ],
    )
    const w = mountDetail({ viewArtifact: art, nodeId: 'insight:1' })
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    await w.vm.$nextTick()

    // Grounded tip trigger present (grounded === true -> "Grounded" label).
    expect(w.find('[data-testid="node-detail-insight-details-tip"]').exists()).toBe(true)
    expect(w.find('[data-testid="node-detail-insight-details-tip"]').text()).toContain('Grounded')

    // Related topics list rendered.
    expect(w.find('[data-testid="node-detail-insight-related-topics"]').exists()).toBe(true)
    expect(w.findAll('[data-testid="node-detail-insight-related-topic-row"]')).toHaveLength(1)

    // Supporting quotes list rendered.
    expect(w.find('[data-testid="node-detail-insight-supporting-quotes"]').exists()).toBe(true)
    expect(w.text()).toContain('Supporting quote one.')

    // Prefill search emits a query from the insight text.
    await w.get('[data-testid="node-detail-insight-prefill-search"]').trigger('click')
    expect(
      (w.emitted('prefill-semantic-search')![0][0] as { query: string }).query,
    ).toContain('grounded insight')

    // Explore filters emit grounded flag + (no confidence) -> null.
    await w.get('[data-testid="node-detail-insight-explore-filters"]').trigger('click')
    expect(w.emitted('open-explore-insight-filters')![0]).toEqual([
      { groundedOnly: true, minConfidence: null },
    ])
  })

  it('clicking a related-topic row emits go-graph (graph-internal expansion)', async () => {
    const art = artifactOf(
      [
        { id: 'insight:1', type: 'Insight', properties: { text: 'Insight body.' } },
        { id: 'topic:ai', type: 'Topic', properties: { label: 'AI' } },
      ],
      [{ from: 'insight:1', to: 'topic:ai', type: 'ABOUT' }],
    )
    const w = mountDetail({ viewArtifact: art, nodeId: 'insight:1' })
    await w.get('[data-testid="node-detail-insight-related-topic-row"]').trigger('click')
    expect(w.emitted('go-graph')).toHaveLength(1)
  })

  it('collapses supporting quotes past the threshold and toggles to show all', async () => {
    const quoteNodes: RawGraphNode[] = []
    const edges: ParsedArtifact['data']['edges'] = []
    for (let i = 0; i < 7; i++) {
      quoteNodes.push({
        id: `quote:${i}`,
        type: 'Quote',
        properties: { text: `Quote number ${i}`, char_start: i * 10 },
      })
      edges!.push({ from: 'insight:1', to: `quote:${i}`, type: 'SUPPORTED_BY' })
    }
    const art = artifactOf(
      [{ id: 'insight:1', type: 'Insight', properties: { text: 'Insight.' } }, ...quoteNodes],
      edges,
    )
    const w = mountDetail({ viewArtifact: art, nodeId: 'insight:1' })

    // Collapsed: only first 5 of 7 quote rows shown.
    const section = w.get('[data-testid="node-detail-insight-supporting-quotes"]')
    expect(section.findAll('li')).toHaveLength(5)
    const toggle = w.get('[data-testid="node-detail-insight-supporting-quotes-toggle-expand"]')
    expect(toggle.text()).toContain('Show all 7')

    await toggle.trigger('click')
    expect(section.findAll('li')).toHaveLength(7)
    expect(toggle.text()).toContain('Show fewer')
  })

  // --- Quote branch ----------------------------------------------------------

  it('renders the full-quote copy section for a Quote node', () => {
    const art = artifactOf([
      {
        id: 'quote:1',
        type: 'Quote',
        properties: { text: 'The full passage of the quote, uncapped.' },
      },
    ])
    const w = mountDetail({ viewArtifact: art, nodeId: 'quote:1' })
    expect(w.find('[data-testid="node-detail-full-quote"]').exists()).toBe(true)
    expect(w.find('[data-testid="node-detail-full-quote"]').text()).toContain('full passage')
    expect(w.find('[data-testid="node-detail-full-quote-copy"]').exists()).toBe(true)
  })

  it('resolves a bare subject id against a GI/KG-prefixed merged node (#967↔#974)', () => {
    // On a real merged corpus ``mergeGiKg`` prefixes GI ids (``g:``); a search hit / subject
    // carries the BARE id (``quote:1``). Exact-match resolution missed the prefixed node, so
    // the rail showed an empty "Node" with no detail. The rail must resolve through the
    // prefix-tolerant lookup. (Single-artifact fixtures never get the ``g:`` prefix, so the
    // e2e suite couldn't catch this — only a real merged corpus exposed it.)
    const art = artifactOf([
      {
        id: 'g:quote:1',
        type: 'Quote',
        properties: { text: 'The full passage of the quote, uncapped.' },
      },
    ])
    const w = mountDetail({ viewArtifact: art, nodeId: 'quote:1' })
    expect(w.find('aside').exists()).toBe(true)
    expect(w.find('[data-testid="node-detail-kind-row"]').text()).toContain('Quote')
    expect(w.find('[data-testid="node-detail-full-quote"]').text()).toContain('full passage')
  })

  // --- Extra properties + diagnostics ---------------------------------------

  it('lists extra properties (sorted, humanized) excluding hidden keys', () => {
    const art = artifactOf([
      {
        id: 'topic:x',
        type: 'Topic',
        properties: { label: 'X', custom_field: 'hello', zeta_value: 42 },
      },
    ])
    const w = mountDetail({ viewArtifact: art, nodeId: 'topic:x' })
    // The diagnostics HelpTip also renders a <dl>; the extra-props list is the
    // last <dl> in the panel.
    const dls = w.findAll('dl')
    const dl = dls[dls.length - 1]
    const text = dl.text()
    expect(text).toContain('Custom Field')
    expect(text).toContain('hello')
    expect(text).toContain('Zeta Value')
    // Hidden prop "label" is not listed as a generic row.
    expect(text).not.toContain('Label')
  })

  it('falls back to the full graph artifact when the node is missing from the view slice', () => {
    const viewArtifact = artifactOf([])
    const w = mountDetail({ viewArtifact, nodeId: 'ep:1' })
    // Not in view and no full artifact loaded -> panel hidden.
    expect(w.find('aside').exists()).toBe(false)
  })

  // --- RFC-097 v3.0 typed entity branch -------------------------------------
  // Organization is a first-class KG node type. Non-person entities now fold
  // their TopicEntityView overview inline (same as topics); the standalone
  // "Open full Entity profile" button is retired. The connections section
  // (stubbed here, tested separately in GraphConnectionsSection.test.ts) renders
  // typed MENTIONS_PERSON / MENTIONS_ORG rows.

  it('Organization node uses the generic node view (no TopicEntityView fold, no profile button)', async () => {
    const art = artifactOf(
      [
        { id: 'org:acme', type: 'Organization', properties: { name: 'Acme' } },
        {
          id: 'insight:1',
          type: 'Insight',
          properties: {
            text: 'Acme launched a new product.',
            grounded: true,
            insight_type: 'claim',
          },
        },
      ],
      [{ from: 'insight:1', to: 'org:acme', type: 'MENTIONS_ORG' }],
    )
    const w = mountDetail({ viewArtifact: art, nodeId: 'org:acme' })
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    await w.vm.$nextTick()

    expect(w.text()).toContain('Acme')
    // Non-person entities use the generic node view — TopicEntityView is
    // topic-oriented and renders empty for orgs, so it is not folded in; and
    // there is no standalone "Open full profile" hop.
    expect(w.find('[data-testid="topic-entity-view"]').exists()).toBe(false)
    expect(w.find('[data-testid="node-detail-open-person-profile"]').exists()).toBe(false)
  })

  it('Organization node still invokes the GraphConnectionsSection with the right artifact + nodeId', async () => {
    /** The connections panel for an Organization surfaces MENTIONS_ORG
     * inbound edges via the stubbed GraphConnectionsSection. We assert
     * the wiring (props), not the rendering (covered separately in
     * GraphConnectionsSection.test.ts).
     */
    const art = artifactOf(
      [
        { id: 'org:acme', type: 'Organization', properties: { name: 'Acme' } },
        {
          id: 'insight:1',
          type: 'Insight',
          properties: { text: 'About Acme.', grounded: true, insight_type: 'observation' },
        },
      ],
      [{ from: 'insight:1', to: 'org:acme', type: 'MENTIONS_ORG' }],
    )
    const w = mountDetail({ viewArtifact: art, nodeId: 'org:acme' })
    const stub = w.findComponent({ name: 'GraphConnectionsSection' })
    expect(stub.exists()).toBe(true)
    expect(stub.props('nodeId')).toBe('org:acme')
    // Edge type pass-through is asserted in GraphConnectionsSection.test.ts.
  })

  it('Insight node passes typed-MENTIONS-rich artifact to the GraphConnectionsSection', async () => {
    /** Wiring assertion: when an Insight has both MENTIONS_PERSON and
     * MENTIONS_ORG outgoing, the connections section receives the
     * artifact intact. Rendering details (precise edge-type text) are
     * asserted in GraphConnectionsSection.test.ts.
     */
    const art = artifactOf(
      [
        {
          id: 'insight:1',
          type: 'Insight',
          properties: {
            text: 'Ada from Acme launched a product.',
            grounded: true,
            insight_type: 'claim',
            position_hint: 0.42,
          },
        },
        { id: 'person:ada', type: 'Person', properties: { name: 'Ada' } },
        { id: 'org:acme', type: 'Organization', properties: { name: 'Acme' } },
        { id: 'topic:tech', type: 'Topic', properties: { label: 'Tech' } },
        { id: 'quote:1', type: 'Quote', properties: { text: 'Quote.' } },
      ],
      [
        { from: 'insight:1', to: 'topic:tech', type: 'ABOUT' },
        { from: 'insight:1', to: 'quote:1', type: 'SUPPORTED_BY' },
        { from: 'insight:1', to: 'person:ada', type: 'MENTIONS_PERSON' },
        { from: 'insight:1', to: 'org:acme', type: 'MENTIONS_ORG' },
      ],
    )
    const w = mountDetail({ viewArtifact: art, nodeId: 'insight:1' })
    const stub = w.findComponent({ name: 'GraphConnectionsSection' })
    expect(stub.exists()).toBe(true)
    expect(stub.props('nodeId')).toBe('insight:1')

    // insight_type chip / label visible directly in the Insight panel
    // (rendered by NodeDetail itself, not the connections stub).
    const text = w.text()
    expect(text.toLowerCase()).toContain('claim')
  })

  it('Position Tracker tab embeds PersonLandingView with view="positions" (not PositionTrackerPanel)', async () => {
    const art = artifactOf([{ id: 'person:ada', type: 'Person', properties: { name: 'Ada' } }])
    const w = mountDetail({ viewArtifact: art, nodeId: 'person:ada', embedInRail: true })
    await w.get('[data-testid="node-detail-rail-tab-position-tracker"]').trigger('click')
    // PersonLandingView stub is used (exists in DOM from the position tracker tab)
    expect(w.find('#node-detail-rail-panel-position-tracker').exists()).toBe(true)
    // PositionTrackerPanel stub NOT present (it was removed from NodeDetail imports)
    expect(w.find('[data-testid="position-tracker-panel"]').exists()).toBe(false)
  })
})
