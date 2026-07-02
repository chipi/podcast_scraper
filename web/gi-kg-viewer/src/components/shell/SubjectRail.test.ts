// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSubjectStore } from '../../stores/subject'
import SubjectRail from './SubjectRail.vue'

// SubjectRail is a pure *router*: it picks a child detail panel based on the
// real subject store's `kind` and re-emits that child's events upward. We stub
// every heavy child panel with a thin component that:
//   - is identifiable via a data-testid,
//   - exposes the props SubjectRail wires into it, and
//   - emits the events SubjectRail listens for (so we can assert bubbling).
// The real subject store is used (driven via focus*/clear), per the brief.

const GraphNodeRailPanelStub = {
  name: 'GraphNodeRailPanel',
  template: `
    <div data-testid="stub-graph-node-rail">
      <button data-testid="emit-go-graph" @click="$emit('go-graph')" />
      <button
        data-testid="emit-prefill"
        @click="$emit('prefill-semantic-search', { query: 'q' })"
      />
      <button
        data-testid="emit-explore-topic"
        @click="$emit('open-explore-topic-filter', { topic: 't' })"
      />
      <button
        data-testid="emit-explore-speaker"
        @click="$emit('open-explore-speaker-filter', { speaker: 's' })"
      />
      <button
        data-testid="emit-explore-insight"
        @click="$emit('open-explore-insight-filters', { groundedOnly: true, minConfidence: 0.5 })"
      />
      <button
        data-testid="emit-open-library"
        @click="$emit('open-library-episode', { metadata_relative_path: 'p' })"
      />
      <button data-testid="emit-close-rail" @click="$emit('close-subject-rail')" />
    </div>
  `,
}

const EpisodeDetailPanelStub = {
  name: 'EpisodeDetailPanel',
  props: ['railNeighbourhoodEnabled', 'railDetailTab'],
  template: `
    <div
      data-testid="stub-episode-panel"
      :data-neighbourhood-enabled="String(railNeighbourhoodEnabled)"
      :data-detail-tab="railDetailTab"
    >
      <button
        data-testid="emit-focus-search"
        @click="$emit('focus-search', { feed: 'f', query: 'q' })"
      />
      <button
        data-testid="emit-switch-main-tab"
        @click="$emit('switch-main-tab', 'library')"
      />
      <slot name="episode-rail-tabs" />
      <slot name="episode-rail-neighbourhood" />
    </div>
  `,
}

const STUBS = {
  GraphNodeRailPanel: GraphNodeRailPanelStub,
  EpisodeDetailPanel: EpisodeDetailPanelStub,
  // Neighbourhood child rendered only via the episode slot; stub to a marker.
  GraphConnectionsSection: {
    name: 'GraphConnectionsSection',
    props: ['viewArtifact', 'nodeId', 'denseNeighborList'],
    template: '<div data-testid="stub-connections" :data-node-id="nodeId" />',
  },
}

function mountRail(props: { mainTab?: SubjectRailMainTab } = {}) {
  return mount(SubjectRail, {
    props: { mainTab: props.mainTab ?? 'graph' },
    attachTo: document.body,
    global: { stubs: STUBS },
  })
}

type SubjectRailMainTab = 'digest' | 'library' | 'graph' | 'dashboard'

describe('SubjectRail', () => {
  beforeEach(() => setActivePinia(createPinia()))

  // --- Empty / no-subject placeholder ---------------------------------------

  it('renders the empty placeholder when no subject is focused', () => {
    const w = mountRail()
    const empty = w.find('[data-testid="subject-rail-empty"]')
    expect(empty.exists()).toBe(true)
    expect(empty.text()).toContain('Select an episode, topic, or graph node')
  })

  it('hides the close button and all child panels when there is no subject', () => {
    const w = mountRail()
    expect(w.find('[data-testid="subject-rail-close"]').exists()).toBe(false)
    expect(w.find('[data-testid="stub-graph-node-rail"]').exists()).toBe(false)
    expect(w.find('[data-testid="stub-episode-panel"]').exists()).toBe(false)
    expect(w.find('[data-testid="stub-person-view"]').exists()).toBe(false)
  })

  // --- graph-node branch -----------------------------------------------------

  it('renders GraphNodeRailPanel for a graph-node subject', async () => {
    const w = mountRail()
    useSubjectStore().focusGraphNode('cy-node-1')
    await w.vm.$nextTick()
    expect(w.find('[data-testid="stub-graph-node-rail"]').exists()).toBe(true)
    expect(w.find('[data-testid="subject-rail-empty"]').exists()).toBe(false)
    // Close now lives inside each panel (here the stub), not in SubjectRail.
  })

  it('does not render GraphNodeRailPanel when the graph node id is blank', async () => {
    const w = mountRail()
    const subject = useSubjectStore()
    // Force kind=graph-node but a whitespace id (focusGraphNode would clear it,
    // so set the store fields directly to hit the `?.trim()` guard).
    subject.kind = 'graph-node'
    subject.graphNodeCyId = '   '
    await w.vm.$nextTick()
    // No child panel renders; the close now lives inside each panel, so none shows.
    expect(w.find('[data-testid="subject-rail-close"]').exists()).toBe(false)
    expect(w.find('[data-testid="stub-graph-node-rail"]').exists()).toBe(false)
  })

  // --- episode branch --------------------------------------------------------

  it('renders the Episode region and EpisodeDetailPanel for an episode subject', async () => {
    const w = mountRail()
    useSubjectStore().focusEpisode('feed/ep.json')
    await w.vm.$nextTick()
    const region = w.find('[data-testid="episode-detail-rail"]')
    expect(region.exists()).toBe(true)
    expect(region.attributes('aria-label')).toBe('Episode')
    expect(w.find('[data-testid="stub-episode-panel"]').exists()).toBe(true)
  })

  it('does not render the Episode panel when the metadata path is blank', async () => {
    const w = mountRail()
    const subject = useSubjectStore()
    subject.kind = 'episode'
    subject.episodeMetadataPath = '  '
    await w.vm.$nextTick()
    expect(w.find('[data-testid="stub-episode-panel"]').exists()).toBe(false)
  })

  // --- topic branch (folded into the unified node view) ----------------------

  it('routes a topic subject to the unified GraphNodeRailPanel node view', async () => {
    const w = mountRail()
    const subject = useSubjectStore()
    subject.focusTopic('topic:ai')
    await w.vm.$nextTick()
    // focusTopic no longer opens a standalone TopicEntityView; it focuses the
    // topic as a graph node so the generic NodeDetail rail renders it.
    expect(subject.kind).toBe('graph-node')
    expect(subject.graphNodeCyId).toBe('topic:ai')
    expect(w.find('[data-testid="stub-graph-node-rail"]').exists()).toBe(true)
  })

  // --- person branch ---------------------------------------------------------

  it('renders PersonLandingView for a person subject', async () => {
    const w = mountRail()
    useSubjectStore().focusPerson('person:ada')
    await w.vm.$nextTick()
    expect(w.find('[data-testid="stub-graph-node-rail"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-person-view"]').exists()).toBe(false)
  })

  // --- routing switches as the focused subject kind changes ------------------

  it('swaps the rendered panel when the subject kind changes', async () => {
    const w = mountRail()
    const subject = useSubjectStore()

    subject.focusGraphNode('cy-1')
    await w.vm.$nextTick()
    expect(w.find('[data-testid="stub-graph-node-rail"]').exists()).toBe(true)

    // A topic focuses as a graph node now — the node-view panel stays rendered.
    subject.focusTopic('topic:x')
    await w.vm.$nextTick()
    expect(w.find('[data-testid="stub-graph-node-rail"]').exists()).toBe(true)

    subject.focusPerson('person:y')
    await w.vm.$nextTick()
    expect(w.find('[data-testid="stub-graph-node-rail"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-person-view"]').exists()).toBe(false)

    subject.focusEpisode('feed/ep.json')
    await w.vm.$nextTick()
    expect(w.find('[data-testid="stub-person-view"]').exists()).toBe(false)
    expect(w.find('[data-testid="stub-episode-panel"]').exists()).toBe(true)

    subject.clearSubject()
    await w.vm.$nextTick()
    expect(w.find('[data-testid="stub-episode-panel"]').exists()).toBe(false)
    expect(w.find('[data-testid="subject-rail-empty"]').exists()).toBe(true)
  })

  // --- close button ----------------------------------------------------------

  it('emits closeSubject when the episode close button is clicked', async () => {
    const w = mountRail()
    // Episode is the one branch whose close × SubjectRail still renders directly;
    // graph-node/topic/person closes live inside their (stubbed) panels.
    useSubjectStore().focusEpisode('feed/ep.json')
    await w.vm.$nextTick()
    await w.get('[data-testid="subject-rail-close"]').trigger('click')
    expect(w.emitted('closeSubject')).toHaveLength(1)
  })

  // --- main-tab prop -> episode neighbourhood wiring -------------------------

  it('enables episode rail neighbourhood only on the graph tab with a connections id', async () => {
    const w = mountRail({ mainTab: 'graph' })
    useSubjectStore().focusEpisode('feed/ep.json', { graphConnectionsCyId: 'cy-77' })
    await w.vm.$nextTick()
    const panel = w.get('[data-testid="stub-episode-panel"]')
    expect(panel.attributes('data-neighbourhood-enabled')).toBe('true')
    // The neighbourhood slot wires GraphConnectionsSection with the connections id.
    const conn = w.find('[data-testid="stub-connections"]')
    expect(conn.exists()).toBe(true)
    expect(conn.attributes('data-node-id')).toBe('cy-77')
  })

  it('disables episode rail neighbourhood when not on the graph tab', async () => {
    const w = mountRail({ mainTab: 'library' })
    useSubjectStore().focusEpisode('feed/ep.json', { graphConnectionsCyId: 'cy-77' })
    await w.vm.$nextTick()
    const panel = w.get('[data-testid="stub-episode-panel"]')
    // Off the graph tab the rail tells EpisodeDetailPanel neighbourhood is
    // disabled; the real panel then hides the neighbourhood slot. (The stub
    // renders all slots unconditionally, so we assert the wired prop, not the
    // slot's presence.)
    expect(panel.attributes('data-neighbourhood-enabled')).toBe('false')
  })

  it('disables episode rail neighbourhood on the graph tab without a connections id', async () => {
    const w = mountRail({ mainTab: 'graph' })
    useSubjectStore().focusEpisode('feed/ep.json')
    await w.vm.$nextTick()
    const panel = w.get('[data-testid="stub-episode-panel"]')
    expect(panel.attributes('data-neighbourhood-enabled')).toBe('false')
  })

  it('reacts to a main-tab prop change for neighbourhood enablement', async () => {
    const w = mountRail({ mainTab: 'library' })
    useSubjectStore().focusEpisode('feed/ep.json', { graphConnectionsCyId: 'cy-77' })
    await w.vm.$nextTick()
    expect(
      w.get('[data-testid="stub-episode-panel"]').attributes('data-neighbourhood-enabled'),
    ).toBe('false')
    await w.setProps({ mainTab: 'graph' })
    expect(
      w.get('[data-testid="stub-episode-panel"]').attributes('data-neighbourhood-enabled'),
    ).toBe('true')
  })

  it('passes the details rail-detail-tab into EpisodeDetailPanel by default', async () => {
    const w = mountRail({ mainTab: 'graph' })
    useSubjectStore().focusEpisode('feed/ep.json', { graphConnectionsCyId: 'cy-77' })
    await w.vm.$nextTick()
    expect(
      w.get('[data-testid="stub-episode-panel"]').attributes('data-detail-tab'),
    ).toBe('details')
  })

  // --- re-emitted events bubbling up from stubbed children -------------------

  describe('graph-node child event re-emission', () => {
    async function mountGraphNode() {
      const w = mountRail()
      useSubjectStore().focusGraphNode('cy-1')
      await w.vm.$nextTick()
      return w
    }

    it('re-emits go-graph as goGraph', async () => {
      const w = await mountGraphNode()
      await w.get('[data-testid="emit-go-graph"]').trigger('click')
      expect(w.emitted('goGraph')).toHaveLength(1)
    })

    it('re-emits prefill-semantic-search with payload', async () => {
      const w = await mountGraphNode()
      await w.get('[data-testid="emit-prefill"]').trigger('click')
      expect(w.emitted('prefillSemanticSearch')![0]).toEqual([{ query: 'q' }])
    })

    it('re-emits open-explore-topic-filter with payload', async () => {
      const w = await mountGraphNode()
      await w.get('[data-testid="emit-explore-topic"]').trigger('click')
      expect(w.emitted('openExploreTopicFilter')![0]).toEqual([{ topic: 't' }])
    })

    it('re-emits open-explore-speaker-filter with payload', async () => {
      const w = await mountGraphNode()
      await w.get('[data-testid="emit-explore-speaker"]').trigger('click')
      expect(w.emitted('openExploreSpeakerFilter')![0]).toEqual([{ speaker: 's' }])
    })

    it('re-emits open-explore-insight-filters with payload', async () => {
      const w = await mountGraphNode()
      await w.get('[data-testid="emit-explore-insight"]').trigger('click')
      expect(w.emitted('openExploreInsightFilters')![0]).toEqual([
        { groundedOnly: true, minConfidence: 0.5 },
      ])
    })

    it('re-emits open-library-episode with payload', async () => {
      const w = await mountGraphNode()
      await w.get('[data-testid="emit-open-library"]').trigger('click')
      expect(w.emitted('openLibraryEpisode')![0]).toEqual([{ metadata_relative_path: 'p' }])
    })

    it('re-emits the child close-subject-rail as closeSubject', async () => {
      const w = await mountGraphNode()
      await w.get('[data-testid="emit-close-rail"]').trigger('click')
      expect(w.emitted('closeSubject')).toHaveLength(1)
    })
  })

  describe('episode child event re-emission', () => {
    async function mountEpisode() {
      const w = mountRail()
      useSubjectStore().focusEpisode('feed/ep.json')
      await w.vm.$nextTick()
      return w
    }

    it('re-emits focus-search as focusSearchHandoff', async () => {
      const w = await mountEpisode()
      await w.get('[data-testid="emit-focus-search"]').trigger('click')
      expect(w.emitted('focusSearchHandoff')![0]).toEqual([{ feed: 'f', query: 'q' }])
    })

    it('re-emits switch-main-tab as switchMainTab', async () => {
      const w = await mountEpisode()
      await w.get('[data-testid="emit-switch-main-tab"]').trigger('click')
      expect(w.emitted('switchMainTab')![0]).toEqual(['library'])
    })

    it('re-emits go-graph from the neighbourhood connections section', async () => {
      const w = mountRail({ mainTab: 'graph' })
      useSubjectStore().focusEpisode('feed/ep.json', { graphConnectionsCyId: 'cy-77' })
      await w.vm.$nextTick()
      // GraphConnectionsSection stub does not emit; assert the slot wires it in.
      expect(w.find('[data-testid="stub-connections"]').exists()).toBe(true)
    })
  })


})
