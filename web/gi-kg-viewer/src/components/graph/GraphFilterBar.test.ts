// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { useArtifactsStore } from '../../stores/artifacts'
import type { GraphFilterState, ParsedArtifact } from '../../types/artifact'
import GraphFilterBar from './GraphFilterBar.vue'

const BAR = '[data-testid="graph-filter-bar"]'
const RESET = '[data-testid="graph-chip-reset-all"]'

/** Minimal filter state matching the displayArtifact watcher output shape. */
function makeState(
  overrides: Partial<GraphFilterState> = {},
): GraphFilterState {
  return {
    allowedTypes: {},
    allowedEdgeTypes: {},
    hideUngroundedInsights: false,
    showGiLayer: true,
    showKgLayer: true,
    graphFeedFilterId: null,
    ...overrides,
  }
}

/** Minimal parsed GI artifact so artifacts.displayArtifact is non-null. */
function makeGiArtifact(): ParsedArtifact {
  return {
    name: 'ep.gi.json',
    kind: 'gi',
    episodeId: 'ep1',
    nodes: 0,
    edges: 0,
    nodeTypes: {},
    data: { nodes: [], edges: [] },
  }
}

/**
 * Stub the five child chips so the bar renders standalone. Each stub emits a
 * predictable testid and surfaces its bound props as JSON for assertions.
 */
const chipStub = (name: string) => ({
  name,
  props: ['typeHistogramCounts', 'degreeHistogramCounts'],
  template: `<div data-testid="stub-${name}" :data-props="JSON.stringify($props)" />`,
})

const STUBS = {
  GraphTypesChip: chipStub('types'),
  GraphFeedChip: chipStub('feed'),
  GraphSourcesChip: chipStub('sources'),
  GraphEdgesChip: chipStub('edges'),
  GraphDegreeChip: chipStub('degree'),
}

const PROPS = {
  typeHistogramCounts: { Speaker: 3, Quote: 7 },
  degreeHistogramCounts: { '1-2': 4, '2-5': 6 },
}

const mountBar = (props = PROPS) =>
  mount(GraphFilterBar, {
    props,
    attachTo: document.body,
    global: { stubs: STUBS },
  })

describe('GraphFilterBar', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('renders nothing until the filter store has state', () => {
    const w = mountBar()
    expect(w.find(BAR).exists()).toBe(false)
  })

  it('renders the bar once filter state is present', () => {
    const gf = useGraphFilterStore()
    gf.state = makeState()
    const w = mountBar()
    expect(w.get(BAR).exists()).toBe(true)
  })

  it('renders all five chips plus the divider', () => {
    const gf = useGraphFilterStore()
    gf.state = makeState()
    const w = mountBar()
    for (const id of ['types', 'feed', 'sources', 'edges', 'degree']) {
      expect(w.find(`[data-testid="stub-${id}"]`).exists()).toBe(true)
    }
    // The decorative divider between filter and view chips.
    expect(w.get(BAR).find('span[aria-hidden="true"]').exists()).toBe(true)
  })

  it('passes typeHistogramCounts down to the Types chip', () => {
    const gf = useGraphFilterStore()
    gf.state = makeState()
    const w = mountBar()
    const raw = w.get('[data-testid="stub-types"]').attributes('data-props')
    expect(JSON.parse(raw!).typeHistogramCounts).toEqual(PROPS.typeHistogramCounts)
  })

  it('passes degreeHistogramCounts down to the Degree chip', () => {
    const gf = useGraphFilterStore()
    gf.state = makeState()
    const w = mountBar()
    const raw = w.get('[data-testid="stub-degree"]').attributes('data-props')
    expect(JSON.parse(raw!).degreeHistogramCounts).toEqual(PROPS.degreeHistogramCounts)
  })

  it('hides the reset-all button when no filter dimension is active', () => {
    const gf = useGraphFilterStore()
    gf.state = makeState()
    const w = mountBar()
    expect(w.find(RESET).exists()).toBe(false)
  })

  it('shows reset-all when a degree bucket is active (explorer-store branch)', async () => {
    const gf = useGraphFilterStore()
    gf.state = makeState()
    const ge = useGraphExplorerStore()
    const w = mountBar()
    expect(w.find(RESET).exists()).toBe(false)

    ge.toggleDegreeBucket('2-5')
    await w.vm.$nextTick()
    expect(w.find(RESET).exists()).toBe(true)
  })

  it('shows reset-all when graph filters are active (filter-store branch)', async () => {
    // filtersAreActive requires a non-null displayArtifact, so seed one.
    const artifacts = useArtifactsStore()
    artifacts.parsedList = [makeGiArtifact()]
    const gf = useGraphFilterStore()
    // The displayArtifact watcher populated gf.state; mark a filter active.
    gf.state!.hideUngroundedInsights = true
    const w = mountBar()
    await w.vm.$nextTick()
    expect(gf.filtersAreActive).toBe(true)
    expect(w.find(RESET).exists()).toBe(true)
  })

  it('reset-all clears the active degree bucket and active filters', async () => {
    const artifacts = useArtifactsStore()
    artifacts.parsedList = [makeGiArtifact()]
    const gf = useGraphFilterStore()
    gf.state!.hideUngroundedInsights = true
    const ge = useGraphExplorerStore()
    ge.toggleDegreeBucket('1-2')
    const w = mountBar()
    await w.vm.$nextTick()
    expect(w.find(RESET).exists()).toBe(true)

    await w.get(RESET).trigger('click')

    expect(ge.activeDegreeBucket).toBeNull()
    expect(gf.state!.hideUngroundedInsights).toBe(false)
    expect(gf.filtersAreActive).toBe(false)
    // Button disappears once everything is reset.
    expect(w.find(RESET).exists()).toBe(false)
  })

  it('exposes an accessible label on the reset button', () => {
    const gf = useGraphFilterStore()
    gf.state = makeState()
    const ge = useGraphExplorerStore()
    ge.toggleDegreeBucket('2-5')
    const w = mountBar()
    const btn = w.get(RESET)
    expect(btn.attributes('aria-label')).toBe('Reset all graph filters')
    expect(btn.attributes('type')).toBe('button')
  })
})
