// @vitest-environment happy-dom
import { mount, type VueWrapper } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import type { Core, NodeSingular } from 'cytoscape'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import cytoscape from 'cytoscape'
import { defineComponent, h, nextTick } from 'vue'

// Cytoscape runs for real, headless (no canvas in happy-dom): graph building, classes,
// neighbourhoods, betweenness and the preset/grid/fcose layouts all execute. Only the pieces
// that need a renderer are faked: `animate` completes synchronously and `navigator` is a spy.
const cyKit = vi.hoisted(() => ({
  cores: [] as Core[],
  animations: [] as Array<Record<string, unknown>>,
  navigatorDestroy: vi.fn(),
  navigatorThrows: { value: false },
  navigatorCalls: [] as Array<Record<string, unknown>>,
}))

vi.mock('cytoscape', async (importOriginal) => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const real = ((await importOriginal()) as { default: any }).default
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wrapped: any = (...args: any[]) => {
    if (typeof args[0] === 'string') return real(...args)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const core: any = real({ ...args[0], container: undefined, headless: true, styleEnabled: true })
    // A headless core has a 0x0 viewport, which makes fit() collapse the zoom to its minimum.
    core.width = () => 800
    core.height = () => 600
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    core.animate = vi.fn((opts: any) => {
      cyKit.animations.push(opts)
      if (typeof opts?.zoom === 'number') core.zoom(opts.zoom)
      opts?.complete?.()
      return core
    })
    core.navigator = vi.fn((opts: Record<string, unknown>) => {
      cyKit.navigatorCalls.push(opts)
      if (cyKit.navigatorThrows.value) throw new Error('navigator boom')
      return { destroy: cyKit.navigatorDestroy }
    })
    cyKit.cores.push(core)
    return core
  }
  wrapped.use = real.use.bind(real)
  return { default: wrapped }
})

vi.mock('../../api/corpusLibraryApi', async (importOriginal) => ({
  ...((await importOriginal()) as Record<string, unknown>),
  fetchNodeEpisodes: vi.fn(),
  fetchCorpusEpisodeDetail: vi.fn(),
  fetchCorpusEpisodes: vi.fn(),
}))

vi.mock('../../composables/useEnrichmentEnvelopeCache', () => ({
  fetchCachedCorpusEnvelope: vi.fn(),
}))

import {
  fetchCorpusEpisodeDetail,
  fetchCorpusEpisodes,
  fetchNodeEpisodes,
} from '../../api/corpusLibraryApi'
import { fetchCachedCorpusEnvelope } from '../../composables/useEnrichmentEnvelopeCache'
import {
  graphCanvasPlugins,
  graphLenses as registeredGraphLenses,
  type GraphCanvasPlugin,
} from '../../extensions'
import { useActiveSearchContextStore } from '../../stores/activeSearchContext'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphAnalyticsStore } from '../../stores/graphAnalytics'
import { useGraphExpansionStore } from '../../stores/graphExpansion'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { useGraphHandoffStore } from '../../stores/graphHandoff'
import { useGraphLensesStore } from '../../stores/graphLenses'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useSearchStore } from '../../stores/search'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { useThemeStore } from '../../stores/theme'
import type { ArtifactData } from '../../types/artifact'
import { parseArtifact } from '../../utils/parsing'
import GraphCanvas from './GraphCanvas.vue'

// happy-dom's Storage is unreliable here; same Map-backed stub as GraphBottomBar.test.ts.
const storage = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (k: string) => storage.get(k) ?? null,
  setItem: (k: string, v: string) => storage.set(k, v),
  removeItem: (k: string) => storage.delete(k),
})

// A layout that starts and never finishes, to exercise the "layout in flight" paths.
const hangKit = { started: 0, stopped: 0 }
// Cytoscape instantiates layouts ES5-style, so this cannot be a class.
function HangLayout(this: { options: unknown }, options: unknown) {
  this.options = options
}
HangLayout.prototype.run = function run() {
  hangKit.started += 1
  return this
}
HangLayout.prototype.stop = function stop() {
  hangKit.stopped += 1
  return this
}
cytoscape('layout', 'hang', HangLayout as never)

const reopenGestures = vi.fn()

const BottomBarStub = defineComponent({
  name: 'GraphBottomBar',
  props: ['showLensControls', 'showGesturesInBottomBar', 'zoomPercent', 'searchHighlightCount', 'preferredLayout'],
  emits: [
    'fit',
    'zoom-in',
    'zoom-out',
    'zoom-reset',
    'export-png',
    'reopen-gestures',
    'relayout',
    'cycle-layout',
    'request-corpus-graph-sync',
    'request-graph-full-reset',
  ],
  setup: () => () => h('div', { 'data-testid': 'bottom-bar-stub' }),
})

const FilterBarStub = defineComponent({
  name: 'GraphFilterBar',
  props: ['typeHistogramCounts', 'degreeHistogramCounts'],
  setup: () => () => h('div', { 'data-testid': 'filter-bar-stub' }),
})

const StatusLineStub = defineComponent({
  name: 'GraphStatusLine',
  props: ['variant', 'bare'],
  setup: () => () => h('div', { 'data-testid': 'status-line-stub' }),
})

const GestureOverlayStub = defineComponent({
  name: 'GraphGestureOverlay',
  props: ['hasNodes'],
  emits: ['dismissed'],
  setup(_p, { expose }) {
    expose({ reopen: reopenGestures })
    return () => h('div', { 'data-testid': 'gesture-overlay-stub' })
  },
})

const STUBS = {
  GraphBottomBar: BottomBarStub,
  GraphFilterBar: FilterBarStub,
  GraphStatusLine: StatusLineStub,
  GraphGestureOverlay: GestureOverlayStub,
}

const EP1_META = 'feeds/a/metadata/ep1.metadata.json'
const EP2_META = 'feeds/a/metadata/ep2.metadata.json'

/** Two episodes sharing topic:ai (degree 2 -> expandable); person:ada hangs off ep1 only. */
function giData(overrides: Partial<ArtifactData> = {}): ArtifactData {
  return {
    model_version: 'm',
    prompt_version: 'p',
    episode_id: 'ep-1',
    nodes: [
      {
        id: 'episode:ep-1',
        type: 'Episode',
        properties: {
          title: 'Episode One',
          episode_id: 'ep-1',
          metadata_relative_path: EP1_META,
          publish_date: '2024-03-01',
        },
      },
      {
        id: 'episode:ep-2',
        type: 'Episode',
        properties: {
          title: 'Episode Two',
          episode_id: 'ep-2',
          metadata_relative_path: EP2_META,
          publish_date: '2024-06-01',
        },
      },
      { id: 'topic:ai', type: 'Topic', properties: { label: 'AI' } },
      { id: 'topic:solo', type: 'Topic', properties: { label: 'Solo' } },
      { id: 'person:ada', type: 'Person', properties: { name: 'Ada' } },
    ],
    edges: [
      { from: 'episode:ep-1', to: 'topic:ai', type: 'ABOUT' },
      { from: 'episode:ep-2', to: 'topic:ai', type: 'ABOUT' },
      { from: 'episode:ep-1', to: 'topic:solo', type: 'ABOUT' },
      { from: 'episode:ep-1', to: 'person:ada', type: 'SPOKE_IN' },
    ],
    ...overrides,
  } as ArtifactData
}

function loadGi(data: ArtifactData = giData(), name = 'ep1.gi.json'): void {
  const artifacts = useArtifactsStore()
  artifacts.parsedList = [parseArtifact(name, data)]
}

/** Advance fake time and flush Vue's queues (flushPromises would hang under fake timers). */
async function tick(ms = 0): Promise<void> {
  await nextTick()
  await vi.advanceTimersByTimeAsync(ms)
  await nextTick()
}

/** Past the post-layout redraw cooldown (300ms) and the paint-hold animation frames. */
const SETTLE_MS = 500

function liveCore(): Core {
  const c = [...cyKit.cores].reverse().find((x) => !x.destroyed())
  if (!c) throw new Error('no live cytoscape core')
  return c
}

async function mountCanvas(
  data: ArtifactData | null = giData(),
  opts: { settle?: boolean } = {},
): Promise<VueWrapper> {
  if (data) loadGi(data)
  const w = mount(GraphCanvas, { attachTo: document.body, global: { stubs: STUBS } })
  if (opts.settle !== false) await tick(SETTLE_MS)
  return w
}

function barVm(w: VueWrapper) {
  return w.findComponent(BottomBarStub)
}

function setLayout(name: 'fcose' | 'breadthfirst' | 'circle' | 'grid' | 'timeline'): void {
  useGraphExplorerStore().preferredLayout = name
}

class FakeResizeObserver {
  static instances: FakeResizeObserver[] = []
  observe = vi.fn()
  disconnect = vi.fn()
  cb: () => void
  constructor(cb: () => void) {
    this.cb = cb
    FakeResizeObserver.instances.push(this)
  }
}
const realResizeObserver = globalThis.ResizeObserver

let mounted: VueWrapper[] = []
async function mountTracked(...args: Parameters<typeof mountCanvas>): Promise<VueWrapper> {
  const w = await mountCanvas(...args)
  mounted.push(w)
  return w
}

beforeEach(() => {
  vi.useFakeTimers()
  FakeResizeObserver.instances = []
  globalThis.ResizeObserver = FakeResizeObserver as never
  setActivePinia(createPinia())
  storage.clear()
  cyKit.cores.length = 0
  cyKit.animations.length = 0
  cyKit.navigatorCalls.length = 0
  cyKit.navigatorThrows.value = false
  cyKit.navigatorDestroy.mockClear()
  hangKit.started = 0
  hangKit.stopped = 0
  reopenGestures.mockClear()
  vi.mocked(fetchNodeEpisodes).mockReset()
  vi.mocked(fetchCorpusEpisodeDetail).mockReset()
  vi.mocked(fetchCorpusEpisodes).mockReset()
  vi.mocked(fetchCachedCorpusEnvelope).mockReset()
  vi.mocked(fetchCachedCorpusEnvelope).mockResolvedValue(null)
  graphCanvasPlugins.reset()
  registeredGraphLenses.reset()
  // happy-dom reports no padding, which the canvas reads as NaN ("cannot lay out yet").
  vi.spyOn(window, 'getComputedStyle').mockImplementation(
    () =>
      ({
        getPropertyValue: (n: string) =>
          n.startsWith('padding') ? '0px' : n === '--ps-graph-canvas' ? ' #123456 ' : '',
      }) as unknown as CSSStyleDeclaration,
  )
  vi.spyOn(console, 'log').mockImplementation(() => {})
  vi.spyOn(console, 'warn').mockImplementation(() => {})
  // Changing the corpus path re-probes /api/health; keep that answer healthy.
  vi.spyOn(globalThis, 'fetch').mockImplementation(
    async () =>
      new Response(JSON.stringify({ status: 'ok', corpus_library_api: true }), { status: 200 }),
  )
})

afterEach(() => {
  for (const w of mounted) w.unmount()
  mounted = []
  globalThis.ResizeObserver = realResizeObserver
  vi.useRealTimers()
  vi.restoreAllMocks()
  graphCanvasPlugins.reset()
  registeredGraphLenses.reset()
})

describe('GraphCanvas rendering', () => {
  it('builds one cytoscape graph holding the visible nodes and edges', async () => {
    await mountTracked()
    expect(cyKit.cores).toHaveLength(1)
    const core = liveCore()
    expect(core.nodes().map((n) => n.id()).sort()).toEqual([
      'episode:ep-1',
      'episode:ep-2',
      'person:ada',
      'topic:ai',
      'topic:solo',
    ])
    expect(core.edges().length).toBeGreaterThanOrEqual(4)
  })

  it('omits node types the graph hides by default (Quote)', async () => {
    const data = giData()
    data.nodes = [...(data.nodes ?? []), { id: 'quote:q1', type: 'Quote', properties: { text: 'hi' } }]
    await mountTracked(data)
    expect(liveCore().$id('quote:q1').empty()).toBe(true)
  })

  it('shows the load-artifacts prompt and builds no graph before anything is loaded', async () => {
    const w = await mountTracked(null)
    expect(cyKit.cores).toHaveLength(0)
    expect(w.text()).toContain('Load artifacts')
    expect(w.find('[data-testid="status-line-stub"]').exists()).toBe(false)
    expect(w.find('[data-testid="bottom-bar-stub"]').exists()).toBe(false)
  })

  it('shows the empty-view message when every node type is filtered out', async () => {
    const data = giData()
    data.nodes = [{ id: 'quote:q1', type: 'Quote', properties: { text: 'hi' } }]
    data.edges = []
    const w = await mountTracked(data)
    expect(cyKit.cores).toHaveLength(0)
    expect(w.text()).toContain('No nodes in this view')
  })

  it('renders the summary strip, filter bar and bottom bar once artifacts are loaded', async () => {
    const w = await mountTracked()
    expect(w.find('[data-testid="status-line-stub"]').exists()).toBe(true)
    expect(w.find('[data-testid="filter-bar-stub"]').exists()).toBe(true)
    const bar = barVm(w)
    expect(bar.exists()).toBe(true)
    expect(bar.props('showLensControls')).toBe(true)
    expect(bar.props('showGesturesInBottomBar')).toBe(false)
    expect(bar.props('preferredLayout')).toBe('fcose')
  })

  it('feeds the filter bar a type histogram and a degree histogram', async () => {
    const w = await mountTracked()
    const fb = w.findComponent(FilterBarStub)
    expect(fb.props('typeHistogramCounts')).toMatchObject({ Episode: 2, Topic: 2, Entity_person: 1 })
    const deg = fb.props('degreeHistogramCounts') as Record<string, number>
    // ada(1), solo(1), ep2(1) -> bucket "1"; ep1(3), ai(2) -> bucket "2-5"
    expect(deg['1']).toBe(3)
    expect(deg['2-5']).toBe(2)
  })

  it('completes a layout pass: invariant clean, FSM quiescent, redraw tracked', async () => {
    const analytics = useGraphAnalyticsStore()
    const track = vi.spyOn(analytics, 'track')
    await mountTracked()
    const gh = useGraphHandoffStore()
    expect(gh.lastInvariant).toMatchObject({ missing: [], extra: [] })
    expect(gh.state).toBe('idle')
    expect(track).toHaveBeenCalledWith('graph_redraw', expect.objectContaining({ nodes: 5 }))
  })

  it('has no loading overlay once the first layout has finished', async () => {
    const w = await mountTracked()
    expect(w.find('[data-testid="graph-layout-loading"]').exists()).toBe(false)
    expect(w.find('[aria-busy]').attributes('aria-busy')).toBe('false')
  })

  it('writes topic degree heat onto hub nodes after layout', async () => {
    await mountTracked()
    const core = liveCore()
    expect(core.$id('topic:ai').data('degreeHeat')).toBeCloseTo(2 / 30)
    expect(core.$id('episode:ep-1').data('degreeHeat')).toBeCloseTo(3 / 30)
  })

  it.each(['breadthfirst', 'circle', 'grid'] as const)('lays out with the %s layout', async (name) => {
    setLayout(name)
    await mountTracked()
    expect(liveCore().nodes()).toHaveLength(5)
    expect(useGraphHandoffStore().lastInvariant).not.toBeNull()
  })

  it('places episodes on a date axis with the timeline layout and counts undated ones', async () => {
    const data = giData()
    data.nodes = [
      ...(data.nodes ?? []),
      { id: 'episode:ep-3', type: 'Episode', properties: { title: 'Undated', episode_id: 'ep-3' } },
      { id: 'insight:i1', type: 'Insight', properties: { text: 'x', episode_id: 'ep-1' } },
    ]
    data.edges = [...(data.edges ?? []), { from: 'episode:ep-1', to: 'insight:i1', type: 'HAS_INSIGHT' }]
    setLayout('timeline')
    const w = await mountTracked(data)
    const core = liveCore()
    const x1 = core.$id('episode:ep-1').position('x')
    const x2 = core.$id('episode:ep-2').position('x')
    expect(x1).toBeLessThan(x2)
    expect(w.get('[data-testid="graph-timeline-missing-date"]').text()).toContain('1 episode has')
  })

  it('timeline falls back to a grid when the graph has no episodes', async () => {
    const data = giData()
    data.nodes = [{ id: 'topic:ai', type: 'Topic', properties: { label: 'AI' } }]
    data.edges = []
    setLayout('timeline')
    const w = await mountTracked(data)
    expect(liveCore().nodes()).toHaveLength(1)
    expect(w.find('[data-testid="graph-timeline-missing-date"]').exists()).toBe(false)
  })

  it('bands plugin-declared topic-like types with Topic on the timeline layout', async () => {
    graphCanvasPlugins.register({ id: 'topicish', topicLikeTypes: ['Theme'] })
    const data = giData()
    data.nodes = [...(data.nodes ?? []), { id: 'theme:1', type: 'Theme', properties: { label: 'T' } }]
    data.edges = [...(data.edges ?? []), { from: 'episode:ep-1', to: 'theme:1', type: 'ABOUT' }]
    setLayout('timeline')
    await mountTracked(data)
    const core = liveCore()
    const ep1x = core.$id('episode:ep-1').position('x')
    expect(core.$id('theme:1').position('x')).toBeCloseTo(ep1x, 0)
  })

  it('draws aggregated episode-topic edges only when the lens is on', async () => {
    const data = giData()
    data.nodes = [...(data.nodes ?? []), { id: 'insight:i1', type: 'Insight', properties: { text: 't' } }]
    data.edges = [
      ...(data.edges ?? []),
      { from: 'episode:ep-1', to: 'insight:i1', type: 'HAS_INSIGHT' },
      { from: 'insight:i1', to: 'topic:ai', type: 'ABOUT' },
    ]
    const lenses = useGraphLensesStore()
    lenses.setAggregatedEdges(true)
    await mountTracked(data)
    expect(liveCore().edges('[edgeType = "ABOUT_AGG"]').length).toBe(1)
    mounted.pop()!.unmount()
    cyKit.cores.length = 0
    lenses.setAggregatedEdges(false)
    await mountTracked(data)
    expect(liveCore().edges('[edgeType = "ABOUT_AGG"]').length).toBe(0)
  })
})

function cyNode(id: string): NodeSingular {
  return liveCore().$id(id) as NodeSingular
}

function emitOn(id: string, type: string, extra: Record<string, unknown> = {}): void {
  cyNode(id).emit({ type, ...extra } as never)
}

function classesOf(id: string): string[] {
  return cyNode(id).classes()
}

describe('GraphCanvas selection and hover dimming', () => {
  it('dims everything outside the neighbourhood of a selected node', async () => {
    await mountTracked()
    cyNode('topic:ai').select()
    expect(classesOf('topic:ai')).toContain('graph-focused')
    expect(classesOf('episode:ep-1')).toContain('graph-neighbour')
    expect(classesOf('episode:ep-2')).toContain('graph-neighbour')
    expect(classesOf('person:ada')).toContain('graph-dimmed')
    expect(classesOf('topic:solo')).toContain('graph-dimmed')
  })

  it('keeps sibling episodes dimmed when an Episode is the focus', async () => {
    await mountTracked()
    // ep-1 and ep-2 are both adjacent to topic:ai but not to each other; add a direct link.
    const core = liveCore()
    core.add({ group: 'edges', data: { id: 'ee', source: 'episode:ep-1', target: 'episode:ep-2' } })
    cyNode('episode:ep-1').select()
    expect(classesOf('episode:ep-1')).toContain('graph-focused')
    expect(classesOf('episode:ep-2')).toContain('graph-dimmed')
    expect(classesOf('episode:ep-2')).not.toContain('graph-neighbour')
    expect(classesOf('topic:ai')).toContain('graph-neighbour')
  })

  it('clears the dimming when the selection is removed', async () => {
    await mountTracked()
    cyNode('topic:ai').select()
    cyNode('topic:ai').unselect()
    for (const n of liveCore().nodes()) {
      expect(n.hasClass('graph-dimmed')).toBe(false)
      expect(n.hasClass('graph-focused')).toBe(false)
    }
  })

  it('re-dims around the remaining selection when one of two selected nodes is released', async () => {
    await mountTracked()
    cyNode('topic:ai').select()
    cyNode('person:ada').select()
    cyNode('topic:ai').unselect()
    expect(classesOf('person:ada')).toContain('graph-focused')
  })

  it('previews the neighbourhood on hover and restores on mouse-out', async () => {
    await mountTracked()
    emitOn('topic:ai', 'mouseover')
    expect(classesOf('topic:ai')).toContain('graph-focused')
    expect(classesOf('person:ada')).toContain('graph-dimmed')
    emitOn('topic:ai', 'mouseout')
    expect(classesOf('person:ada')).not.toContain('graph-dimmed')
  })

  it('does not let hover override an existing selection', async () => {
    await mountTracked()
    cyNode('person:ada').select()
    emitOn('topic:ai', 'mouseover')
    expect(classesOf('topic:ai')).not.toContain('graph-focused')
    emitOn('topic:ai', 'mouseout')
    expect(classesOf('person:ada')).toContain('graph-focused')
  })

  it('gives a bridge node a tooltip on hover and removes it elsewhere', async () => {
    const w = await mountTracked()
    const host = w.get('.graph-canvas').element as HTMLElement
    cyNode('topic:ai').addClass('graph-bridge')
    emitOn('topic:ai', 'mouseover')
    expect(host.getAttribute('title')).toBe('Bridge node — connects distinct neighbourhoods')
    emitOn('topic:ai', 'mouseout')
    expect(host.hasAttribute('title')).toBe(false)
    emitOn('topic:ai', 'mouseover')
    emitOn('topic:solo', 'mouseover')
    expect(host.hasAttribute('title')).toBe(false)
  })

  it('lets a plugin supply the bridge tooltip text', async () => {
    graphCanvasPlugins.register({ id: 'tip', bridgeTitle: () => 'custom tip' })
    const w = await mountTracked()
    cyNode('topic:ai').addClass('graph-bridge')
    emitOn('topic:ai', 'mouseover')
    expect((w.get('.graph-canvas').element as HTMLElement).getAttribute('title')).toBe('custom tip')
  })

  it('flags a topic that bridges two neighbourhoods once the debounced betweenness pass runs', async () => {
    const data = giData()
    data.nodes = [
      { id: 'episode:ep-1', type: 'Episode', properties: { episode_id: 'ep-1' } },
      { id: 'episode:ep-2', type: 'Episode', properties: { episode_id: 'ep-2' } },
      { id: 'topic:bridge', type: 'Topic', properties: { label: 'Bridge' } },
    ]
    data.edges = [
      { from: 'episode:ep-1', to: 'topic:bridge', type: 'ABOUT' },
      { from: 'episode:ep-2', to: 'topic:bridge', type: 'ABOUT' },
    ]
    await mountTracked(data)
    expect(classesOf('topic:bridge')).toContain('graph-bridge')
  })

  it('removes bridge flags when the bridge-ring lens is switched off, and restores them when on', async () => {
    const data = giData()
    data.nodes = [
      { id: 'episode:ep-1', type: 'Episode', properties: { episode_id: 'ep-1' } },
      { id: 'episode:ep-2', type: 'Episode', properties: { episode_id: 'ep-2' } },
      { id: 'topic:bridge', type: 'Topic', properties: { label: 'Bridge' } },
    ]
    data.edges = [
      { from: 'episode:ep-1', to: 'topic:bridge', type: 'ABOUT' },
      { from: 'episode:ep-2', to: 'topic:bridge', type: 'ABOUT' },
    ]
    await mountTracked(data)
    const lenses = useGraphLensesStore()
    lenses.setBridgeRing(false)
    await tick()
    expect(classesOf('topic:bridge')).not.toContain('graph-bridge')
    lenses.setBridgeRing(true)
    await tick(400)
    expect(classesOf('topic:bridge')).toContain('graph-bridge')
  })
})

describe('GraphCanvas taps and gestures', () => {
  it('selects a tapped node and records a node-tap analytics event', async () => {
    const analytics = useGraphAnalyticsStore()
    const track = vi.spyOn(analytics, 'track')
    await mountTracked()
    emitOn('topic:ai', 'tap')
    expect(cyNode('topic:ai').selected()).toBe(true)
    expect(track).toHaveBeenCalledWith('graph_node_tap', { id: 'topic:ai', kind: 'Topic' })
  })

  it('replaces the selection when a different node is tapped', async () => {
    await mountTracked()
    emitOn('topic:ai', 'tap')
    emitOn('person:ada', 'tap')
    expect(cyNode('topic:ai').selected()).toBe(false)
    expect(cyNode('person:ada').selected()).toBe(true)
  })

  it('lets a plugin take over a node tap', async () => {
    const onNodeTap = vi.fn(() => true)
    graphCanvasPlugins.register({ id: 'tap', onNodeTap })
    await mountTracked()
    emitOn('topic:ai', 'tap')
    expect(onNodeTap).toHaveBeenCalledTimes(1)
    expect(cyNode('topic:ai').selected()).toBe(false)
  })

  it('clears selection, subject and notifies plugins on an empty-canvas tap', async () => {
    const onEmptyTap = vi.fn()
    graphCanvasPlugins.register({ id: 'empty', onEmptyTap })
    await mountTracked()
    const subject = useSubjectStore()
    subject.focusGraphNode('topic:ai')
    emitOn('topic:ai', 'tap')
    liveCore().emit('tap')
    expect(cyNode('topic:ai').selected()).toBe(false)
    expect(subject.kind).toBeNull()
    expect(onEmptyTap).toHaveBeenCalledTimes(1)
  })

  it('clears the subject when an edge is tapped', async () => {
    await mountTracked()
    const subject = useSubjectStore()
    subject.focusGraphNode('topic:ai')
    liveCore().edges().first().emit('tap')
    expect(subject.kind).toBeNull()
  })

  it('opens the node rail and centres the camera on a single tap of a topic', async () => {
    await mountTracked()
    const subject = useSubjectStore()
    cyKit.animations.length = 0
    emitOn('topic:ai', 'onetap')
    await tick()
    expect(subject.kind).toBe('graph-node')
    expect(subject.graphNodeCyId).toBe('topic:ai')
    expect(cyKit.animations).toHaveLength(1)
    expect(cyKit.animations[0]).toMatchObject({ duration: 320 })
    expect(cyKit.animations[0].zoom as number).toBeGreaterThanOrEqual(1.3)
    expect(useGraphHandoffStore().state).not.toBe('idle')
  })

  it('opens the episode rail from the loaded metadata when an episode is single-tapped', async () => {
    await mountTracked()
    const subject = useSubjectStore()
    emitOn('episode:ep-1', 'onetap')
    await tick()
    expect(subject.kind).toBe('episode')
    expect(subject.episodeMetadataPath).toBe(EP1_META)
    expect(subject.episodeId).toBe('ep-1')
    expect(subject.graphConnectionsCyId).toBe('episode:ep-1')
  })

  it('falls back to the graph-node rail for an episode with no resolvable metadata path', async () => {
    const data = giData()
    data.nodes = (data.nodes ?? []).map((n) =>
      n.id === 'episode:ep-1' ? { ...n, properties: { title: 'x' } } : n,
    )
    await mountTracked(data)
    const subject = useSubjectStore()
    emitOn('episode:ep-1', 'onetap')
    await tick()
    expect(subject.kind).toBe('graph-node')
    expect(subject.graphNodeCyId).toBe('episode:ep-1')
  })

  describe('episode path lookup in the corpus catalogue', () => {
    const noPath = () => {
      const data = giData()
      data.nodes = (data.nodes ?? []).map((n) =>
        n.id === 'episode:ep-1' ? { ...n, properties: { title: 'x' } } : n,
      )
      return data
    }

    beforeEach(() => {
      storage.set('ps_corpus_path', '/corpus')
    })

    it('opens the episode rail with the path the catalogue reports', async () => {
      vi.mocked(fetchCorpusEpisodes).mockResolvedValue({
        items: [{ episode_id: 'ep-1', metadata_relative_path: ' feeds/a/metadata/found.metadata.json ' }],
        next_cursor: null,
      } as never)
      await mountTracked(noPath())
      useShellStore().healthStatus = 'ok'
      emitOn('episode:ep-1', 'onetap')
      await tick()
      const subject = useSubjectStore()
      expect(fetchCorpusEpisodes).toHaveBeenCalledWith('/corpus', { limit: 200, cursor: null })
      expect(subject.kind).toBe('episode')
      expect(subject.episodeMetadataPath).toBe('feeds/a/metadata/found.metadata.json')
    })

    it('pages through the catalogue until the episode turns up', async () => {
      vi.mocked(fetchCorpusEpisodes)
        .mockResolvedValueOnce({ items: [{ episode_id: 'other', metadata_relative_path: 'x' }], next_cursor: 'c2' } as never)
        .mockResolvedValueOnce({
          items: [{ episode_id: 'ep-1', metadata_relative_path: 'feeds/a/metadata/p2.metadata.json' }],
          next_cursor: null,
        } as never)
      await mountTracked(noPath())
      useShellStore().healthStatus = 'ok'
      emitOn('episode:ep-1', 'onetap')
      await tick()
      expect(fetchCorpusEpisodes).toHaveBeenCalledTimes(2)
      expect(useSubjectStore().episodeMetadataPath).toBe('feeds/a/metadata/p2.metadata.json')
    })

    it('falls back to the graph rail when the catalogue is unreachable', async () => {
      vi.mocked(fetchCorpusEpisodes).mockRejectedValue(new Error('offline'))
      await mountTracked(noPath())
      useShellStore().healthStatus = 'ok'
      emitOn('episode:ep-1', 'onetap')
      await tick()
      expect(useSubjectStore().kind).toBe('graph-node')
    })

    it('does not ask the catalogue when the corpus is not healthy', async () => {
      await mountTracked(noPath())
      emitOn('episode:ep-1', 'onetap')
      await tick()
      expect(fetchCorpusEpisodes).not.toHaveBeenCalled()
      expect(useSubjectStore().kind).toBe('graph-node')
    })

    it('drops a slow catalogue answer once the user has tapped something else', async () => {
      let release!: (v: unknown) => void
      vi.mocked(fetchCorpusEpisodes).mockReturnValue(new Promise((r) => (release = r)) as never)
      await mountTracked(noPath())
      useShellStore().healthStatus = 'ok'
      emitOn('episode:ep-1', 'onetap')
      await tick()
      emitOn('topic:ai', 'onetap')
      await tick()
      release({
        items: [{ episode_id: 'ep-1', metadata_relative_path: 'feeds/a/metadata/late.metadata.json' }],
        next_cursor: null,
      })
      await tick()
      const subject = useSubjectStore()
      expect(subject.kind).toBe('graph-node')
      expect(subject.graphNodeCyId).toBe('topic:ai')
    })
  })

  it('lets a plugin take over a single tap', async () => {
    const onNodeOneTap = vi.fn(() => true)
    graphCanvasPlugins.register({ id: 'one', onNodeOneTap })
    await mountTracked()
    emitOn('topic:ai', 'onetap')
    await tick()
    expect(onNodeOneTap).toHaveBeenCalled()
    expect(useSubjectStore().kind).toBeNull()
  })

  it('ignores a single tap on the background', async () => {
    await mountTracked()
    liveCore().emit('onetap')
    await tick()
    expect(useSubjectStore().kind).toBeNull()
  })

  it('narrows to a one-hop ego view on shift+double-tap and back on a second one', async () => {
    await mountTracked()
    const nav = useGraphNavigationStore()
    emitOn('person:ada', 'dbltap', { originalEvent: { shiftKey: true } })
    await tick(SETTLE_MS)
    expect(nav.graphEgoFocusCyId).toBe('person:ada')
    expect(liveCore().nodes().map((n) => n.id()).sort()).toEqual(['episode:ep-1', 'person:ada'])
    emitOn('person:ada', 'dbltap', { originalEvent: { shiftKey: true } })
    await tick(SETTLE_MS)
    expect(nav.graphEgoFocusCyId).toBeNull()
    expect(liveCore().nodes()).toHaveLength(5)
  })

  it('leaves the ego view on a plain double-tap of the background', async () => {
    await mountTracked()
    const nav = useGraphNavigationStore()
    emitOn('person:ada', 'dbltap', { originalEvent: { shiftKey: true } })
    await tick(SETTLE_MS)
    expect(nav.graphEgoFocusCyId).toBe('person:ada')
    liveCore().emit('dbltap')
    await tick(SETTLE_MS)
    expect(nav.graphEgoFocusCyId).toBeNull()
    expect(liveCore().nodes()).toHaveLength(5)
  })

  it('ignores a background double-tap when there is no ego focus', async () => {
    await mountTracked()
    const before = cyKit.cores.length
    liveCore().emit('dbltap')
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(before)
  })

  it('lets a plugin take over a double-tap', async () => {
    const onNodeDblTap = vi.fn(() => true)
    graphCanvasPlugins.register({ id: 'dbl', onNodeDblTap })
    await mountTracked()
    emitOn('topic:ai', 'dbltap')
    expect(onNodeDblTap).toHaveBeenCalled()
    expect(cyNode('topic:ai').selected()).toBe(false)
  })
})

describe('GraphCanvas cross-episode expand', () => {
  beforeEach(() => {
    // Read once when the shell store is created, so no health re-probe is scheduled.
    storage.set('ps_corpus_path', '/corpus')
  })

  async function armedCanvas() {
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    const artifacts = useArtifactsStore()
    const append = vi.spyOn(artifacts, 'appendRelativeArtifacts').mockImplementation(async (paths) => {
      artifacts.selectedRelPaths = [...artifacts.selectedRelPaths, ...paths]
    })
    const w = await mountTracked()
    return { w, append, artifacts }
  }

  it('draws no expand ring until the corpus probe has answered', async () => {
    await armedCanvas()
    expect(classesOf('topic:ai')).not.toContain('graph-expand-eligible')
  })

  it('appends the corpus episodes that mention a node and records the expansion', async () => {
    const { append } = await armedCanvas()
    vi.mocked(fetchNodeEpisodes).mockResolvedValue({
      episodes: [
        { gi_relative_path: 'a/ep3.gi.json', kg_relative_path: 'a/ep3.kg.json' },
        { gi_relative_path: ' ', kg_relative_path: null },
      ],
      truncated: true,
      total_matched: 40,
    } as never)
    emitOn('topic:ai', 'dbltap')
    await tick()
    const exp = useGraphExpansionStore()
    expect(append).toHaveBeenCalledWith(['a/ep3.gi.json', 'a/ep3.kg.json'])
    expect(exp.isExpanded('topic:ai')).toBe(true)
    expect(exp.expandedBySeed['topic:ai'].addedRelPaths).toEqual(['a/ep3.gi.json', 'a/ep3.kg.json'])
    expect(exp.truncationLine).toContain('Showing 2 of 40 episodes')
    expect(exp.expansionBusyCyId).toBeNull()
  })

  it('chunks very large expansions into batches of twelve paths', async () => {
    const { append } = await armedCanvas()
    vi.mocked(fetchNodeEpisodes).mockResolvedValue({
      episodes: Array.from({ length: 8 }, (_, i) => ({
        gi_relative_path: `a/e${i}.gi.json`,
        kg_relative_path: `a/e${i}.kg.json`,
      })),
      truncated: false,
    } as never)
    emitOn('topic:ai', 'dbltap')
    await tick()
    expect(append).toHaveBeenCalledTimes(2)
    expect(append.mock.calls[0][0]).toHaveLength(12)
    expect(append.mock.calls[1][0]).toHaveLength(4)
  })

  it('says so when no other episode mentions the node', async () => {
    const { append } = await armedCanvas()
    vi.mocked(fetchNodeEpisodes).mockResolvedValue({ episodes: [], truncated: false } as never)
    emitOn('topic:ai', 'dbltap')
    await tick()
    expect(append).not.toHaveBeenCalled()
    expect(useGraphExpansionStore().truncationLine).toBe(
      'No other episodes in the corpus reference this node.',
    )
  })

  it('surfaces a fetch failure as the truncation line', async () => {
    await armedCanvas()
    vi.mocked(fetchNodeEpisodes).mockRejectedValue(new Error('corpus unreachable'))
    emitOn('topic:ai', 'dbltap')
    await tick()
    expect(useGraphExpansionStore().truncationLine).toBe('corpus unreachable')
    expect(useGraphExpansionStore().expansionBusyCyId).toBeNull()
  })

  it('asks for a healthy corpus path instead of fetching when the corpus is not usable', async () => {
    await armedCanvas()
    useShellStore().healthStatus = null
    emitOn('topic:ai', 'dbltap')
    await tick()
    expect(fetchNodeEpisodes).not.toHaveBeenCalled()
    expect(useGraphExpansionStore().truncationLine).toBe('Set a healthy corpus path to expand across episodes.')
  })

  it('does nothing when a non-expandable node is double-tapped', async () => {
    await armedCanvas()
    emitOn('topic:solo', 'dbltap')
    await tick()
    expect(fetchNodeEpisodes).not.toHaveBeenCalled()
    expect(cyNode('topic:solo').selected()).toBe(true)
  })

  it('collapses an already-expanded node by removing what it added', async () => {
    const { artifacts } = await armedCanvas()
    const remove = vi.spyOn(artifacts, 'removeRelativeArtifacts').mockResolvedValue(undefined as never)
    useGraphExpansionStore().recordExpand('topic:ai', ['a/ep3.gi.json'])
    emitOn('topic:ai', 'dbltap')
    await tick()
    expect(remove).toHaveBeenCalledWith(['a/ep3.gi.json'])
    expect(useGraphExpansionStore().isExpanded('topic:ai')).toBe(false)
  })

  it('probes the corpus for nodes with more episodes beyond the loaded selection', async () => {
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusLibraryApiAvailable = true
    vi.mocked(fetchNodeEpisodes).mockImplementation(async (_root, id) => ({
      episodes: id === 'topic:ai' ? [{ gi_relative_path: 'z/new.gi.json', kg_relative_path: null }] : [],
      truncated: false,
    }) as never)
    await mountTracked()
    await tick(SETTLE_MS)
    const exp = useGraphExpansionStore()
    expect(exp.corpusBeyondAppendKnown('topic:ai')).toBe(true)
    expect(classesOf('topic:ai')).toContain('graph-expand-eligible')
  })

  it('records a failed probe as "nothing beyond"', async () => {
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusLibraryApiAvailable = true
    vi.mocked(fetchNodeEpisodes).mockRejectedValue(new Error('nope'))
    await mountTracked()
    await tick(SETTLE_MS)
    expect(useGraphExpansionStore().corpusBeyondAppendKnown('topic:ai')).toBe(false)
  })

  it('re-probes when the corpus path changes', async () => {
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusLibraryApiAvailable = true
    vi.mocked(fetchNodeEpisodes).mockResolvedValue({ episodes: [], truncated: false } as never)
    await mountTracked()
    await tick(SETTLE_MS)
    const exp = useGraphExpansionStore()
    expect(exp.corpusBeyondAppendKnown('topic:ai')).toBe(false)
    vi.mocked(fetchNodeEpisodes).mockClear()
    shell.corpusPath = '/other'
    await tick()
    expect(exp.corpusBeyondAppendKnown('topic:ai')).toBeUndefined()
    await tick(SETTLE_MS)
    expect(fetchNodeEpisodes).toHaveBeenCalled()
  })

  it('resets expansion state when the selection is emptied', async () => {
    const { artifacts } = await armedCanvas()
    artifacts.selectedRelPaths = ['a.gi.json']
    await tick()
    const exp = useGraphExpansionStore()
    exp.recordExpand('topic:ai', ['x'])
    artifacts.selectedRelPaths = []
    await tick()
    expect(exp.isExpanded('topic:ai')).toBe(false)
  })

  it('re-applies expand hints when the expansion state changes', async () => {
    await armedCanvas()
    const exp = useGraphExpansionStore()
    exp.recordExpand('topic:ai', ['x'])
    await tick()
    expect(classesOf('topic:ai')).toContain('graph-expand-seed')
    exp.resetExpansionState()
    await tick()
    expect(classesOf('topic:ai')).not.toContain('graph-expand-seed')
  })
})

function hit(sourceId: string, extra: Record<string, unknown> = {}) {
  return {
    doc_id: `d-${sourceId}`,
    score: 0.9,
    text: 'snippet',
    metadata: { doc_type: 'kg_topic', source_id: sourceId, episode_id: 'ep-1', ...extra },
  }
}

describe('GraphCanvas bottom-bar wiring', () => {
  it('forwards sync and full-reset requests from the bottom bar', async () => {
    const w = await mountTracked()
    barVm(w).vm.$emit('request-corpus-graph-sync')
    barVm(w).vm.$emit('request-graph-full-reset')
    expect(w.emitted('request-corpus-graph-sync')).toHaveLength(1)
    expect(w.emitted('request-graph-full-reset')).toHaveLength(1)
  })

  it('zooms in and out in 1.2x steps and reports the percentage', async () => {
    const w = await mountTracked()
    const core = liveCore()
    const z0 = core.zoom()
    barVm(w).vm.$emit('zoom-in')
    await tick()
    expect(core.zoom()).toBeCloseTo(z0 * 1.2)
    expect(barVm(w).props('zoomPercent')).toBe(Math.round(z0 * 1.2 * 100))
    barVm(w).vm.$emit('zoom-out')
    barVm(w).vm.$emit('zoom-out')
    await tick()
    expect(core.zoom()).toBeCloseTo(z0 / 1.2)
  })

  it('resets to 100% and centres on the visible graph', async () => {
    const w = await mountTracked()
    const core = liveCore()
    core.zoom(2.5)
    const center = vi.spyOn(core, 'center')
    barVm(w).vm.$emit('zoom-reset')
    await tick()
    expect(core.zoom()).toBe(1)
    expect(center).toHaveBeenCalledTimes(1)
    expect(barVm(w).props('zoomPercent')).toBe(100)
  })

  it('fits the visible elements with an animated 24px-padded fit', async () => {
    const w = await mountTracked()
    cyKit.animations.length = 0
    barVm(w).vm.$emit('fit')
    await tick()
    expect(cyKit.animations).toHaveLength(1)
    const fit = cyKit.animations[0].fit as { eles: { length: number }; padding: number }
    expect(fit.padding).toBe(24)
    expect(fit.eles.length).toBeGreaterThan(5)
  })

  it('exposes fitAnimated to the parent', async () => {
    const w = await mountTracked()
    cyKit.animations.length = 0
    ;(w.vm as unknown as { fitAnimated: () => void }).fitAnimated()
    expect(cyKit.animations).toHaveLength(1)
  })

  it('opens the gesture help from the bottom bar and from the summary strip', async () => {
    const w = await mountTracked()
    barVm(w).vm.$emit('reopen-gestures')
    await w.get('[data-testid="graph-gesture-overlay-reopen"]').trigger('click')
    expect(reopenGestures).toHaveBeenCalledTimes(2)
  })

  it('tells the gesture overlay whether there is a graph and refocuses the canvas on dismiss', async () => {
    const w = await mountTracked()
    expect(w.findComponent(GestureOverlayStub).props('hasNodes')).toBe(true)
    const host = w.get('[tabindex="-1"]').element as HTMLElement
    const focus = vi.spyOn(host, 'focus')
    w.findComponent(GestureOverlayStub).vm.$emit('dismissed')
    await tick()
    expect(focus).toHaveBeenCalled()
  })

  it('cycles the preferred layout and re-lays the graph out', async () => {
    const w = await mountTracked()
    const ge = useGraphExplorerStore()
    const track = vi.spyOn(useGraphAnalyticsStore(), 'track')
    barVm(w).vm.$emit('cycle-layout')
    await tick()
    expect(ge.preferredLayout).toBe('breadthfirst')
    expect(barVm(w).props('preferredLayout')).toBe('breadthfirst')
    expect(track).toHaveBeenCalledWith('graph_redraw', expect.anything())
  })

  it('re-runs the layout in place on relayout without rebuilding the graph', async () => {
    const w = await mountTracked()
    const core = liveCore()
    const track = vi.spyOn(useGraphAnalyticsStore(), 'track')
    barVm(w).vm.$emit('relayout')
    await tick()
    expect(cyKit.cores).toHaveLength(1)
    expect(liveCore()).toBe(core)
    expect(track).toHaveBeenCalledWith('graph_redraw', expect.anything())
  })

  it('relayout keeps a selected node where it was on screen', async () => {
    const w = await mountTracked()
    emitOn('topic:ai', 'tap')
    const before = cyNode('topic:ai').renderedPosition()
    barVm(w).vm.$emit('relayout')
    await tick()
    const after = cyNode('topic:ai').renderedPosition()
    expect(after.x).toBeCloseTo(before.x, 0)
    expect(after.y).toBeCloseTo(before.y, 0)
  })

  it('relayout on the timeline layout rebuilds positions from the date axis', async () => {
    setLayout('timeline')
    const w = await mountTracked()
    cyNode('episode:ep-1').position({ x: 9999, y: 9999 })
    barVm(w).vm.$emit('relayout')
    await tick()
    expect(cyNode('episode:ep-1').position('x')).not.toBe(9999)
  })

  it('does nothing on fit or export when no graph exists', async () => {
    const w = await mountTracked(null)
    expect(cyKit.cores).toHaveLength(0)
    expect(() => {
      ;(w.vm as unknown as { fitAnimated: () => void }).fitAnimated()
      ;(w.vm as unknown as { exportGraphPng: () => void }).exportGraphPng()
    }).not.toThrow()
  })

  it('falls back to a plain fit when the animated fit throws', async () => {
    const w = await mountTracked()
    const core = liveCore()
    ;(core.animate as unknown as ReturnType<typeof vi.fn>).mockImplementationOnce(() => {
      throw new Error('anim')
    })
    const fit = vi.spyOn(core, 'fit')
    barVm(w).vm.$emit('fit')
    expect(fit).toHaveBeenCalledWith(expect.anything(), 24)
  })
})

describe('GraphCanvas PNG export', () => {
  let clicked: HTMLAnchorElement[] = []
  beforeEach(() => {
    clicked = []
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicked.push(this)
    })
  })

  it('downloads a timestamped PNG using the canvas background colour', async () => {
    const w = await mountTracked()
    const core = liveCore()
    const png = vi.fn(() => 'data:image/png;base64,AAAA')
    ;(core as unknown as { png: unknown }).png = png
    barVm(w).vm.$emit('export-png')
    expect(png).toHaveBeenCalledWith({ output: 'base64uri', full: true, scale: 2, bg: '#123456' })
    expect(clicked).toHaveLength(1)
    expect(clicked[0].href).toContain('data:image/png;base64,AAAA')
    expect(clicked[0].download).toMatch(/^gi-kg-viewer-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.png$/)
    expect(document.body.querySelector('a[download]')).toBeNull()
  })

  it('is available to the parent through the exposed API', async () => {
    const w = await mountTracked()
    ;(liveCore() as unknown as { png: unknown }).png = () => 'data:image/png;base64,BBBB'
    ;(w.vm as unknown as { exportGraphPng: () => void }).exportGraphPng()
    expect(clicked).toHaveLength(1)
  })

  it('downloads nothing when the renderer cannot produce an image', async () => {
    const w = await mountTracked()
    ;(liveCore() as unknown as { png: unknown }).png = () => {
      throw new Error('no canvas')
    }
    barVm(w).vm.$emit('export-png')
    expect(clicked).toHaveLength(0)
  })

  it('falls back to the page canvas token, then a built-in dark background', async () => {
    const w = await mountTracked()
    const png = vi.fn(() => 'data:image/png;base64,CC')
    ;(liveCore() as unknown as { png: unknown }).png = png
    vi.mocked(window.getComputedStyle).mockImplementation(
      () =>
        ({
          getPropertyValue: (n: string) =>
            n === '--ps-canvas' ? '#abcdef' : n.startsWith('padding') ? '0px' : '',
        }) as unknown as CSSStyleDeclaration,
    )
    barVm(w).vm.$emit('export-png')
    expect(png).toHaveBeenLastCalledWith(expect.objectContaining({ bg: '#abcdef' }))
    vi.mocked(window.getComputedStyle).mockImplementation(
      () => ({ getPropertyValue: () => '' }) as unknown as CSSStyleDeclaration,
    )
    barVm(w).vm.$emit('export-png')
    expect(png).toHaveBeenLastCalledWith(expect.objectContaining({ bg: '#0a0d10' }))
    vi.mocked(window.getComputedStyle).mockImplementation(() => {
      throw new Error('no style')
    })
    barVm(w).vm.$emit('export-png')
    expect(png).toHaveBeenLastCalledWith(expect.objectContaining({ bg: '#0a0d10' }))
  })
})

describe('GraphCanvas minimap', () => {
  it('mounts the navigator into the minimap host when the minimap is opened', async () => {
    const w = await mountTracked()
    const ge = useGraphExplorerStore()
    ge.minimapOpen = true
    await tick()
    expect(cyKit.navigatorCalls).toHaveLength(1)
    expect(cyKit.navigatorCalls[0]).toMatchObject({ container: '#gi-kg-graph-minimap' })
    expect(w.find('[data-testid="graph-minimap"]').exists()).toBe(true)
  })

  it('sets the navigator up after the first layout when the minimap starts open', async () => {
    useGraphExplorerStore().minimapOpen = true
    await mountTracked()
    expect(cyKit.navigatorCalls).toHaveLength(1)
  })

  it('tears the navigator down when the minimap is closed', async () => {
    await mountTracked()
    const ge = useGraphExplorerStore()
    ge.minimapOpen = true
    await tick()
    ge.minimapOpen = false
    await tick()
    expect(cyKit.navigatorDestroy).toHaveBeenCalledTimes(1)
  })

  it('closes from the minimap close button', async () => {
    const w = await mountTracked()
    const ge = useGraphExplorerStore()
    ge.minimapOpen = true
    await tick()
    await w.get('[data-testid="graph-minimap-close"]').trigger('click')
    expect(ge.minimapOpen).toBe(false)
  })

  it('hides the generated thumbnail alt text from assistive tech', async () => {
    const w = await mountTracked()
    const host = w.get('#gi-kg-graph-minimap').element
    const img = document.createElement('img')
    img.alt = 'Graph navigator'
    host.appendChild(img)
    useGraphExplorerStore().minimapOpen = true
    await tick()
    expect(img.alt).toBe('')
    expect(img.getAttribute('aria-hidden')).toBe('true')
  })

  it('survives a navigator that refuses to start', async () => {
    cyKit.navigatorThrows.value = true
    const w = await mountTracked()
    useGraphExplorerStore().minimapOpen = true
    await tick()
    expect(w.exists()).toBe(true)
    // A later successful open still works.
    cyKit.navigatorThrows.value = false
    useGraphExplorerStore().minimapOpen = false
    await tick()
    useGraphExplorerStore().minimapOpen = true
    await tick()
    expect(cyKit.navigatorCalls.length).toBeGreaterThanOrEqual(2)
  })

  it('does not mount the navigator when the minimap is closed during the deferred setup', async () => {
    await mountTracked()
    const ge = useGraphExplorerStore()
    ge.minimapOpen = true
    ge.minimapOpen = false
    await tick()
    expect(cyKit.navigatorCalls).toHaveLength(0)
  })
})

describe('GraphCanvas radial focus mode', () => {
  function radialData() {
    const data = giData()
    // topic:far is three hops from ep-1 (ep-1 -> ai -> ep-2 -> far): outside both radial rings.
    data.nodes = [...(data.nodes ?? []), { id: 'topic:far', type: 'Topic', properties: { label: 'Far' } }]
    data.edges = [...(data.edges ?? []), { from: 'episode:ep-2', to: 'topic:far', type: 'ABOUT' }]
    return data
  }
  const altR = () =>
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'r', altKey: true, cancelable: true }))
  const esc = () =>
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }))

  it('Alt+R enters radial mode around the selected node and hides the outer hops', async () => {
    const w = await mountTracked(radialData())
    cyNode('episode:ep-1').select()
    altR()
    await tick()
    expect(w.find('[data-testid="graph-radial-mode-active"]').exists()).toBe(true)
    expect(w.get('[aria-live="polite"].sr-only').text()).toBe('Radial view centred on Episode One.')
    expect(cyNode('topic:far').style('display')).toBe('none')
    expect(cyNode('topic:ai').style('display')).toBe('element')
    expect(cyNode('episode:ep-2').style('display')).toBe('element')
  })

  it('Escape leaves radial mode and restores positions and visibility', async () => {
    const w = await mountTracked(radialData())
    const before = { ...cyNode('topic:ai').position() }
    cyNode('episode:ep-1').select()
    altR()
    await tick()
    esc()
    await tick()
    expect(w.find('[data-testid="graph-radial-mode-active"]').exists()).toBe(false)
    expect(w.get('[aria-live="polite"].sr-only').text()).toBe('Radial view exited.')
    expect(cyNode('topic:far').style('display')).toBe('element')
    expect(cyNode('topic:ai').position()).toEqual(before)
  })

  it('Alt+R a second time toggles radial mode back off', async () => {
    const w = await mountTracked(radialData())
    cyNode('episode:ep-1').select()
    altR()
    await tick()
    altR()
    await tick()
    expect(w.find('[data-testid="graph-radial-mode-active"]').exists()).toBe(false)
  })

  it('does nothing on Alt+R without a selection, and ignores Escape when not radial', async () => {
    const w = await mountTracked(radialData())
    altR()
    esc()
    await tick()
    expect(w.find('[data-testid="graph-radial-mode-active"]').exists()).toBe(false)
  })

  it('drops radial mode when the graph is rebuilt', async () => {
    const w = await mountTracked(radialData())
    cyNode('episode:ep-1').select()
    altR()
    await tick()
    ;(w.vm as unknown as { redraw: () => void }).redraw()
    await tick(SETTLE_MS)
    expect(w.find('[data-testid="graph-radial-mode-active"]').exists()).toBe(false)
  })

  it('lets a plugin consume a key before the canvas does', async () => {
    const onKeydown = vi.fn(() => true)
    graphCanvasPlugins.register({ id: 'keys', onKeydown })
    const w = await mountTracked(radialData())
    cyNode('episode:ep-1').select()
    altR()
    await tick()
    expect(onKeydown).toHaveBeenCalled()
    expect(w.find('[data-testid="graph-radial-mode-active"]').exists()).toBe(false)
  })

  it('stops listening for keys once unmounted', async () => {
    const w = await mountTracked(radialData())
    const core = liveCore()
    core.$id('episode:ep-1').select()
    w.unmount()
    mounted = mounted.filter((x) => x !== w)
    const ev = new KeyboardEvent('keydown', { key: 'r', altKey: true, cancelable: true })
    window.dispatchEvent(ev)
    expect(ev.defaultPrevented).toBe(false)
  })
})

describe('GraphCanvas shift-drag box zoom', () => {
  function startDrag(): void {
    liveCore().emit({
      type: 'mousedown',
      originalEvent: { shiftKey: true, clientX: 10, clientY: 10, preventDefault: vi.fn() },
    } as never)
  }
  const move = (x: number, y: number) =>
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: x, clientY: y }))
  const up = (x: number, y: number) =>
    window.dispatchEvent(new MouseEvent('mouseup', { clientX: x, clientY: y }))
  const rect = (w: VueWrapper) => w.get('.border-dashed')

  it('shows a selection rectangle while shift-dragging and fits the dragged model box on release', async () => {
    const w = await mountTracked()
    const core = liveCore()
    ;(core as unknown as { renderer: unknown }).renderer = () => ({
      projectIntoViewport: (x: number, y: number) => [x * 2, y * 2],
    })
    const fit = vi.spyOn(core, 'fit')
    const pan = vi.spyOn(core, 'panningEnabled')
    startDrag()
    expect(pan).toHaveBeenCalledWith(false)
    move(110, 90)
    await tick()
    expect(rect(w).attributes('style')).not.toContain('display: none')
    expect(rect(w).attributes('style')).toContain('width: 100px')
    up(110, 90)
    await tick()
    expect(fit).toHaveBeenCalledWith({ x1: 20, y1: 20, x2: 220, y2: 180 }, 32)
    expect(pan).toHaveBeenLastCalledWith(true)
    expect(rect(w).attributes('style')).toContain('display: none')
  })

  it('does not draw the rectangle for a sub-4px wobble, and does not zoom on a tiny drag', async () => {
    const w = await mountTracked()
    const fit = vi.spyOn(liveCore(), 'fit')
    startDrag()
    move(12, 12)
    await tick()
    expect(rect(w).attributes('style')).toContain('display: none')
    up(14, 14)
    expect(fit).not.toHaveBeenCalled()
  })

  it('ignores a shift mouse-down that lands on a node rather than the background', async () => {
    await mountTracked()
    const pan = vi.spyOn(liveCore(), 'panningEnabled')
    emitOn('topic:ai', 'mousedown', {
      originalEvent: { shiftKey: true, clientX: 1, clientY: 1, preventDefault: vi.fn() },
    })
    expect(pan).not.toHaveBeenCalledWith(false)
  })

  it('ignores a mouse-down without shift and a mouse-up with no drag', async () => {
    await mountTracked()
    const pan = vi.spyOn(liveCore(), 'panningEnabled')
    liveCore().emit({ type: 'mousedown', originalEvent: { shiftKey: false } } as never)
    up(50, 50)
    expect(pan).not.toHaveBeenCalledWith(false)
  })

  it('does not zoom when the viewport cannot project the drag into model space', async () => {
    await mountTracked()
    const core = liveCore()
    const fit = vi.spyOn(core, 'fit')
    startDrag()
    move(200, 200)
    up(200, 200)
    expect(fit).not.toHaveBeenCalled()
  })

  it('cancels a drag in progress when the graph is rebuilt', async () => {
    const w = await mountTracked()
    startDrag()
    move(200, 200)
    ;(w.vm as unknown as { redraw: () => void }).redraw()
    await tick(SETTLE_MS)
    expect(rect(w).attributes('style')).toContain('display: none')
  })
})

describe('GraphCanvas search highlights and context emphasis', () => {
  it('rings nodes that match search results and shows a highlight count', async () => {
    const w = await mountTracked()
    const search = useSearchStore()
    search.results = [hit('topic:ai'), hit('person:ada', { doc_type: 'kg_entity' })] as never
    await tick()
    expect(classesOf('topic:ai')).toContain('search-hit')
    expect(classesOf('person:ada')).toContain('search-hit')
    expect(classesOf('topic:solo')).not.toContain('search-hit')
    expect(w.get('[data-testid="graph-search-highlight-chip"]').text()).toBe('2 highlights')
    expect(barVm(w).props('searchHighlightCount')).toBe(2)
  })

  it('uses the singular label for one highlight and counts a repeated node once', async () => {
    const w = await mountTracked()
    useSearchStore().results = [hit('topic:ai'), hit('topic:ai')] as never
    await tick()
    expect(w.get('[data-testid="graph-search-highlight-chip"]').text()).toBe('1 highlight')
  })

  it('ignores hits that are not graph-focusable or whose node is not on the canvas', async () => {
    const w = await mountTracked()
    useSearchStore().results = [
      hit('topic:ai', { doc_type: 'transcript' }),
      hit('topic:nowhere'),
    ] as never
    await tick()
    expect(w.find('[data-testid="graph-search-highlight-chip"]').exists()).toBe(false)
  })

  it('removes the ring when the results are cleared', async () => {
    await mountTracked()
    const search = useSearchStore()
    search.results = [hit('topic:ai')] as never
    await tick()
    search.results = []
    await tick()
    expect(classesOf('topic:ai')).not.toContain('search-hit')
  })

  it('also highlights episodes the library asks to emphasise', async () => {
    await mountTracked()
    useGraphNavigationStore().setLibraryEpisodeHighlights(['ep-2'])
    await tick()
    expect(classesOf('episode:ep-2')).toContain('search-hit')
  })

  it('frames the highlights when a fit-after-load was staged, then clears the flag', async () => {
    await mountTracked()
    const nav = useGraphNavigationStore()
    const fit = vi.spyOn(liveCore(), 'fit')
    nav.setRequestFitAfterLoad()
    useSearchStore().results = [hit('topic:ai')] as never
    await tick()
    expect(fit).toHaveBeenCalledWith(expect.anything(), 60)
    expect(nav.requestFitAfterLoad).toBe(false)
  })

  it('emphasises the episodes that the active search context favours', async () => {
    await mountTracked()
    const ctx = useActiveSearchContextStore()
    ctx.setContext('ai', [hit('topic:ai', { episode_id: 'ep-1' })] as never)
    await tick()
    expect(classesOf('episode:ep-1')).toContain('context-relevant')
    expect(classesOf('episode:ep-2')).not.toContain('context-relevant')
    ctx.clear()
    await tick()
    expect(classesOf('episode:ep-1')).not.toContain('context-relevant')
  })

  it('applies an already-active search context when the graph is first drawn', async () => {
    useActiveSearchContextStore().setContext('ai', [hit('topic:ai', { episode_id: 'ep-2' })] as never)
    await mountTracked()
    expect(classesOf('episode:ep-2')).toContain('context-relevant')
  })

  it('emphasises insights whose episode the active search context favours', async () => {
    const data = giData()
    data.nodes = [
      ...(data.nodes ?? []),
      { id: 'insight:i1', type: 'Insight', properties: { text: 'x', episode_id: 'ep-1' } },
      { id: 'insight:i2', type: 'Insight', properties: { text: 'y', episode_id: 'ep-2' } },
    ]
    data.edges = [
      ...(data.edges ?? []),
      { from: 'episode:ep-1', to: 'insight:i1', type: 'HAS_INSIGHT' },
      { from: 'episode:ep-2', to: 'insight:i2', type: 'HAS_INSIGHT' },
    ]
    await mountTracked(data)
    useActiveSearchContextStore().setContext('ai', [hit('topic:ai', { episode_id: 'ep-1' })] as never)
    await tick()
    expect(classesOf('insight:i1')).toContain('context-relevant')
    expect(classesOf('insight:i2')).not.toContain('context-relevant')
  })
})

describe('GraphCanvas degree filter', () => {
  it('hides nodes outside the chosen degree bucket and the edges that lose an endpoint', async () => {
    await mountTracked()
    const ge = useGraphExplorerStore()
    ge.activeDegreeBucket = '2-5'
    await tick()
    expect(cyNode('topic:ai').style('display')).toBe('element')
    expect(cyNode('episode:ep-1').style('display')).toBe('element')
    expect(cyNode('person:ada').style('display')).toBe('none')
    expect(cyNode('episode:ep-2').style('display')).toBe('none')
    const visibleEdges = liveCore()
      .edges()
      .filter((e) => e.style('display') !== 'none')
    expect(visibleEdges.length).toBe(1)
    ge.activeDegreeBucket = null
    await tick()
    expect(cyNode('person:ada').style('display')).toBe('element')
    expect(liveCore().edges().filter((e) => e.style('display') === 'none')).toHaveLength(0)
  })
})

describe('GraphCanvas zoom behaviour', () => {
  it('keeps a selected node anchored on screen across incremental zooms', async () => {
    await mountTracked()
    emitOn('topic:ai', 'tap')
    const core = liveCore()
    const before = cyNode('topic:ai').renderedPosition()
    core.zoom(core.zoom() * 1.2)
    const after = cyNode('topic:ai').renderedPosition()
    expect(after.x).toBeCloseTo(before.x, 3)
    expect(after.y).toBeCloseTo(before.y, 3)
  })

  it('zooms about the origin when nothing is selected (no anchor correction)', async () => {
    await mountTracked()
    const core = liveCore()
    const pan = vi.spyOn(core, 'panBy')
    const before = cyNode('topic:ai').renderedPosition()
    core.zoom(core.zoom() * 1.2)
    expect(pan).not.toHaveBeenCalled()
    expect(cyNode('topic:ai').renderedPosition().x).not.toBeCloseTo(before.x, 3)
  })

  it('does not pan to re-anchor on a huge (non-incremental) zoom jump', async () => {
    await mountTracked()
    emitOn('topic:ai', 'tap')
    const core = liveCore()
    const pan = vi.spyOn(core, 'panBy')
    core.zoom(core.zoom() * 4)
    for (const call of pan.mock.calls) {
      const d = call[0] as { x: number; y: number }
      expect(Math.abs(d.x) + Math.abs(d.y)).toBeLessThan(1e-6)
    }
  })

  it('re-centres on the graph shortly after zooming out when nothing is selected', async () => {
    await mountTracked()
    const core = liveCore()
    const center = vi.spyOn(core, 'center')
    core.zoom(core.zoom() / 1.3)
    expect(center).not.toHaveBeenCalled()
    await tick(200)
    expect(center).toHaveBeenCalledTimes(1)
  })

  it('does not re-centre after zooming out while a node is selected', async () => {
    await mountTracked()
    emitOn('topic:ai', 'tap')
    const core = liveCore()
    const center = vi.spyOn(core, 'center')
    core.zoom(core.zoom() / 1.3)
    await tick(200)
    expect(center).not.toHaveBeenCalled()
  })

  it('re-anchors on the node\'s new screen position after the user pans', async () => {
    await mountTracked()
    emitOn('topic:ai', 'tap')
    const core = liveCore()
    core.pan({ x: core.pan().x + 40, y: core.pan().y + 20 })
    const panned = cyNode('topic:ai').renderedPosition()
    core.zoom(core.zoom() * 1.2)
    const after = cyNode('topic:ai').renderedPosition()
    expect(after.x).toBeCloseTo(panned.x, 3)
    expect(after.y).toBeCloseTo(panned.y, 3)
  })
})

describe('GraphCanvas lens overlays', () => {
  beforeEach(() => {
    storage.set('ps_corpus_path', '/corpus')
  })

  it('paints person credibility from the grounding_rate enricher and clears it when switched off', async () => {
    vi.mocked(fetchCachedCorpusEnvelope).mockImplementation((async (_root: string, name: string) =>
      name === 'grounding_rate'
        ? { data: { persons: [{ person_id: 'person:ada', rate: 0.9 }] } }
        : null) as never)
    await mountTracked()
    const lenses = useGraphLensesStore()
    lenses.setPersonCredibility(true)
    await tick()
    expect(fetchCachedCorpusEnvelope).toHaveBeenCalledWith('/corpus', 'grounding_rate')
    expect(classesOf('person:ada')).toContain('credibility-high')
    lenses.setPersonCredibility(false)
    await tick()
    expect(classesOf('person:ada')).not.toContain('credibility-high')
  })

  it('adds co-guest edges from the guest_coappearance enricher', async () => {
    const data = giData()
    data.nodes = [...(data.nodes ?? []), { id: 'person:bo', type: 'Person', properties: { name: 'Bo' } }]
    data.edges = [...(data.edges ?? []), { from: 'episode:ep-1', to: 'person:bo', type: 'SPOKE_IN' }]
    vi.mocked(fetchCachedCorpusEnvelope).mockResolvedValue({
      data: { pairs: [{ person_a_id: 'person:ada', person_b_id: 'person:bo', episode_count: 3 }] },
    } as never)
    await mountTracked(data)
    const lenses = useGraphLensesStore()
    lenses.setCoGuestEdges(true)
    await tick()
    expect(liveCore().edges('.lens-coguest-edge')).toHaveLength(1)
    lenses.setCoGuestEdges(false)
    await tick()
    expect(liveCore().edges('.lens-coguest-edge')).toHaveLength(0)
  })

  it('tints person communities from the guest_coappearance enricher', async () => {
    vi.mocked(fetchCachedCorpusEnvelope).mockResolvedValue({
      data: { communities: [{ community_id: 'c1', member_ids: ['person:ada'] }] },
    } as never)
    await mountTracked()
    useGraphLensesStore().setPersonCommunities(true)
    await tick()
    expect(cyNode('person:ada').data('personCommunityId')).toBe('c1')
    useGraphLensesStore().setPersonCommunities(false)
    await tick()
    expect(cyNode('person:ada').data('personCommunityId')).toBeUndefined()
  })

  it('degrades silently when an enricher envelope cannot be fetched', async () => {
    vi.mocked(fetchCachedCorpusEnvelope).mockRejectedValue(new Error('404'))
    await mountTracked()
    const lenses = useGraphLensesStore()
    lenses.setPersonCredibility(true)
    lenses.setCoGuestEdges(true)
    lenses.setPersonCommunities(true)
    await tick()
    expect(classesOf('person:ada')).not.toContain('credibility-high')
    expect(liveCore().nodes()).toHaveLength(5)
  })

  it('does not fetch any enricher without a corpus path', async () => {
    storage.set('ps_corpus_path', '')
    await mountTracked()
    useGraphLensesStore().setPersonCredibility(true)
    await tick()
    expect(fetchCachedCorpusEnvelope).not.toHaveBeenCalled()
  })

  it('applies and clears a lens a build registered', async () => {
    const apply = vi.fn()
    const clear = vi.fn()
    registeredGraphLenses.register({
      id: 'demo',
      label: 'Demo',
      description: 'd',
      testid: 'demo',
      apply,
      clear,
    })
    await mountTracked()
    expect(clear).toHaveBeenCalled()
    const lenses = useGraphLensesStore()
    lenses.setLens('demo', true)
    await tick()
    expect(apply).toHaveBeenCalledWith(liveCore(), '/corpus')
    clear.mockClear()
    lenses.setLens('demo', false)
    await tick()
    expect(clear).toHaveBeenCalledWith(liveCore())
  })

  it('swallows an error from a registered lens', async () => {
    registeredGraphLenses.register({
      id: 'bad',
      label: 'Bad',
      description: 'd',
      testid: 'bad',
      defaultOn: true,
      apply: () => {
        throw new Error('lens broke')
      },
      clear: () => {},
    })
    useGraphLensesStore().setLens('bad', true)
    await mountTracked()
    expect(liveCore().nodes()).toHaveLength(5)
  })
})

describe('GraphCanvas plugins', () => {
  it('runs setup on mount with a context onto the live core, and the disposer on unmount', async () => {
    const dispose = vi.fn()
    let ctx!: Parameters<NonNullable<GraphCanvasPlugin['setup']>>[0]
    graphCanvasPlugins.register({
      id: 'life',
      setup: (c) => {
        ctx = c
        return dispose
      },
    })
    const w = await mountTracked()
    expect(ctx.core()).toBe(liveCore())
    const seen = vi.fn()
    ctx.guarded('probe', seen)
    expect(seen).toHaveBeenCalledWith(liveCore())
    w.unmount()
    mounted = mounted.filter((x) => x !== w)
    expect(dispose).toHaveBeenCalledTimes(1)
    expect(ctx.core()).toBeNull()
    seen.mockClear()
    ctx.guarded('probe', seen)
    expect(seen).not.toHaveBeenCalled()
  })

  it('calls onLayoutPass and afterRestore after each layout pass', async () => {
    const onLayoutPass = vi.fn()
    const afterRestore = vi.fn()
    graphCanvasPlugins.register({ id: 'pass', onLayoutPass, afterRestore })
    const w = await mountTracked()
    expect(onLayoutPass).toHaveBeenCalledWith(liveCore())
    expect(afterRestore).toHaveBeenCalledWith(liveCore())
    barVm(w).vm.$emit('relayout')
    await tick()
    expect(onLayoutPass).toHaveBeenCalledTimes(2)
  })

  it('keeps drawing when a plugin hook throws', async () => {
    graphCanvasPlugins.register({
      id: 'boom',
      onLayoutPass: () => {
        throw new Error('hook boom')
      },
      afterRestore: () => {
        throw new Error('restore boom')
      },
    })
    await mountTracked()
    expect(liveCore().nodes()).toHaveLength(5)
    expect(useGraphHandoffStore().lastInvariant).not.toBeNull()
  })

  it('renders plugin overlays and notifies plugins when interaction state is reset', async () => {
    const onInteractionReset = vi.fn()
    graphCanvasPlugins.register({
      id: 'ov',
      overlay: defineComponent({ setup: () => () => h('div', { 'data-testid': 'plugin-overlay' }) }),
      onInteractionReset,
    })
    const w = await mountTracked()
    expect(w.find('[data-testid="plugin-overlay"]').exists()).toBe(true)
    ;(w.vm as unknown as { clearInteractionState: () => void }).clearInteractionState()
    expect(onInteractionReset).toHaveBeenCalledTimes(1)
  })
})

describe('GraphCanvas clearInteractionState', () => {
  it('drops selection, subject, pending focus and highlights', async () => {
    const w = await mountTracked()
    const subject = useSubjectStore()
    const nav = useGraphNavigationStore()
    emitOn('topic:ai', 'tap')
    subject.focusGraphNode('topic:ai')
    nav.setLibraryEpisodeHighlights(['ep-1'])
    ;(w.vm as unknown as { clearInteractionState: () => void }).clearInteractionState()
    expect(cyNode('topic:ai').selected()).toBe(false)
    expect(subject.kind).toBeNull()
    expect(nav.libraryHighlightSourceIds).toEqual([])
    expect(nav.pendingFocusNodeId).toBeNull()
  })

  it('leaves an ego view and restores the full graph', async () => {
    const w = await mountTracked()
    const nav = useGraphNavigationStore()
    emitOn('person:ada', 'dbltap', { originalEvent: { shiftKey: true } })
    await tick(SETTLE_MS)
    expect(liveCore().nodes()).toHaveLength(2)
    ;(w.vm as unknown as { clearInteractionState: () => void }).clearInteractionState()
    await tick(SETTLE_MS)
    expect(nav.graphEgoFocusCyId).toBeNull()
    expect(liveCore().nodes()).toHaveLength(5)
  })

  it('can skip the redraw when leaving an ego view', async () => {
    const w = await mountTracked()
    emitOn('person:ada', 'dbltap', { originalEvent: { shiftKey: true } })
    await tick(SETTLE_MS)
    const n = cyKit.cores.length
    ;(w.vm as unknown as { clearInteractionState: (o: object) => void }).clearInteractionState({
      skipRedraw: true,
    })
    expect(cyKit.cores).toHaveLength(n)
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(n)
  })
})

const handoff = (over: Record<string, unknown> = {}) =>
  ({
    kind: 'graph-node',
    cyId: 'topic:ai',
    source: 'search',
    loadSource: 'graph-internal',
    camera: { kind: 'center', cyId: 'topic:ai' },
    ...over,
  }) as never

describe('GraphCanvas pending focus', () => {
  it('selects, opens the rail for and centres a node requested through the navigation store', async () => {
    await mountTracked()
    const nav = useGraphNavigationStore()
    cyKit.animations.length = 0
    nav.requestFocusNode('topic:ai')
    await tick()
    expect(cyNode('topic:ai').selected()).toBe(true)
    expect(nav.pendingFocusNodeId).toBeNull()
    expect(useSubjectStore().graphNodeCyId).toBe('topic:ai')
    expect(cyKit.animations.length).toBeGreaterThanOrEqual(1)
  })

  it('frames extra requested nodes together with the target', async () => {
    await mountTracked()
    cyKit.animations.length = 0
    useGraphNavigationStore().requestFocusNode('topic:ai', undefined, ['ep-2'])
    await tick()
    const center = cyKit.animations[0].center as { eles: { length: number } }
    expect(center.eles.length).toBe(2)
  })

  it('resolves the target through id prefixes', async () => {
    await mountTracked()
    useGraphNavigationStore().requestFocusNode('ep-2')
    await tick()
    expect(cyNode('episode:ep-2').selected()).toBe(true)
  })

  it('uses the fallback node but leaves the rail on the user\'s actual target', async () => {
    await mountTracked()
    const subject = useSubjectStore()
    subject.focusGraphNode('quote:q1')
    useGraphNavigationStore().requestFocusNode('quote:q1', 'ep-1')
    await tick()
    expect(cyNode('episode:ep-1').selected()).toBe(true)
    expect(subject.graphNodeCyId).toBe('quote:q1')
  })

  it('applies a request staged before the graph exists once the first layout lands', async () => {
    useGraphNavigationStore().requestFocusNode('person:ada')
    await mountTracked()
    expect(cyNode('person:ada').selected()).toBe(true)
    expect(useGraphNavigationStore().pendingFocusNodeId).toBeNull()
    expect(cyKit.animations.length).toBeGreaterThanOrEqual(1)
  })

  it('tells plugins when the requested node is not on the canvas', async () => {
    const onFocusTargetMissing = vi.fn()
    graphCanvasPlugins.register({ id: 'miss', onFocusTargetMissing })
    await mountTracked()
    useGraphNavigationStore().requestFocusNode('topic:ghost')
    await tick()
    expect(onFocusTargetMissing).toHaveBeenCalledWith('topic:ghost')
  })

  it('waits for a node that arrives after the request and then focuses it', async () => {
    await mountTracked()
    useGraphHandoffStore().handoffRequested(handoff({ cyId: 'topic:late', camera: { kind: 'center', cyId: 'topic:late' } }))
    useGraphNavigationStore().requestFocusNode('topic:late')
    await tick(100)
    expect(useGraphNavigationStore().pendingFocusNodeId).toBe('topic:late')
    liveCore().add({ group: 'nodes', data: { id: 'topic:late', type: 'Topic', label: 'Late' } })
    await tick(100)
    expect(cyNode('topic:late').selected()).toBe(true)
    expect(useGraphNavigationStore().pendingFocusNodeId).toBeNull()
  })

  it('fails the handoff when the target never appears', async () => {
    await mountTracked()
    const gh = useGraphHandoffStore()
    gh.handoffRequested(handoff({ cyId: 'topic:never', camera: { kind: 'center', cyId: 'topic:never' } }))
    useGraphNavigationStore().requestFocusNode('topic:never')
    await tick(5000)
    expect(gh.lastResult).toMatchObject({ status: 'failed' })
    expect(gh.lastResult?.reason).toContain('no graph node for focus target "topic:never"')
    expect(useGraphNavigationStore().pendingFocusNodeId).toBeNull()
  })

  it('names the fallback in the failure when there was one', async () => {
    await mountTracked()
    const gh = useGraphHandoffStore()
    gh.handoffRequested(handoff({ cyId: 'topic:never', camera: { kind: 'center', cyId: 'topic:never' } }))
    useGraphNavigationStore().requestFocusNode('topic:never', 'ep-nope')
    await tick(5000)
    expect(gh.lastResult?.reason).toContain('(fallback "ep-nope")')
  })
})

describe('GraphCanvas handoff state machine', () => {
  it('marks a pending handoff applied once its target is on the canvas', async () => {
    useGraphHandoffStore().handoffRequested(handoff())
    await mountTracked()
    expect(useGraphHandoffStore().lastResult).toMatchObject({ status: 'applied', appliedCyId: 'topic:ai' })
    expect(useGraphHandoffStore().pending).toBeNull()
  })

  it('finds an episode envelope by metadata path when it carries no node id', async () => {
    useGraphHandoffStore().handoffRequested(
      handoff({
        kind: 'episode',
        cyId: undefined,
        metadataPath: EP2_META,
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      }),
    )
    await mountTracked()
    expect(useGraphHandoffStore().lastResult).toMatchObject({ status: 'applied', appliedCyId: 'episode:ep-2' })
  })

  it('finds an episode envelope by episode id alone', async () => {
    useGraphHandoffStore().handoffRequested(
      handoff({
        kind: 'episode',
        cyId: undefined,
        episodeId: 'ep-1',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      }),
    )
    await mountTracked()
    expect(useGraphHandoffStore().lastResult).toMatchObject({ status: 'applied', appliedCyId: 'episode:ep-1' })
  })

  it('re-scans once, then fails, when the target never shows up on the canvas', async () => {
    useGraphHandoffStore().handoffRequested(handoff({ cyId: 'topic:ghost', camera: { kind: 'center', cyId: 'topic:ghost' } }))
    await mountTracked()
    await tick(1500)
    expect(useGraphHandoffStore().lastResult).toMatchObject({ status: 'failed' })
    expect(useGraphHandoffStore().lastResult?.reason).toContain('apply failed')
  })

  it('does not fail an episode handoff whose target is in the artifact but not yet drawn', async () => {
    const gh = useGraphHandoffStore()
    gh.handoffRequested(
      handoff({
        kind: 'episode',
        cyId: undefined,
        metadataPath: EP2_META,
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      }),
    )
    const w = await mountTracked()
    // The canvas drops episode:ep-2 from view while the artifact still has it, then settles again.
    liveCore().$id('episode:ep-2').remove()
    gh.handoffRequested(
      handoff({
        kind: 'episode',
        cyId: undefined,
        metadataPath: EP2_META,
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      }),
    )
    barVm(w).vm.$emit('relayout')
    await tick()
    expect(gh.lastResult?.status).not.toBe('failed')
  })

  it('parks an intermediate FSM state back at ready after a layout with no envelope', async () => {
    const gh = useGraphHandoffStore()
    expect(gh.advanceState('loading_fetch')).not.toBeNull()
    await mountTracked()
    expect(gh.state).toBe('ready')
  })

  it('restores the last applied selection after the graph is rebuilt', async () => {
    useGraphHandoffStore().handoffRequested(handoff())
    const w = await mountTracked()
    expect(cyNode('topic:ai').selected()).toBe(false)
    cyKit.animations.length = 0
    ;(w.vm as unknown as { redraw: () => void }).redraw()
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(2)
    expect(cyNode('topic:ai').selected()).toBe(true)
    expect(classesOf('topic:ai')).toContain('graph-focused')
    expect(cyKit.animations.length).toBeGreaterThanOrEqual(1)
  })

  it('re-selects and re-centres the applied node when the layout pass itself dropped the selection', async () => {
    const selectedAfterRestore: number[] = []
    graphCanvasPlugins.register({
      id: 'clear',
      onLayoutPass: (c) => void c.nodes().unselect(),
      afterRestore: (c) => void selectedAfterRestore.push(c.nodes(':selected').length),
    })
    useGraphHandoffStore().handoffRequested(handoff())
    const w = await mountTracked()
    expect(selectedAfterRestore.at(-1)).toBe(0)
    cyKit.animations.length = 0
    ;(w.vm as unknown as { redraw: () => void }).redraw()
    await tick(SETTLE_MS)
    // Checked inside the layout pass, before redraw()'s own final re-select could mask a missing restore.
    expect(selectedAfterRestore.at(-1)).toBe(1)
    expect(cyNode('topic:ai').selected()).toBe(true)
    expect(cyKit.animations.length).toBeGreaterThanOrEqual(1)
  })

  it('does not resurrect a selection the user cleared', async () => {
    const gh = useGraphHandoffStore()
    gh.handoffRequested(handoff())
    const w = await mountTracked()
    gh.focusCleared()
    ;(w.vm as unknown as { redraw: () => void }).redraw()
    await tick(SETTLE_MS)
    expect(cyNode('topic:ai').selected()).toBe(false)
  })

  it('recovers a handoff abandoned by a tab switch when the tab is shown again', async () => {
    const Host = defineComponent({
      components: { GraphCanvas },
      props: { show: Boolean },
      template: '<KeepAlive><GraphCanvas v-if="show" /></KeepAlive>',
    })
    loadGi()
    const w = mount(Host, { props: { show: true }, attachTo: document.body, global: { stubs: STUBS } })
    mounted.push(w)
    await tick(SETTLE_MS)
    const core = liveCore()
    const resize = vi.spyOn(core, 'resize')
    await w.setProps({ show: false })
    await tick()
    const gh = useGraphHandoffStore()
    gh.handoffRequested(handoff())
    expect(gh.state).not.toBe('ready')
    await w.setProps({ show: true })
    await tick(SETTLE_MS)
    expect(resize).toHaveBeenCalled()
    expect(core.$id('topic:ai').selected()).toBe(true)
    expect(gh.lastResult).toMatchObject({ status: 'applied', appliedCyId: 'topic:ai' })
  })

  it('leaves a handoff alone on tab return while an artifact load is still in flight', async () => {
    const Host = defineComponent({
      components: { GraphCanvas },
      props: { show: Boolean },
      template: '<KeepAlive><GraphCanvas v-if="show" /></KeepAlive>',
    })
    loadGi()
    const w = mount(Host, { props: { show: true }, attachTo: document.body, global: { stubs: STUBS } })
    mounted.push(w)
    await tick(SETTLE_MS)
    await w.setProps({ show: false })
    const gh = useGraphHandoffStore()
    gh.handoffRequested(handoff())
    useArtifactsStore().loading = true
    await w.setProps({ show: true })
    await tick()
    expect(gh.lastResult?.status).not.toBe('applied')
  })
})

describe('GraphCanvas episode territory', () => {
  beforeEach(() => {
    storage.set('ps_corpus_path', '/corpus')
  })

  const STRIP = '[data-testid="graph-episode-territory-strip"]'
  const LOAD = '[data-testid="graph-episode-territory-load"]'
  const DISMISS = '[data-testid="graph-episode-territory-dismiss"]'

  async function armed() {
    useShellStore().healthStatus = 'ok'
    const artifacts = useArtifactsStore()
    const append = vi.spyOn(artifacts, 'appendRelativeArtifacts').mockResolvedValue(undefined as never)
    const w = await mountTracked()
    return { w, append, artifacts }
  }

  const detail = (over: Record<string, unknown> = {}) =>
    ({
      episode_id: 'ep-9',
      episode_title: 'Nine',
      has_gi: true,
      gi_relative_path: 'a/ep9.gi.json',
      has_kg: true,
      kg_relative_path: 'a/ep9.kg.json',
      ...over,
    }) as never

  it('selects and centres the focused episode when it is on the canvas', async () => {
    const { w } = await armed()
    cyKit.animations.length = 0
    useSubjectStore().focusEpisode(EP1_META, { episodeId: 'ep-1' })
    await tick()
    expect(cyNode('episode:ep-1').selected()).toBe(true)
    expect(w.find(STRIP).exists()).toBe(false)
    expect(cyKit.animations.length).toBeGreaterThanOrEqual(1)
    expect(classesOf('episode:ep-2')).toContain('graph-dimmed')
  })

  it('trusts the connections node when the metadata path matches no graph node', async () => {
    await armed()
    useSubjectStore().focusEpisode('feeds/z/metadata/unknown.metadata.json', {
      graphConnectionsCyId: 'episode:ep-2',
    })
    await tick()
    expect(cyNode('episode:ep-2').selected()).toBe(true)
  })

  it('keeps the path-matched episode when the connections node is a different episode', async () => {
    await armed()
    useSubjectStore().focusEpisode(EP1_META, { graphConnectionsCyId: 'episode:ep-2' })
    await tick()
    expect(cyNode('episode:ep-1').selected()).toBe(true)
    expect(cyNode('episode:ep-2').selected()).toBe(false)
  })

  it('finds the episode by id when the metadata path does not match any node', async () => {
    await armed()
    useSubjectStore().focusEpisode('feeds/z/metadata/unknown.metadata.json', { episodeId: 'ep-2' })
    await tick()
    expect(cyNode('episode:ep-2').selected()).toBe(true)
  })

  it('offers to load an episode that is not in the graph, and loads it automatically once', async () => {
    const { w, append } = await armed()
    vi.mocked(fetchCorpusEpisodeDetail).mockResolvedValue(detail())
    const subject = useSubjectStore()
    subject.focusEpisode('feeds/z/metadata/new.metadata.json')
    await tick()
    expect(w.find(STRIP).exists()).toBe(true)
    expect(fetchCorpusEpisodeDetail).toHaveBeenCalledTimes(1)
    expect(fetchCorpusEpisodeDetail).toHaveBeenCalledWith('/corpus', 'feeds/z/metadata/new.metadata.json')
    expect(append).toHaveBeenCalledWith(['a/ep9.gi.json', 'a/ep9.kg.json'])
    expect(subject.episodeId).toBe('ep-9')
    expect(subject.episodeUiLabel).toBe('Nine')
    expect(useArtifactsStore().currentLoadSource).toBe('subject-external')
    // The auto-load guard stops it from looping on the same path.
    await tick(SETTLE_MS)
    expect(fetchCorpusEpisodeDetail).toHaveBeenCalledTimes(1)
  })

  it('walks the handoff FSM forward when the load was started by a handoff', async () => {
    const { append } = await armed()
    const gh = useGraphHandoffStore()
    gh.handoffRequested(handoff({ kind: 'episode', cyId: undefined, metadataPath: 'feeds/z/metadata/new.metadata.json', camera: { kind: 'center-on-target' } }))
    vi.mocked(fetchCorpusEpisodeDetail).mockResolvedValue(detail())
    useSubjectStore().focusEpisode('feeds/z/metadata/new.metadata.json')
    await tick()
    expect(append).toHaveBeenCalled()
    expect(['loading_merge', 'redrawing_full', 'applying', 'ready']).toContain(gh.state)
  })

  it('loads on demand with the strip button and honours a stale handoff', async () => {
    const { w, append } = await armed()
    vi.mocked(fetchCorpusEpisodeDetail).mockResolvedValue(detail())
    useSubjectStore().focusEpisode('feeds/z/metadata/new.metadata.json')
    await tick()
    append.mockClear()
    await w.get(LOAD).trigger('click')
    await tick()
    expect(fetchCorpusEpisodeDetail).toHaveBeenCalledTimes(2)
    expect(append).toHaveBeenCalledTimes(1)
  })

  it('drops a load result that a newer handoff has superseded', async () => {
    const { w, append } = await armed()
    let release!: (v: unknown) => void
    vi.mocked(fetchCorpusEpisodeDetail).mockReturnValue(new Promise((r) => (release = r)) as never)
    useSubjectStore().focusEpisode('feeds/z/metadata/new.metadata.json')
    await tick()
    useGraphHandoffStore().handoffRequested(handoff())
    release(detail())
    await tick()
    expect(append).not.toHaveBeenCalled()
    expect(w.exists()).toBe(true)
  })

  it('reports a failed slice load through the handoff state machine', async () => {
    await armed()
    vi.mocked(fetchCorpusEpisodeDetail).mockRejectedValue(new Error('boom'))
    useSubjectStore().focusEpisode('feeds/z/metadata/new.metadata.json')
    await tick()
    expect(useGraphHandoffStore().lastResult).toMatchObject({
      status: 'failed',
      reason: 'territory fetch: boom',
    })
  })

  it('appends nothing when the episode has neither a GI nor a KG file', async () => {
    const { append } = await armed()
    vi.mocked(fetchCorpusEpisodeDetail).mockResolvedValue(
      detail({ has_gi: false, has_kg: false }),
    )
    useSubjectStore().focusEpisode('feeds/z/metadata/new.metadata.json')
    await tick()
    expect(append).not.toHaveBeenCalled()
  })

  it('cannot load without a healthy corpus', async () => {
    const w = await mountTracked()
    useSubjectStore().focusEpisode('feeds/z/metadata/new.metadata.json')
    await tick()
    expect(w.find(STRIP).exists()).toBe(true)
    expect(w.get(LOAD).attributes('disabled')).toBeDefined()
    expect(fetchCorpusEpisodeDetail).not.toHaveBeenCalled()
  })

  it('dismissing the strip clears the focus state and stops auto-loading', async () => {
    const { w } = await armed()
    vi.mocked(fetchCorpusEpisodeDetail).mockResolvedValue(detail({ has_gi: false, has_kg: false }))
    useSubjectStore().focusEpisode('feeds/z/metadata/new.metadata.json')
    await tick()
    await w.get(DISMISS).trigger('click')
    await tick()
    expect(w.find(STRIP).exists()).toBe(false)
    await tick(SETTLE_MS)
    expect(fetchCorpusEpisodeDetail).toHaveBeenCalledTimes(1)
  })

  it('keeps the strip dismissed through later subject updates, and offers it again for another episode', async () => {
    const { w } = await armed()
    vi.mocked(fetchCorpusEpisodeDetail).mockResolvedValue(detail({ has_gi: false, has_kg: false }))
    const subject = useSubjectStore()
    subject.focusEpisode('feeds/z/metadata/new.metadata.json')
    await tick()
    await w.get(DISMISS).trigger('click')
    subject.setEpisodeUiLabel('A title arrived later')
    await tick()
    expect(w.find(STRIP).exists()).toBe(false)
    subject.focusEpisode('feeds/z/metadata/another.metadata.json')
    await tick()
    expect(w.find(STRIP).exists()).toBe(true)
  })

  it('goes away when the subject is no longer an episode', async () => {
    const { w } = await armed()
    vi.mocked(fetchCorpusEpisodeDetail).mockResolvedValue(detail({ has_gi: false, has_kg: false }))
    const subject = useSubjectStore()
    subject.focusEpisode('feeds/z/metadata/new.metadata.json')
    await tick()
    expect(w.find(STRIP).exists()).toBe(true)
    subject.focusGraphNode('topic:ai')
    await tick()
    expect(w.find(STRIP).exists()).toBe(false)
  })

  it('tries a new episode afresh after a failed one', async () => {
    await armed()
    vi.mocked(fetchCorpusEpisodeDetail).mockRejectedValueOnce(new Error('first'))
    const subject = useSubjectStore()
    subject.focusEpisode('feeds/z/metadata/one.metadata.json')
    await tick()
    vi.mocked(fetchCorpusEpisodeDetail).mockResolvedValue(detail({ has_gi: false, has_kg: false }))
    subject.focusEpisode('feeds/z/metadata/two.metadata.json')
    await tick()
    expect(fetchCorpusEpisodeDetail).toHaveBeenCalledTimes(2)
  })
})

/** giData plus a topic hanging off episode two. */
function grownData(): ArtifactData {
  const d = giData()
  d.nodes = [...(d.nodes ?? []), { id: 'topic:new', type: 'Topic', properties: { label: 'New' } }]
  d.edges = [...(d.edges ?? []), { from: 'episode:ep-2', to: 'topic:new', type: 'ABOUT' }]
  return d
}

describe('GraphCanvas artifact changes', () => {
  it('adds appended nodes next to their neighbour without rebuilding the graph', async () => {
    await mountTracked()
    const core = liveCore()
    const kept = { ...cyNode('episode:ep-1').position() }
    loadGi(grownData())
    await tick(50)
    expect(cyKit.cores).toHaveLength(1)
    expect(liveCore()).toBe(core)
    expect(cyNode('topic:new').empty()).toBe(false)
    const a = cyNode('topic:new').position()
    const b = cyNode('episode:ep-2').position()
    expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeCloseTo(40, 0)
    expect(cyNode('episode:ep-1').position()).toEqual(kept)
    expect(useGraphHandoffStore().lastInvariant).toMatchObject({ missing: [] })
  })

  it('re-runs the layout-pass overlays after an in-place append', async () => {
    const onLayoutPass = vi.fn()
    graphCanvasPlugins.register({ id: 'pass', onLayoutPass })
    await mountTracked()
    onLayoutPass.mockClear()
    loadGi(grownData())
    await tick(50)
    expect(onLayoutPass).toHaveBeenCalledTimes(1)
  })

  it('keeps the selection across an in-place append', async () => {
    await mountTracked()
    emitOn('topic:ai', 'tap')
    loadGi(grownData())
    await tick(50)
    expect(cyNode('topic:ai').selected()).toBe(true)
  })

  it('rebuilds the graph when the artifact is replaced by a different one', async () => {
    await mountTracked()
    const ge = useGraphExplorerStore()
    emitOn('topic:ai', 'tap')
    const smaller = giData()
    smaller.nodes = (smaller.nodes ?? []).filter((n) => n.id !== 'topic:solo')
    smaller.edges = (smaller.edges ?? []).filter((e) => e.to !== 'topic:solo')
    loadGi(smaller)
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(2)
    expect(liveCore().$id('topic:solo').empty()).toBe(true)
    expect(ge.activeDegreeBucket).toBeNull()
  })

  it('resets the degree filter when a new artifact replaces the graph', async () => {
    await mountTracked()
    const ge = useGraphExplorerStore()
    ge.activeDegreeBucket = '1'
    await tick()
    const smaller = giData()
    smaller.nodes = (smaller.nodes ?? []).filter((n) => n.id !== 'person:ada')
    smaller.edges = (smaller.edges ?? []).filter((e) => e.to !== 'person:ada')
    loadGi(smaller)
    await tick(SETTLE_MS)
    expect(ge.activeDegreeBucket).toBeNull()
  })

  it('does a full rebuild for an external navigation even when the new graph is a superset', async () => {
    await mountTracked()
    useArtifactsStore().setLoadSource('subject-external')
    loadGi(grownData())
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(2)
    expect(cyNode('topic:new').empty()).toBe(false)
  })

  it('consumes the load-source marker once it has seen the change', async () => {
    await mountTracked()
    const artifacts = useArtifactsStore()
    artifacts.setLoadSource('digest-external')
    loadGi(grownData())
    await tick(SETTLE_MS)
    expect(artifacts.currentLoadSource).toBeNull()
  })

  it('lays out only the new nodes when the selection grows quickly after a layout', async () => {
    const artifacts = useArtifactsStore()
    artifacts.selectedRelPaths = ['a.gi.json']
    await mountTracked(giData(), { settle: false })
    await tick(50)
    const core = liveCore()
    const kept = { ...cyNode('episode:ep-1').position() }
    artifacts.selectedRelPaths = ['a.gi.json', 'b.gi.json']
    loadGi(grownData())
    // Cooldown remainder (~250ms) + debounce (150ms) + the re-entered debounce (150ms).
    await tick(1000)
    expect(cyKit.cores).toHaveLength(1)
    expect(liveCore()).toBe(core)
    expect(cyNode('topic:new').empty()).toBe(false)
    expect(cyNode('episode:ep-1').position()).toEqual(kept)
  })

  it('shows the empty-view message when the graph is replaced by one with nothing visible', async () => {
    const artifacts = useArtifactsStore()
    artifacts.selectedRelPaths = ['a.gi.json']
    const w = await mountTracked(giData(), { settle: false })
    await tick(50)
    artifacts.selectedRelPaths = ['a.gi.json', 'b.gi.json']
    const onlyQuote = giData()
    onlyQuote.nodes = [{ id: 'quote:q', type: 'Quote', properties: { text: 'q' } }]
    onlyQuote.edges = []
    loadGi(onlyQuote)
    await tick(1000)
    expect(w.text()).toContain('No nodes in this view')
  })

  it('rebuilds when the breadcrumb trail grows', async () => {
    await mountTracked()
    useGraphNavigationStore().addToTrail('topic:solo')
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(2)
  })

  it('rebuilds when the theme changes', async () => {
    await mountTracked()
    useThemeStore().choice = 'light'
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(2)
  })

  it('coalesces a burst of rebuild triggers into one', async () => {
    await mountTracked()
    const nav = useGraphNavigationStore()
    nav.addToTrail('topic:solo')
    await tick(10)
    nav.addToTrail('person:ada')
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(2)
  })

  it('defers a rebuild requested during the post-layout cooldown instead of dropping it', async () => {
    await mountTracked(giData(), { settle: false })
    await tick(50)
    useGraphNavigationStore().addToTrail('topic:solo')
    await tick(10)
    expect(cyKit.cores).toHaveLength(1)
    await tick(1000)
    expect(cyKit.cores).toHaveLength(2)
  })

  it('rebuilds immediately when a handoff is in flight', async () => {
    await mountTracked()
    useGraphHandoffStore().handoffRequested(handoff())
    useGraphNavigationStore().addToTrail('topic:solo')
    await tick(5)
    expect(cyKit.cores).toHaveLength(2)
  })

  it('restores the full-graph viewport when leaving ego view', async () => {
    await mountTracked()
    const before = cyNode('episode:ep-1').renderedPosition()
    const zoom = liveCore().zoom()
    emitOn('episode:ep-1', 'dbltap', { originalEvent: { shiftKey: true } })
    await tick(SETTLE_MS)
    emitOn('episode:ep-1', 'dbltap', { originalEvent: { shiftKey: true } })
    await tick(SETTLE_MS)
    expect(liveCore().zoom()).toBeCloseTo(zoom, 6)
    expect(cyNode('episode:ep-1').renderedPosition().x).toBeCloseTo(before.x, 0)
  })

  it('exposes redraw so the parent can force a rebuild', async () => {
    const w = await mountTracked()
    ;(w.vm as unknown as { redraw: () => void }).redraw()
    expect(cyKit.cores).toHaveLength(2)
    expect(cyKit.cores[0].destroyed()).toBe(true)
  })

  it('queues a rebuild requested while one is running', async () => {
    const w = await mountTracked()
    let inner: Core | null = null
    graphCanvasPlugins.register({
      id: 'reenter',
      onLayoutPass: (c) => {
        inner = c
        ;(w.vm as unknown as { redraw: () => void }).redraw()
      },
    })
    ;(w.vm as unknown as { redraw: () => void }).redraw()
    await tick(SETTLE_MS)
    expect(inner).not.toBeNull()
    expect(liveCore().nodes()).toHaveLength(5)
  })
})

describe('GraphCanvas lifecycle', () => {
  it('destroys the graph and stops watching the window when unmounted', async () => {
    const w = await mountTracked()
    const core = liveCore()
    w.unmount()
    mounted = mounted.filter((x) => x !== w)
    expect(core.destroyed()).toBe(true)
    expect(FakeResizeObserver.instances.every((o) => o.disconnect.mock.calls.length > 0)).toBe(true)
    expect(() => {
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 5, clientY: 5 }))
      window.dispatchEvent(new MouseEvent('mouseup', { clientX: 5, clientY: 5 }))
    }).not.toThrow()
  })

  it('cancels pending timers on unmount so nothing rebuilds afterwards', async () => {
    const w = await mountTracked()
    useGraphNavigationStore().addToTrail('topic:solo')
    await tick(10)
    w.unmount()
    mounted = mounted.filter((x) => x !== w)
    const n = cyKit.cores.length
    await tick(SETTLE_MS * 2)
    expect(cyKit.cores).toHaveLength(n)
  })

  it('gives up cleanly when the configured layout does not exist', async () => {
    setLayout('nope' as never)
    const w = await mountTracked()
    expect(cyKit.cores.length).toBeGreaterThanOrEqual(1)
    expect(cyKit.cores.every((c) => c.destroyed())).toBe(true)
    expect(w.find('[data-testid="graph-layout-loading"]').exists()).toBe(false)
  })

  it('keeps the graph when a relayout is requested with a layout that does not exist', async () => {
    const w = await mountTracked()
    setLayout('nope' as never)
    barVm(w).vm.$emit('relayout')
    await tick()
    expect(liveCore().nodes()).toHaveLength(5)
  })

  it('waits for a usable container size, then draws once it has one', async () => {
    vi.mocked(window.getComputedStyle).mockImplementation(
      () => ({ getPropertyValue: () => '' }) as unknown as CSSStyleDeclaration,
    )
    await mountTracked()
    expect(cyKit.cores).toHaveLength(0)
    const waiter = FakeResizeObserver.instances.find((o) => o.observe.mock.calls.length > 0)
    expect(waiter).toBeDefined()
    waiter!.cb()
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(0)
    vi.mocked(window.getComputedStyle).mockImplementation(
      () => ({ getPropertyValue: (n: string) => (n.startsWith('padding') ? '0px' : '') }) as never,
    )
    waiter!.cb()
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(1)
    expect(waiter!.disconnect).toHaveBeenCalled()
  })

  it('re-centres on a freshly focused node when the canvas is resized', async () => {
    await mountTracked()
    useGraphNavigationStore().requestFocusNode('topic:ai')
    await tick()
    const core = liveCore()
    const center = vi.spyOn(core, 'center')
    const resize = vi.spyOn(core, 'resize')
    for (const o of FakeResizeObserver.instances) o.cb()
    expect(resize).toHaveBeenCalled()
    expect(center).toHaveBeenCalled()
  })

  it('stops re-centring once the focus window has elapsed', async () => {
    await mountTracked()
    useGraphNavigationStore().requestFocusNode('topic:ai')
    await tick()
    const core = liveCore()
    await tick(6000)
    const center = vi.spyOn(core, 'center')
    for (const o of FakeResizeObserver.instances) o.cb()
    expect(center).not.toHaveBeenCalled()
  })

  it('rebuilds the stylesheet when the reduced-motion preference changes', async () => {
    const handlers: Array<() => void> = []
    const mql = {
      matches: false,
      addEventListener: vi.fn((_t: string, fn: () => void) => handlers.push(fn)),
      removeEventListener: vi.fn(),
    }
    vi.spyOn(window, 'matchMedia').mockReturnValue(mql as unknown as MediaQueryList)
    const w = await mountTracked()
    expect(window.matchMedia).toHaveBeenCalledWith('(prefers-reduced-motion: reduce)')
    expect(handlers).toHaveLength(1)
    const core = liveCore()
    const style = vi.spyOn(core, 'style')
    mql.matches = true
    handlers[0]()
    expect(style).toHaveBeenCalled()
    w.unmount()
    mounted = mounted.filter((x) => x !== w)
    expect(mql.removeEventListener).toHaveBeenCalledWith('change', handlers[0])
  })

  it('ignores a reduced-motion change before any graph exists', async () => {
    const handlers: Array<() => void> = []
    vi.spyOn(window, 'matchMedia').mockReturnValue({
      matches: false,
      addEventListener: (_t: string, fn: () => void) => handlers.push(fn),
      removeEventListener: () => {},
    } as unknown as MediaQueryList)
    await mountTracked(null)
    expect(() => handlers[0]()).not.toThrow()
  })

  it('sizes hub nodes by degree only while the size-by-degree lens is on', async () => {
    const lenses = useGraphLensesStore()
    lenses.setNodeSizeByDegree(true)
    await mountTracked()
    const on = Number(cyNode('episode:ep-1').style('width').replace('px', ''))
    mounted.pop()!.unmount()
    cyKit.cores.length = 0
    lenses.setNodeSizeByDegree(false)
    await mountTracked()
    const off = Number(cyNode('episode:ep-1').style('width').replace('px', ''))
    expect(on).not.toBe(off)
  })
})

describe('GraphCanvas keeps the user\'s subject across a graph replacement', () => {
  function smallerData(): ArtifactData {
    const d = giData()
    d.nodes = (d.nodes ?? []).filter((n) => n.id !== 'topic:solo')
    d.edges = (d.edges ?? []).filter((e) => e.to !== 'topic:solo')
    return d
  }

  it('re-selects the graph node the rail is showing once the new graph is drawn', async () => {
    await mountTracked()
    const subject = useSubjectStore()
    subject.focusGraphNode('topic:ai')
    loadGi(smallerData())
    await tick(SETTLE_MS)
    expect(subject.kind).toBe('graph-node')
    expect(cyNode('topic:ai').selected()).toBe(true)
    expect(useGraphNavigationStore().pendingFocusNodeId).toBeNull()
  })

  it('re-selects the focused episode, preferring the connections node', async () => {
    await mountTracked()
    const nav = useGraphNavigationStore()
    const request = vi.spyOn(nav, 'requestFocusNode')
    useSubjectStore().focusEpisode(EP1_META, { graphConnectionsCyId: 'episode:ep-1' })
    loadGi(smallerData())
    await tick(SETTLE_MS)
    expect(request).toHaveBeenCalledWith('episode:ep-1')
  })

  it('finds the focused episode by metadata path when there is no connections node', async () => {
    await mountTracked()
    const request = vi.spyOn(useGraphNavigationStore(), 'requestFocusNode')
    useSubjectStore().focusEpisode(EP2_META)
    loadGi(smallerData())
    await tick(SETTLE_MS)
    expect(request).toHaveBeenCalledWith('episode:ep-2')
  })

  it('clears a subject that is not tied to the graph', async () => {
    await mountTracked()
    const subject = useSubjectStore()
    subject.focusShow('feed-1')
    loadGi(smallerData())
    await tick(SETTLE_MS)
    expect(subject.kind).toBeNull()
  })

  it('does not queue a second focus while a handoff is already in flight', async () => {
    await mountTracked()
    const request = vi.spyOn(useGraphNavigationStore(), 'requestFocusNode')
    useSubjectStore().focusGraphNode('topic:ai')
    useGraphHandoffStore().handoffRequested(handoff({ cyId: 'person:ada', camera: { kind: 'center', cyId: 'person:ada' } }))
    loadGi(smallerData())
    await tick(SETTLE_MS)
    expect(request).not.toHaveBeenCalled()
  })

  it('does not re-request the node the handoff just applied, even under the other layer prefix', async () => {
    const gh = useGraphHandoffStore()
    gh.handoffRequested(handoff({ cyId: 'g:topic:ai' }))
    gh.recordApplied('g:topic:ai')
    await mountTracked()
    const request = vi.spyOn(useGraphNavigationStore(), 'requestFocusNode')
    useSubjectStore().focusGraphNode('k:topic:ai')
    loadGi(smallerData())
    await tick(SETTLE_MS)
    expect(request).not.toHaveBeenCalled()
  })
})

describe('GraphCanvas viewport on layout', () => {
  it('frames the search hits (not the whole graph) when a fit was requested', async () => {
    const w = await mountTracked()
    const nav = useGraphNavigationStore()
    useSearchStore().results = [hit('topic:ai')] as never
    await tick()
    const fit = vi.spyOn(liveCore(), 'fit')
    nav.setRequestFitAfterLoad()
    barVm(w).vm.$emit('relayout')
    await tick()
    const target = fit.mock.calls[0][0] as { length: number }
    expect(fit.mock.calls[0][1]).toBe(60)
    expect(target.length).toBe(1)
  })

  it('frames every visible element when a fit was requested and nothing is highlighted', async () => {
    const w = await mountTracked()
    const fit = vi.spyOn(liveCore(), 'fit')
    useGraphNavigationStore().setRequestFitAfterLoad()
    barVm(w).vm.$emit('relayout')
    await tick()
    const target = fit.mock.calls[0][0] as { length: number }
    expect(target.length).toBeGreaterThan(5)
    expect(useGraphNavigationStore().requestFitAfterLoad).toBe(false)
  })

  it('leaves the camera to the pending focus while one is queued', async () => {
    const w = await mountTracked()
    const fit = vi.spyOn(liveCore(), 'fit')
    useGraphNavigationStore().requestFocusNode('topic:ghost')
    barVm(w).vm.$emit('relayout')
    await tick()
    expect(fit).not.toHaveBeenCalled()
  })
})

describe('GraphCanvas layouts that are still running', () => {
  it('shows a loading overlay while a relayout is in flight and removes it when a new one finishes', async () => {
    const w = await mountTracked()
    setLayout('hang' as never)
    barVm(w).vm.$emit('relayout')
    await tick()
    expect(hangKit.started).toBe(1)
    expect(w.find('[data-testid="graph-layout-loading"]').exists()).toBe(true)
    expect(w.find('[aria-busy]').attributes('aria-busy')).toBe('true')
    setLayout('grid')
    barVm(w).vm.$emit('relayout')
    await tick(SETTLE_MS)
    expect(hangKit.stopped).toBe(1)
    expect(w.find('[data-testid="graph-layout-loading"]').exists()).toBe(false)
    expect(w.find('[aria-busy]').attributes('aria-busy')).toBe('false')
  })

  it('stops a layout that is still running when the graph is rebuilt', async () => {
    const w = await mountTracked()
    setLayout('hang' as never)
    barVm(w).vm.$emit('relayout')
    await tick()
    setLayout('grid')
    ;(w.vm as unknown as { redraw: () => void }).redraw()
    await tick(SETTLE_MS)
    expect(hangKit.stopped).toBe(1)
  })

  it('aborts an incremental layout that never converges and still finishes the pass', async () => {
    const artifacts = useArtifactsStore()
    artifacts.selectedRelPaths = ['a.gi.json']
    await mountTracked(giData(), { settle: false })
    await tick(50)
    const gh = useGraphHandoffStore()
    const before = gh.lastInvariant?.ts ?? 0
    setLayout('hang' as never)
    artifacts.selectedRelPaths = ['a.gi.json', 'b.gi.json']
    loadGi(grownData())
    await tick(1000)
    expect(hangKit.started).toBe(1)
    await tick(8100)
    expect(hangKit.stopped).toBe(1)
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('layout timeout after 8000ms'))
    expect(gh.lastInvariant!.ts).toBeGreaterThan(before)
    expect(cyNode('topic:new').empty()).toBe(false)
  })
})

describe('GraphCanvas incremental placement of appended nodes', () => {
  it('seeds new nodes next to topic neighbours, other neighbours, or beside the graph when unconnected', async () => {
    const artifacts = useArtifactsStore()
    artifacts.selectedRelPaths = ['a.gi.json']
    await mountTracked(giData(), { settle: false })
    await tick(50)
    const data = grownData()
    data.nodes = [
      ...(data.nodes ?? []),
      { id: 'person:new', type: 'Person', properties: { name: 'New' } },
      { id: 'topic:orphan', type: 'Topic', properties: { label: 'Orphan' } },
    ]
    data.edges = [...(data.edges ?? []), { from: 'person:new', to: 'topic:ai', type: 'SPOKE_IN' }]
    const kept = { ...cyNode('episode:ep-1').position() }
    artifacts.selectedRelPaths = ['a.gi.json', 'b.gi.json']
    loadGi(data)
    await tick(1000)
    expect(cyKit.cores).toHaveLength(1)
    for (const id of ['topic:new', 'person:new', 'topic:orphan']) {
      const p = cyNode(id).position()
      expect(Number.isFinite(p.x) && Number.isFinite(p.y)).toBe(true)
    }
    expect(cyNode('episode:ep-1').position()).toEqual(kept)
    expect(useGraphHandoffStore().lastInvariant).toMatchObject({ missing: [], extra: [] })
  })
})

describe('GraphCanvas timeline details', () => {
  it('places an insight with no episode id near the episodes it connects to', async () => {
    const data = giData()
    data.nodes = [...(data.nodes ?? []), { id: 'insight:legacy', type: 'Insight', properties: { text: 'x' } }]
    data.edges = [...(data.edges ?? []), { from: 'episode:ep-2', to: 'insight:legacy', type: 'HAS_INSIGHT' }]
    setLayout('timeline')
    await mountTracked(data)
    expect(cyNode('insight:legacy').position('x')).toBeCloseTo(cyNode('episode:ep-2').position('x'), 3)
  })

  it('parks an insight with no connected episode at the default x', async () => {
    const data = giData()
    data.nodes = [...(data.nodes ?? []), { id: 'insight:lone', type: 'Insight', properties: { text: 'x' } }]
    setLayout('timeline')
    await mountTracked(data)
    expect(cyNode('insight:lone').position('x')).toBe(0)
  })
})

describe('GraphCanvas tab return with an episode handoff', () => {
  const KeepAliveHost = defineComponent({
    components: { GraphCanvas },
    props: { show: Boolean },
    template: '<KeepAlive><GraphCanvas v-if="show" /></KeepAlive>',
  })

  it('finds the episode through its metadata path when the tab is shown again', async () => {
    loadGi()
    const w = mount(KeepAliveHost, { props: { show: true }, attachTo: document.body, global: { stubs: STUBS } })
    mounted.push(w)
    await tick(SETTLE_MS)
    const core = liveCore()
    await w.setProps({ show: false })
    const gh = useGraphHandoffStore()
    gh.handoffRequested(
      handoff({
        kind: 'episode',
        cyId: undefined,
        metadataPath: EP2_META,
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      }),
    )
    await w.setProps({ show: true })
    await tick(SETTLE_MS)
    expect(core.$id('episode:ep-2').selected()).toBe(true)
    expect(gh.lastResult).toMatchObject({ status: 'applied', appliedCyId: 'episode:ep-2' })
  })

  it('leaves the handoff in flight when its target is not on the canvas', async () => {
    loadGi()
    const w = mount(KeepAliveHost, { props: { show: true }, attachTo: document.body, global: { stubs: STUBS } })
    mounted.push(w)
    await tick(SETTLE_MS)
    await w.setProps({ show: false })
    const gh = useGraphHandoffStore()
    gh.handoffRequested(
      handoff({
        kind: 'episode',
        cyId: undefined,
        metadataPath: 'feeds/q/metadata/absent.metadata.json',
        episodeId: 'ep-absent',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      }),
    )
    await w.setProps({ show: true })
    await tick(100)
    expect(gh.pending).not.toBeNull()
    expect(gh.lastResult?.status).not.toBe('applied')
  })
})

describe('GraphCanvas redraw scheduling and in-place appends', () => {
  it('holds a requested rebuild back while a layout is still running', async () => {
    const w = await mountTracked()
    setLayout('hang' as never)
    barVm(w).vm.$emit('relayout')
    await tick()
    useGraphNavigationStore().addToTrail('topic:solo')
    await tick(SETTLE_MS)
    expect(cyKit.cores).toHaveLength(1)
  })

  it('still refreshes overlays when an identical artifact is re-supplied', async () => {
    await mountTracked()
    const gh = useGraphHandoffStore()
    const before = gh.lastInvariant!.ts
    vi.setSystemTime(Date.now() + 10)
    loadGi(giData())
    await tick(50)
    expect(cyKit.cores).toHaveLength(1)
    expect(gh.lastInvariant!.ts).toBeGreaterThan(before)
  })

  it('positions an appended node next to its neighbour when the neighbour is the edge target', async () => {
    await mountTracked()
    const data = giData()
    data.nodes = [...(data.nodes ?? []), { id: 'person:late', type: 'Person', properties: { name: 'Late' } }]
    data.edges = [...(data.edges ?? []), { from: 'person:late', to: 'topic:ai', type: 'SPOKE_IN' }]
    loadGi(data)
    await tick(50)
    const a = cyNode('person:late').position()
    const b = cyNode('topic:ai').position()
    expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeCloseTo(40, 0)
  })

  it('adds an appended node that has no neighbour yet without positioning it', async () => {
    await mountTracked()
    const data = giData()
    data.nodes = [...(data.nodes ?? []), { id: 'topic:lonely', type: 'Topic', properties: { label: 'Lonely' } }]
    loadGi(data)
    await tick(50)
    expect(cyNode('topic:lonely').empty()).toBe(false)
    expect(cyKit.cores).toHaveLength(1)
  })
})

describe('GraphCanvas handoff target scan', () => {
  it('finds an episode node by its own data when neither the artifact nor the id schemes resolve it', async () => {
    graphCanvasPlugins.register({
      id: 'inject',
      onLayoutPass: (c) => {
        if (c.$id('weird-1').empty()) {
          c.add({ group: 'nodes', data: { id: 'weird-1', type: 'Episode', episode_id: 'zz' } })
        }
      },
    })
    useGraphHandoffStore().handoffRequested(
      handoff({
        kind: 'episode',
        cyId: undefined,
        episodeId: 'zz',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      }),
    )
    await mountTracked()
    expect(useGraphHandoffStore().lastResult).toMatchObject({ status: 'applied', appliedCyId: 'weird-1' })
  })
})
