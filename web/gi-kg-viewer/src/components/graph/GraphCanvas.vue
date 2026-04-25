<script setup lang="ts">
import cytoscape, { type Core, type NodeSingular } from 'cytoscape'
// @ts-expect-error — package has no TypeScript types (CommonJS extension)
import registerNavigator from 'cytoscape-navigator'
import 'cytoscape-navigator/cytoscape.js-navigator.css'
import { storeToRefs } from 'pinia'
import {
  computed,
  nextTick,
  onActivated,
  onMounted,
  onUnmounted,
  reactive,
  ref,
  watch,
} from 'vue'
import {
  fetchCorpusEpisodeDetail,
  fetchNodeEpisodes,
  GRAPH_NODE_EPISODES_EXPAND_MAX,
} from '../../api/corpusLibraryApi'
import { useArtifactsStore } from '../../stores/artifacts'
import { useSubjectStore } from '../../stores/subject'
import { useGraphExpansionStore } from '../../stores/graphExpansion'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphLensesStore } from '../../stores/graphLenses'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useSearchStore } from '../../stores/search'
import { useShellStore } from '../../stores/shell'
import { useThemeStore } from '../../stores/theme'
import type { RawGraphNode } from '../../types/artifact'
import { graphNodeFill, graphNodeLegendLabel } from '../../utils/colors'
import { degreeBucketFor, emptyDegreeCounts } from '../../utils/graphDegreeBuckets'
import {
  findEpisodeGraphNodeIdForMetadataPath,
  graphCyIdRepresentsEpisodeNode,
  logicalEpisodeIdFromGraphNodeId,
  metadataPathFromEpisodeProperties,
  normalizeCorpusMetadataPath,
  resolveEpisodeMetadataFromLoadedArtifacts,
  resolveEpisodeMetadataViaCorpusCatalog,
} from '../../utils/graphEpisodeMetadata'
import * as giKgCoseLayout from '../../utils/cyCoseLayoutOptions'
import { syncGraphLabelTierClasses } from '../../utils/cyGraphLabelTier'
import { buildGiKgCyStylesheet, cytoscapeSideLabelMarginXCallback } from '../../utils/cyGraphStylesheet'
import {
  computeRadialPositions,
  type RadialSnapshot,
} from '../../utils/cyRadialLayout'
import {
  graphNodeExpandableForCrossEpisodeExpand,
  syncCrossEpisodeExpandNodeClasses,
} from '../../utils/graphCrossEpisodeExpand'
import { wouldCrossEpisodeExpandAppendNewArtifacts } from '../../utils/graphCorpusBeyondSelection'
import { findRawNodeInArtifact, toCytoElements, toGraphElements } from '../../utils/parsing'
import { graphNodeIdFromSearchHit, resolveCyNodeId } from '../../utils/searchFocus'
import { StaleGeneration } from '../../utils/staleGeneration'
import { visualNodeTypeCounts } from '../../utils/visualGroup'
import GraphBottomBar from './GraphBottomBar.vue'
import GraphFiltersPopover from './GraphFiltersPopover.vue'
import GraphGestureOverlay from './GraphGestureOverlay.vue'
import GraphStatusLine from './GraphStatusLine.vue'

registerNavigator(cytoscape)

const emit = defineEmits<{
  'request-corpus-graph-sync': []
  'request-graph-full-reset': []
}>()

const gf = useGraphFilterStore()
const lenses = useGraphLensesStore()
const ge = useGraphExplorerStore()
const { preferredLayout, minimapOpen, activeDegreeBucket } = storeToRefs(ge)
const nav = useGraphNavigationStore()
const subject = useSubjectStore()
const artifacts = useArtifactsStore()
const graphExpansion = useGraphExpansionStore()
const { expandedBySeed } = storeToRefs(graphExpansion)
const shell = useShellStore()
const searchStore = useSearchStore()
const themeStore = useThemeStore()

const graphEpisodeOpenGate = new StaleGeneration()

/** Minimum zoom when animating to a focused node (digest/search hand-off + canvas single-tap). */
const GRAPH_FOCUS_FRAME_MIN_ZOOM = 1.3

/** Episode on Graph: select Episode node for subject episode; Cytoscape 1-hop neighbourhood dim. */
const episodeTerritoryMode = ref<'off' | 'empty'>('off')
const episodeTerritoryLoadBusy = ref(false)
const episodeTerritoryDismissed = ref(false)

watch(
  () => subject.episodeMetadataPath,
  () => {
    episodeTerritoryDismissed.value = false
  },
)

/** Prefer ego slice (matches canvas); fall back to merged graph so Library-appended episodes resolve. */
function rawNodeForRailInteraction(cyId: string): RawGraphNode | null {
  const ego = gf.viewWithEgo(nav.graphEgoFocusCyId)
  const fromEgo = ego ? findRawNodeInArtifact(ego, cyId) : null
  if (fromEgo) {
    return fromEgo
  }
  const full = gf.filteredArtifact
  return full ? findRawNodeInArtifact(full, cyId) : null
}

async function openGraphEpisodeOrNodeRail(
  cyId: string,
  rawNode: RawGraphNode | null,
): Promise<void> {
  const token = graphEpisodeOpenGate.bump()
  if (!graphCyIdRepresentsEpisodeNode(cyId, rawNode)) {
    subject.focusGraphNode(cyId)
    return
  }
  const episodeRaw = rawNode?.type === 'Episode' ? rawNode : null
  let meta = episodeRaw ? metadataPathFromEpisodeProperties(episodeRaw)?.trim() || null : null
  const eid =
    logicalEpisodeIdFromGraphNodeId(cyId) ||
    (episodeRaw && typeof episodeRaw.properties?.episode_id === 'string'
      ? episodeRaw.properties.episode_id.trim()
      : null)
  if (!meta && eid) {
    meta = resolveEpisodeMetadataFromLoadedArtifacts(
      eid,
      artifacts.parsedList,
      artifacts.selectedRelPaths,
    )
  }
  if (!meta && eid && shell.corpusPath.trim() && shell.healthStatus) {
    try {
      meta = await resolveEpisodeMetadataViaCorpusCatalog(
        shell.corpusPath.trim(),
        eid,
        8,
        () => graphEpisodeOpenGate.isStale(token),
      )
    } catch {
      meta = null
    }
  }
  if (graphEpisodeOpenGate.isStale(token)) return
  if (meta) {
    subject.focusEpisode(meta, { graphConnectionsCyId: cyId })
  } else {
    subject.focusGraphNode(cyId)
  }
}

async function expandOrCollapseGraphNode(core: Core, cyId: string): Promise<void> {
  const id = cyId.trim()
  if (!id) {
    return
  }
  if (graphExpansion.isExpanded(id)) {
    graphExpansion.clearTruncationLine()
    await graphExpansion.collapseSeed(id)
    applyCrossEpisodeExpandHints(core)
    return
  }
  const rawNode = rawNodeForRailInteraction(id)
  if (!graphNodeExpandableForCrossEpisodeExpand(core, id, rawNode)) {
    return
  }
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    graphExpansion.setTruncationLine('Set a healthy corpus path to expand across episodes.')
    return
  }
  graphExpansion.setBusy(id)
  graphExpansion.clearTruncationLine()
  try {
    const res = await fetchNodeEpisodes(root, id, GRAPH_NODE_EPISODES_EXPAND_MAX)
    const flat: string[] = []
    for (const ep of res.episodes) {
      const gi = ep.gi_relative_path?.trim()
      const kg = ep.kg_relative_path?.trim()
      if (gi) {
        flat.push(gi)
      }
      if (kg) {
        flat.push(kg)
      }
    }
    if (flat.length === 0) {
      graphExpansion.setTruncationLine('No other episodes in the corpus reference this node.')
      return
    }
    const prevSel = new Set(artifacts.selectedRelPaths.map((p) => p.replace(/\\/g, '/')))
    const CHUNK = 12
    for (let i = 0; i < flat.length; i += CHUNK) {
      await artifacts.appendRelativeArtifacts(flat.slice(i, i + CHUNK))
    }
    const added = artifacts.selectedRelPaths.filter((p) => !prevSel.has(p.replace(/\\/g, '/')))
    graphExpansion.recordExpand(id, added)
    if (res.truncated && res.total_matched != null) {
      graphExpansion.setTruncationLine(
        `Showing ${res.episodes.length} of ${res.total_matched} episodes (max_episodes cap).`,
      )
    }
  } catch (e) {
    graphExpansion.setTruncationLine(e instanceof Error ? e.message : String(e))
  } finally {
    graphExpansion.setBusy(null)
    applyCrossEpisodeExpandHints(core)
  }
}

watch(
  expandedBySeed,
  () => {
    if (cy) {
      applyCrossEpisodeExpandHints(cy)
    }
  },
  { deep: true },
)

const container = ref<HTMLDivElement | null>(null)
const canvasHost = ref<HTMLDivElement | null>(null)
/** True while cytoscape is rebuilding (after redraw); hides default/square pre-layout frame. */
const graphContentHiddenUntilLayout = ref(false)

/** Sync hide on the graph host before Cytoscape paints (Vue class bindings can lag one tick). */
function applyGraphCanvasImmediateHide(el: HTMLElement): void {
  el.setAttribute('data-gi-graph-paint-hold', '1')
  el.style.visibility = 'hidden'
  el.style.opacity = '0'
  el.style.pointerEvents = 'none'
}

function clearGraphCanvasImmediateHide(el: HTMLElement | null | undefined): void {
  if (!el?.getAttribute('data-gi-graph-paint-hold')) return
  el.removeAttribute('data-gi-graph-paint-hold')
  el.style.removeProperty('visibility')
  el.style.removeProperty('opacity')
  el.style.removeProperty('pointer-events')
}

function releaseGraphCanvasLayoutHold(): void {
  graphContentHiddenUntilLayout.value = false
  clearGraphCanvasImmediateHide(container.value)
  const cam = pendingFocusCameraAfterLayoutHold
  pendingFocusCameraAfterLayoutHold = null
  const core = cy
  if (cam && core) {
    void nextTick(() => {
      requestAnimationFrame(() => {
        if (!cy || cy !== core) {
          return
        }
        try {
          core.resize()
        } catch {
          /* ignore */
        }
        animateCameraToFocusedNode(core, cam.cyId, {
          extraRawIds: cam.extras.length ? cam.extras : undefined,
        })
      })
    })
  }
}

const minimapHost = ref<HTMLDivElement | null>(null)

const focusNodeId = ref<string | null>(null)
const selectedNodeId = ref<string | null>(null)
const graphPrefersReducedMotion = ref(false)
const searchHighlightCount = ref(0)
/** Node count for the active Cytoscape view (0 when empty or not mounted). */
const graphCyNodeCount = ref(0)
const gestureOverlayRef = ref<{ reopen: () => void } | null>(null)

function focusCanvasHost(): void {
  void nextTick(() => {
    canvasHost.value?.focus()
  })
}

const zoomPercent = ref(100)
const degreeHistogramCounts = ref<Record<string, number>>({})

const boxZoomRect = reactive({
  show: false,
  left: 0,
  top: 0,
  width: 0,
  height: 0,
})

let cy: Core | null = null
/** Prevents nested ``redraw()`` (e.g. artifact watcher during sync COSE ``run()``) from destroying the active instance mid-layout. */
let redrawGateDepth = 0
/** Debounce ``POST /api/corpus/node-episodes`` probes that decide the teal ring (corpus beyond selection). */
let nodeEpisodesCorpusBeyondDebounce: ReturnType<typeof setTimeout> | null = null
let redrawPending = false
let resizeObs: ResizeObserver | null = null
let zoomCenterTimer: ReturnType<typeof setTimeout> | null = null
let lastZoomLevel = 1
let navInstance: { destroy: () => void } | null = null
let zoomPanListenerAttached = false
let reducedMotionMql: MediaQueryList | null = null
let reducedMotionMqlHandler: (() => void) | null = null

let boxDragging = false
let boxStartClient = { x: 0, y: 0 }
let boxListenersAttached = false

/** In-flight element layout (e.g. COSE rAF). Must be stopped before starting another or positions revert. */
let activeElesLayout: { stop: () => void } | null = null

/** Bumps when a new layout run starts so stale layoutstop handlers from a stopped layout are ignored. */
const graphLayoutGate = new StaleGeneration()

/**
 * After each successful layout pass we record the visible node id set + selection size + ego focus.
 * When the user appends GI/KG (selection grows) we restore prior positions and run COSE only on the
 * new subgraph so the existing graph does not reflow.
 */
let lastSelectedRelPathsCountAfterLayout = 0
let lastCommittedFilteredNodeIds = new Set<string>()
let lastCommittedEgoFocusCyId = ''

/** Captured before relayout or ego exit; consumed in finishLayoutPass to avoid fit() jumping the view. */
type ViewportPreserveSnap = {
  cyId: string
  zoom: number
  rx: number
  ry: number
  /** Relayout: only restore if this node is still selected. Ego exit: restore regardless of selection. */
  requireSelectedMatch: boolean
}

/** Library/Digest handoff: run ``animateCameraToFocusedNode`` after paint hold so Cytoscape has real layout size. */
let pendingFocusCameraAfterLayoutHold: { cyId: string; extras: string[] } | null = null

let pendingViewportPreserve: ViewportPreserveSnap | null = null
/** Full-graph viewport before entering 1-hop (shift+dbl-click); applied when returning to full graph. */
let egoPriorFullGraphViewportPreserve: ViewportPreserveSnap | null = null

/** Target rendered (screen) position for the selected node; kept stable across wheel + toolbar zoom. */
let selectedNodeZoomAnchor: { x: number; y: number } | null = null

/** Skip pan-by-anchor in zoom handler while applyViewportPreserveOrFit runs (zoom+pan ordering). */
let suspendSelectedNodeZoomAnchorCorrection = 0

function clearSelectedNodeZoomAnchor(): void {
  selectedNodeZoomAnchor = null
}

function refreshSelectedNodeZoomAnchor(core: Core): void {
  const sid = selectedNodeId.value
  if (!sid || !nodeOkForViewportPreserve(core, sid)) {
    selectedNodeZoomAnchor = null
    return
  }
  try {
    const rp = core.$id(sid).renderedPosition()
    selectedNodeZoomAnchor = { x: rp.x, y: rp.y }
  } catch {
    selectedNodeZoomAnchor = null
  }
}

function nodeOkForViewportPreserve(core: Core, cyId: string): boolean {
  try {
    const n = core.$id(cyId)
    return (
      !n.empty() &&
      n.visible() &&
      String(n.style('display')) !== 'none'
    )
  } catch {
    return false
  }
}

/** Snapshot of selected node in viewport coords (for restore after layout / degree filter). */
function captureSelectedViewportAnchor(core: Core): ViewportPreserveSnap | null {
  const sid = selectedNodeId.value
  if (!sid || !nodeOkForViewportPreserve(core, sid)) {
    return null
  }
  try {
    const rp = core.$id(sid).renderedPosition()
    return {
      cyId: sid,
      zoom: core.zoom(),
      rx: rp.x,
      ry: rp.y,
      requireSelectedMatch: true,
    }
  } catch {
    return null
  }
}

/** Preserve zoom + screen position of a node (used when entering ego from full graph). */
function captureViewportAnchorForCyId(core: Core, cyId: string): ViewportPreserveSnap | null {
  if (!cyId || !nodeOkForViewportPreserve(core, cyId)) {
    return null
  }
  try {
    const rp = core.$id(cyId).renderedPosition()
    return {
      cyId,
      zoom: core.zoom(),
      rx: rp.x,
      ry: rp.y,
      requireSelectedMatch: false,
    }
  } catch {
    return null
  }
}

function applyViewportPreserveOrFit(
  core: Core,
  snap: ViewportPreserveSnap | null,
): void {
  suspendSelectedNodeZoomAnchorCorrection += 1
  try {
    if (
      snap &&
      nodeOkForViewportPreserve(core, snap.cyId) &&
      (!snap.requireSelectedMatch || selectedNodeId.value === snap.cyId)
    ) {
      try {
        core.zoom(snap.zoom)
        const nr = core.$id(snap.cyId).renderedPosition()
        core.panBy({ x: snap.rx - nr.x, y: snap.ry - nr.y })
        return
      } catch {
        /* fall through to fit */
      }
    }
    try {
      core.fit(core.elements(':visible'), 24)
    } catch {
      /* ignore */
    }
  } finally {
    suspendSelectedNodeZoomAnchorCorrection -= 1
    refreshSelectedNodeZoomAnchor(core)
  }
}

function destroyNavigator(): void {
  if (navInstance) {
    try {
      navInstance.destroy()
    } catch {
      /* ignore */
    }
    navInstance = null
  }
}

/** cytoscape-navigator sets img alt to "Graph navigator", which browsers show before src loads. */
function clearMinimapThumbnailAlt(): void {
  const host = minimapHost.value ?? document.getElementById('gi-kg-graph-minimap')
  const img = host?.querySelector('img')
  if (!img) {
    return
  }
  img.alt = ''
  img.setAttribute('aria-hidden', 'true')
}

function setupNavigator(core: Core): void {
  destroyNavigator()
  if (!minimapOpen.value || !minimapHost.value) {
    return
  }
  try {
    // cytoscape-navigator only mounts into a user container when `container` is a
    // non-empty string selector; a DOM element is ignored and a 400×400 fixed
    // panel is appended to document.body instead.
    const api = (core as unknown as { navigator: (o: object) => { destroy: () => void } }).navigator(
      {
        container: '#gi-kg-graph-minimap',
        removeCustomContainer: false,
        viewLiveFramerate: 0,
        // Throttle full-graph PNG for minimap; too low hammers the main thread on pan/zoom.
        rerenderDelay: 300,
      },
    )
    navInstance = api
    clearMinimapThumbnailAlt()
    // Thumbnail updates hook `cy.onRender`. We often attach the navigator after the
    // graph has already drawn (deferred setup after first paint), so no `render` fires
    // until the user interacts — force one draw so the minimap gets an initial PNG.
    try {
      core.forceRender()
    } catch {
      /* ignore */
    }
  } catch {
    navInstance = null
  }
}

function applySearchHighlights(core: Core): void {
  core.nodes().removeClass('search-hit')
  const matched = new Set<string>()
  for (const hit of searchStore.results) {
    const rawId = graphNodeIdFromSearchHit(hit)
    if (!rawId) continue
    const cyId = resolveCyNodeId(core, rawId)
    if (cyId && !matched.has(cyId)) {
      matched.add(cyId)
      core.$id(cyId).addClass('search-hit')
    }
  }
  for (const raw of nav.libraryHighlightSourceIds) {
    const cyId = resolveCyNodeId(core, raw)
    if (cyId && !matched.has(cyId)) {
      matched.add(cyId)
      core.$id(cyId).addClass('search-hit')
    }
  }
  searchHighlightCount.value = matched.size
}

function buildCyStyle() {
  return [
    ...(buildGiKgCyStylesheet({
      includeSearchHit: true,
      prefersReducedMotion: graphPrefersReducedMotion.value,
      // RFC-080 V5 — opt-in via lens flag (defaults off; user toggle
      // persists across reloads via useGraphLensesStore).
      enableNodeSizeByDegree: lenses.nodeSizeByDegree,
    }) as Record<string, unknown>[]),
    {
      selector: 'node',
      style: {
        'text-margin-x': cytoscapeSideLabelMarginXCallback(false),
      },
    },
  ] as never
}

function clearGraphSelectionDim(core: Core): void {
  core.batch(() => {
    core.nodes().removeClass('graph-dimmed graph-focused graph-neighbour')
    core.edges().removeClass('graph-edge-dimmed graph-edge-neighbour')
  })
}

function clearEpisodeRepresentativeGraphState(core: Core | null): void {
  episodeTerritoryMode.value = 'off'
  if (!core) {
    return
  }
  try {
    core.nodes().unselect()
  } catch {
    /* ignore */
  }
  selectedNodeId.value = null
  clearGraphSelectionDim(core)
  applySearchHighlights(core)
}

function applyEpisodeRepresentativeFocusIfNeeded(
  core: Core,
  opts?: { skipCamera?: boolean },
): void {
  if (episodeTerritoryDismissed.value) {
    clearEpisodeRepresentativeGraphState(core)
    return
  }
  if (subject.kind !== 'episode') {
    episodeTerritoryMode.value = 'off'
    return
  }
  const meta = subject.episodeMetadataPath?.trim()
  if (!meta) {
    episodeTerritoryMode.value = 'off'
    return
  }
  const metaNorm = normalizeCorpusMetadataPath(meta)
  const best = findEpisodeGraphNodeIdForMetadataPath(gf.filteredArtifact, meta)
  const preferred = subject.graphConnectionsCyId?.trim() || ''

  let cyEpisodeId: string | null = best
  if (preferred) {
    const prefColl = core.$id(preferred)
    if (!prefColl.empty()) {
      /** Merged graph only: ego slice (``rawNodeForRailInteraction``) can miss the Episode row right after Open in graph / reload while ``best`` still resolves on the full artifact — then we must not keep a stale ``best`` and ignore ``graphConnectionsCyId``. */
      const rawPrefMerged = gf.filteredArtifact
        ? findRawNodeInArtifact(gf.filteredArtifact, preferred)
        : null
      const prefIsEpisode = graphCyIdRepresentsEpisodeNode(preferred, rawPrefMerged)
      if (prefIsEpisode) {
        if (!best) {
          /** Corpus ``metadata_relative_path`` can disagree with graph row text; ``graphConnectionsCyId`` is authoritative. */
          cyEpisodeId = preferred
        } else if (rawPrefMerged?.type === 'Episode') {
          const mp = metadataPathFromEpisodeProperties(rawPrefMerged)?.trim()
          if (mp && normalizeCorpusMetadataPath(mp) === metaNorm) {
            cyEpisodeId = preferred
          }
        } else if (!rawPrefMerged && logicalEpisodeIdFromGraphNodeId(preferred)?.trim()) {
          cyEpisodeId = preferred
        }
      }
    }
  }

  if (!cyEpisodeId) {
    episodeTerritoryMode.value = 'empty'
    clearEpisodeRepresentativeGraphState(core)
    return
  }
  const coll = core.$id(cyEpisodeId)
  if (coll.empty()) {
    episodeTerritoryMode.value = 'empty'
    clearEpisodeRepresentativeGraphState(core)
    return
  }
  episodeTerritoryMode.value = 'off'
  const node = coll.first() as NodeSingular
  const selectedNow = core.nodes(':selected')
  const onlyThisEpisodeSelected =
    selectedNow.length === 1 && (selectedNow.first() as NodeSingular).id() === cyEpisodeId

  if (!onlyThisEpisodeSelected) {
    try {
      core.nodes().unselect()
    } catch {
      /* ignore */
    }
    selectedNodeId.value = null
    node.select()
    selectedNodeId.value = cyEpisodeId
  } else {
    selectedNodeId.value = cyEpisodeId
  }
  refreshSelectedNodeZoomAnchor(core)
  applySearchHighlights(core)
  if (!opts?.skipCamera) {
    try {
      /** ``fit(closedNeighborhood())`` zoomed out on hub episodes and fought ``tryApplyPendingFocus``. */
      animateCameraToFocusedNode(core, cyEpisodeId)
    } catch {
      /* ignore */
    }
  }
  try {
    applyGraphSelectionDimFromNode(core, node)
  } catch {
    /* ignore */
  }
}

function dismissEpisodeTerritoryStrip(): void {
  episodeTerritoryDismissed.value = true
  clearEpisodeRepresentativeGraphState(cy)
}

async function loadEpisodeSliceForTerritoryStrip(): Promise<void> {
  const meta = subject.episodeMetadataPath?.trim()
  const root = shell.corpusPath.trim()
  if (!meta || !root || !shell.healthStatus) {
    return
  }
  episodeTerritoryLoadBusy.value = true
  try {
    const d = await fetchCorpusEpisodeDetail(root, meta)
    const paths: string[] = []
    if (d.has_gi && d.gi_relative_path?.trim()) {
      paths.push(d.gi_relative_path.trim())
    }
    if (d.has_kg && d.kg_relative_path?.trim()) {
      paths.push(d.kg_relative_path.trim())
    }
    if (paths.length === 0) {
      return
    }
    await artifacts.appendRelativeArtifacts(paths)
    subject.setEpisodeUiLabel(d.episode_title ?? null)
    episodeTerritoryDismissed.value = false
  } catch {
    /* ignore — caller sees empty strip until retry */
  } finally {
    episodeTerritoryLoadBusy.value = false
  }
}

function cyNodeDataType(n: NodeSingular): string {
  return String(n.data('type') ?? '')
}

/** Cytoscape ``data.type`` mirrors ``visualGroupForNode`` (usually ``Episode``); tolerate casing from parsers. */
function cyNodeIsEpisodeForSelectionDim(n: NodeSingular): boolean {
  return cyNodeDataType(n).trim().toLowerCase() === 'episode'
}

/**
 * Single-step highlight: ``closedNeighborhood()`` (focused node + incident edges + adjacent nodes).
 * Not multi-hop BFS — if a hub touches many episodes, one hop can still look very busy.
 * When focus is an **Episode**, other Episode nodes stay dimmed so shared GI/KG structure does not
 * light up every episode at once.
 */
function applyGraphSelectionDimFromNode(core: Core, node: NodeSingular): void {
  const focusIsEpisode = cyNodeIsEpisodeForSelectionDim(node)
  core.batch(() => {
    core.nodes().addClass('graph-dimmed')
    core.edges().addClass('graph-edge-dimmed')
    node.removeClass('graph-dimmed').addClass('graph-focused')
    const hood = node.closedNeighborhood()
    hood.nodes().forEach((nn) => {
      nn.addClass('graph-neighbour').removeClass('graph-dimmed')
    })
    if (focusIsEpisode) {
      hood.nodes().forEach((nn) => {
        if (nn.id() !== node.id() && cyNodeIsEpisodeForSelectionDim(nn)) {
          nn.removeClass('graph-neighbour').addClass('graph-dimmed')
        }
      })
    }
    core.edges().forEach((ee) => {
      const sn = ee.source()
      const tn = ee.target()
      const sDim = sn.hasClass('graph-dimmed')
      const tDim = tn.hasClass('graph-dimmed')
      if (!sDim && !tDim) {
        ee.addClass('graph-edge-neighbour').removeClass('graph-edge-dimmed')
      } else {
        ee.addClass('graph-edge-dimmed').removeClass('graph-edge-neighbour')
      }
    })
  })
}

/** WIP §4.3 — Topic hub emphasis from graph degree (post-layout). */
function applyTopicDegreeHeat(core: Core): void {
  const maxDegree = 30
  core.batch(() => {
    core.nodes('[type = "Topic"]').forEach((n) => {
      const d = n.degree(false)
      const heat = Math.min(1, d / maxDegree)
      try {
        n.data('degreeHeat', heat)
      } catch {
        /* ignore */
      }
      if (heat > 0.7) {
        n.addClass('graph-topic-heat-high')
      } else {
        n.removeClass('graph-topic-heat-high')
      }
    })
  })
}

function layoutOptionsFor(name: string): Record<string, unknown> {
  if (name === 'cose') {
    // Namespace import + fallback: avoids rare `ReferenceError` from named-import/HMR chunks.
    try {
      const fn = giKgCoseLayout.giKgCoseLayoutOptionsMain
      if (typeof fn === 'function') {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return fn() as any
      }
    } catch (e) {
      console.warn('[GraphCanvas] giKgCoseLayoutOptionsMain failed, using fallback', e)
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return giKgCoseLayout.giKgCoseLayoutOptionsMainFallback() as any
  }
  return { name, padding: 36 }
}

type CyModelPosition = { x: number; y: number }

type ModelBBox = { x1: number; y1: number; x2: number; y2: number }

function stableHashString(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(31, h) + s.charCodeAt(i)) | 0
  }
  return h
}

function modelBBoxOfFixedNodes(core: Core, addedIds: Set<string>): ModelBBox | null {
  let x1 = Infinity
  let y1 = Infinity
  let x2 = -Infinity
  let y2 = -Infinity
  let any = false
  core.nodes().forEach((n) => {
    if (addedIds.has(n.id())) {
      return
    }
    any = true
    const p = n.position()
    x1 = Math.min(x1, p.x)
    y1 = Math.min(y1, p.y)
    x2 = Math.max(x2, p.x)
    y2 = Math.max(y2, p.y)
  })
  if (!any || !(x2 > x1 && y2 > y1)) {
    return null
  }
  return { x1, y1, x2, y2 }
}

/** Connected components of the induced subgraph on ``addedIds`` (edges with both ends in ``addedIds``). */
function incrementalAddedComponents(core: Core, addedIds: Set<string>): string[][] {
  const adj = new Map<string, string[]>()
  for (const id of addedIds) {
    adj.set(id, [])
  }
  core.edges().forEach((e) => {
    const s = e.source().id()
    const t = e.target().id()
    if (addedIds.has(s) && addedIds.has(t)) {
      adj.get(s)?.push(t)
      adj.get(t)?.push(s)
    }
  })
  const seen = new Set<string>()
  const comps: string[][] = []
  for (const id of addedIds) {
    if (seen.has(id)) {
      continue
    }
    const stack = [id]
    seen.add(id)
    const comp: string[] = []
    while (stack.length) {
      const u = stack.pop()!
      comp.push(u)
      for (const v of adj.get(u) ?? []) {
        if (!seen.has(v)) {
          seen.add(v)
          stack.push(v)
        }
      }
    }
    comps.push(comp)
  }
  return comps
}

function collectAnchorsForAddedComponent(
  core: Core,
  comp: string[],
  addedIds: Set<string>,
): { topicLike: CyModelPosition[]; otherFixed: CyModelPosition[] } {
  const topicLike: CyModelPosition[] = []
  const otherFixed: CyModelPosition[] = []
  const seenNeighbor = new Set<string>()
  for (const id of comp) {
    const n = core.$id(id)
    if (n.empty() || !n.isNode()) {
      continue
    }
    n.connectedEdges().forEach((e) => {
      const src = e.source()
      const tgt = e.target()
      const o = src.id() === n.id() ? tgt : src
      const oid = o.id()
      if (addedIds.has(oid)) {
        return
      }
      if (seenNeighbor.has(oid)) {
        return
      }
      seenNeighbor.add(oid)
      const t = String(o.data('type') ?? '')
      const p = o.position()
      if (t === 'Topic' || t === 'TopicCluster') {
        topicLike.push(p)
      } else {
        otherFixed.push(p)
      }
    })
  }
  return { topicLike, otherFixed }
}

/**
 * Seed model positions for newly appended nodes before a localized COSE pass.
 * Prefers Topic / TopicCluster neighbours already on the graph; otherwise uses other fixed neighbours;
 * disconnected new components go to the right of the existing bbox.
 */
function seedPositionsForIncrementalAppend(core: Core, addedIds: Set<string>): void {
  const bbox = modelBBoxOfFixedNodes(core, addedIds)
  const margin = 120
  const comps = incrementalAddedComponents(core, addedIds)
  let orphanComponentIndex = 0
  core.batch(() => {
    for (const comp of comps) {
      const { topicLike, otherFixed } = collectAnchorsForAddedComponent(core, comp, addedIds)
      let ax: number
      let ay: number
      if (topicLike.length) {
        const sx = topicLike.reduce((a, p) => a + p.x, 0)
        const sy = topicLike.reduce((a, p) => a + p.y, 0)
        ax = sx / topicLike.length
        ay = sy / topicLike.length
      } else if (otherFixed.length) {
        const sx = otherFixed.reduce((a, p) => a + p.x, 0)
        const sy = otherFixed.reduce((a, p) => a + p.y, 0)
        ax = sx / otherFixed.length
        ay = sy / otherFixed.length
      } else if (bbox) {
        ax = bbox.x2 + margin
        ay = (bbox.y1 + bbox.y2) / 2 + orphanComponentIndex * 140
        orphanComponentIndex += 1
      } else {
        ax = margin
        ay = margin + orphanComponentIndex * 140
        orphanComponentIndex += 1
      }

      comp.forEach((nid, idx) => {
        const nn = core.$id(nid)
        if (nn.empty() || !nn.isNode()) {
          return
        }
        const h = stableHashString(nid)
        const ang = ((h & 0xfffffff) / 0xfffffff) * Math.PI * 2
        const jr = 36 + (Math.abs(h) % 48)
        const jx = Math.cos(ang) * jr * 0.35 + ((idx * 13) % 56) - 28
        const jy = Math.sin(ang) * jr * 0.35 + ((idx * 17) % 56) - 28
        try {
          nn.position({ x: ax + jx, y: ay + jy })
        } catch {
          /* ignore */
        }
      })
    }
  })
}

function applyDegreeVisibility(core: Core): void {
  const bucket = activeDegreeBucket.value
  core.batch(() => {
    if (!bucket) {
      core.nodes().style('display', 'element')
      core.edges().style('display', 'element')
      return
    }
    for (const n of core.nodes()) {
      const deg = n.degree(false)
      const b = degreeBucketFor(deg)
      n.style('display', b === bucket ? 'element' : 'none')
    }
    for (const e of core.edges()) {
      const sh = e.source().style('display')
      const th = e.target().style('display')
      const hidden = sh === 'none' || th === 'none'
      e.style('display', hidden ? 'none' : 'element')
    }
  })
}

function recomputeDegreeHistogram(core: Core): void {
  const counts = emptyDegreeCounts()
  core.nodes().forEach((n) => {
    const d = n.degree(false)
    const b = degreeBucketFor(d)
    counts[b] += 1
  })
  degreeHistogramCounts.value = { ...counts }
}

function updateZoomPercentDisplay(core: Core): void {
  try {
    zoomPercent.value = Math.round(core.zoom() * 100)
  } catch {
    zoomPercent.value = 100
  }
}

function attachZoomRecenter(core: Core): void {
  if (zoomPanListenerAttached) {
    return
  }
  zoomPanListenerAttached = true
  lastZoomLevel = core.zoom()
  core.on('zoom', () => {
    if (!cy) return
    const c = cy
    const z = c.zoom()
    const prevZ = lastZoomLevel

    if (suspendSelectedNodeZoomAnchorCorrection > 0) {
      lastZoomLevel = z
      updateZoomPercentDisplay(c)
      syncGraphLabelTierClasses(c)
      return
    }

    const sid = selectedNodeId.value
    if (
      sid &&
      selectedNodeZoomAnchor &&
      nodeOkForViewportPreserve(c, sid)
    ) {
      const ratio = prevZ > 1e-9 ? z / prevZ : 1
      const incremental = ratio >= 0.65 && ratio <= 1.55
      if (incremental) {
        try {
          const rp = c.$id(sid).renderedPosition()
          c.panBy({
            x: selectedNodeZoomAnchor.x - rp.x,
            y: selectedNodeZoomAnchor.y - rp.y,
          })
        } catch {
          /* ignore */
        }
      }
      refreshSelectedNodeZoomAnchor(c)
    } else if (!sid) {
      clearSelectedNodeZoomAnchor()
    }

    const zoomedOut = z < prevZ - 1e-9
    lastZoomLevel = c.zoom()
    updateZoomPercentDisplay(c)
    syncGraphLabelTierClasses(c)
    if (!zoomedOut) return
    if (selectedNodeId.value) {
      return
    }
    if (zoomCenterTimer != null) {
      clearTimeout(zoomCenterTimer)
    }
    zoomCenterTimer = setTimeout(() => {
      zoomCenterTimer = null
      if (!cy) return
      if (selectedNodeId.value) {
        return
      }
      try {
        cy.center(cy.elements(':visible'))
      } catch {
        /* ignore */
      }
    }, 160)
  })

  core.on('pan', () => {
    if (!cy || suspendSelectedNodeZoomAnchorCorrection > 0) return
    if (selectedNodeId.value) {
      refreshSelectedNodeZoomAnchor(cy)
    }
  })
}

/**
 * Minimap uses cytoscape-navigator, which calls `cy.png({ full: true })` on a throttled
 * `onRender` hook — expensive on large graphs. Schedule it only after the main graph is
 * visible so first paint is not blocked behind that export.
 */
function scheduleMinimapSetup(core: Core): void {
  if (!minimapOpen.value) {
    return
  }
  void nextTick(() => {
    if (!minimapOpen.value || !cy || cy !== core) {
      return
    }
    setupNavigator(core)
  })
}

function releaseGraphPaintAfterLayout(core: Core): void {
  if (!graphContentHiddenUntilLayout.value) {
    scheduleMinimapSetup(core)
    return
  }
  void nextTick(() => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (!cy || cy !== core) {
          pendingFocusCameraAfterLayoutHold = null
          return
        }
        releaseGraphCanvasLayoutHold()
        scheduleMinimapSetup(core)
      })
    })
  })
}

function finishLayoutPass(core: Core): void {
  if (!cy || cy !== core) {
    return
  }
  const snap = pendingViewportPreserve
  pendingViewportPreserve = null

  recomputeDegreeHistogram(cy)
  applyTopicDegreeHeat(cy)
  applyDegreeVisibility(cy)
  applyViewportPreserveOrFit(cy, snap)

  lastZoomLevel = cy.zoom()
  updateZoomPercentDisplay(cy)
  syncGraphLabelTierClasses(cy)
  attachZoomRecenter(core)
  applySearchHighlights(core)
  /** ``tryApplyPendingFocus`` first so Library/Digest **Open in graph** camera wins; episode strip skips
   * duplicate ``animateCameraToFocusedNode`` when pending focus was consumed. */
  const appliedPending = tryApplyPendingFocus(core)
  applyEpisodeRepresentativeFocusIfNeeded(core, { skipCamera: appliedPending })
  reapplySelectionDimmingIfAny(core)
  applyTopicClusterMemberCollapse(core)
  applyCrossEpisodeExpandHints(core)
  scheduleNodeEpisodesCorpusBeyondProbes()
  const viewArt = gf.viewWithEgo(focusNodeId.value)
  if (viewArt) {
    lastCommittedFilteredNodeIds = new Set(
      toGraphElements(viewArt).visNodes.map((v) => v.id),
    )
  } else {
    lastCommittedFilteredNodeIds.clear()
  }
  lastSelectedRelPathsCountAfterLayout = artifacts.selectedRelPaths.length
  lastCommittedEgoFocusCyId = focusNodeId.value?.trim() ?? ''
  releaseGraphPaintAfterLayout(core)
}

function applyCrossEpisodeExpandHints(core: Core): void {
  if (!cy || cy !== core) {
    return
  }
  syncCrossEpisodeExpandNodeClasses(core, {
    isExpandedSeed: (id) => graphExpansion.isExpanded(id),
    rawNode: rawNodeForRailInteraction,
    corpusWouldAppendOutsideGraph: (id) => graphExpansion.corpusBeyondAppendKnown(id),
  })
}

async function runNodeEpisodesCorpusBeyondProbes(): Promise<void> {
  const core = cy
  if (!core) {
    return
  }
  const waveGen = graphExpansion.peekCorpusBeyondProbeGen()
  const root = shell.corpusPath.trim()
  if (!root || !shell.corpusLibraryApiAvailable) {
    return
  }
  const selected = artifacts.selectedRelPaths
  const candidates: string[] = []
  core.nodes().forEach((ele) => {
    try {
      if (!ele || typeof ele.isNode !== 'function' || !ele.isNode()) {
        return
      }
      const id = ele.id()
      if (graphExpansion.isExpanded(id)) {
        return
      }
      const raw = rawNodeForRailInteraction(id)
      if (!graphNodeExpandableForCrossEpisodeExpand(core, id, raw)) {
        return
      }
      if (graphExpansion.corpusBeyondAppendKnown(id) !== undefined) {
        return
      }
      candidates.push(id)
    } catch {
      /* ignore */
    }
  })
  for (const id of candidates) {
    if (!cy || cy !== core) {
      return
    }
    if (graphExpansion.peekCorpusBeyondProbeGen() !== waveGen) {
      return
    }
    try {
      const res = await fetchNodeEpisodes(root, id, GRAPH_NODE_EPISODES_EXPAND_MAX)
      const would = wouldCrossEpisodeExpandAppendNewArtifacts(res.episodes, selected)
      graphExpansion.commitCorpusBeyondProbe(waveGen, id, would)
    } catch {
      graphExpansion.commitCorpusBeyondProbe(waveGen, id, false)
    }
    if (cy === core) {
      applyCrossEpisodeExpandHints(core)
    }
  }
}

function scheduleNodeEpisodesCorpusBeyondProbes(): void {
  if (nodeEpisodesCorpusBeyondDebounce != null) {
    clearTimeout(nodeEpisodesCorpusBeyondDebounce)
  }
  nodeEpisodesCorpusBeyondDebounce = setTimeout(() => {
    nodeEpisodesCorpusBeyondDebounce = null
    void runNodeEpisodesCorpusBeyondProbes()
  }, 400)
}

/** Hide member Topic nodes inside collapsed TopicCluster compounds (detail rail toggle). */
function applyTopicClusterMemberCollapse(core: Core): void {
  const collapsed = nav.topicClusterCanvasCollapsedIds
  core.batch(() => {
    core.nodes('[parent]').forEach((ele) => {
      try {
        const p = ele.data('parent')
        if (typeof p !== 'string' || !p.trim()) {
          return
        }
        const hide = collapsed.includes(p.trim())
        ele.style('display', hide ? 'none' : 'element')
      } catch {
        /* ignore */
      }
    })
  })
}

/* RFC-080 V4 — radial focus mode state. The mode is enter/exit only;
 * snapshots capture node positions + per-element display so an exit
 * restores the graph exactly. Held at module scope (not pinia) because
 * the snapshot is meaningful only against the current `cy` instance —
 * a destroy + redraw invalidates it. */
const radialModeActive = ref(false)
const radialAriaMessage = ref('')
let radialSnapshot: RadialSnapshot | null = null

function enterRadialMode(centreId: string): boolean {
  const c = cy
  if (!c) return false
  const centre = c.$id(centreId)
  if (centre.empty() || !centre.isNode()) return false

  // Ring 1 = 1-hop neighbour nodes; ring 2 = 2-hop minus ring 1 minus
  // the centre itself. Compound (TopicCluster) members participate in
  // ring 1 alongside external 1-hop neighbours per RFC-080 V4.
  const ring1 = centre.neighborhood('node').union(centre.children('node'))
  const ring2 = ring1.neighborhood('node').difference(ring1).difference(centre)
  const ring1Ids = ring1.map((n) => n.id())
  const ring2Ids = ring2.map((n) => n.id())

  // V5 interaction: ring radius adapts to the largest ring-1 node radius
  // so size-by-degree doesn't push neighbours into each other.
  let maxR1Radius = 0
  ring1.forEach((n) => {
    const r = (n.width() ?? 0) / 2
    if (r > maxR1Radius) maxR1Radius = r
  })
  const out = computeRadialPositions(centreId, ring1Ids, ring2Ids, {
    maxRing1NodeRadius: maxR1Radius,
  })

  // Snapshot positions + display state for clean restore on exit. We
  // walk all nodes / edges (not just the visible set) because outer-
  // hop elements get hidden as part of entering the mode.
  const positions: Record<string, { x: number; y: number }> = {}
  const displays: Record<string, string> = {}
  c.nodes().forEach((n) => {
    const p = n.position()
    positions[n.id()] = { x: p.x, y: p.y }
    displays[n.id()] = String(n.style('display') ?? 'element')
  })
  const edgeDisplays: Record<string, string> = {}
  c.edges().forEach((e) => {
    edgeDisplays[e.id()] = String(e.style('display') ?? 'element')
  })
  radialSnapshot = { positions, displays, edgeDisplays, centreId }

  // Hide everything outside ring 1 ∪ ring 2 ∪ centre. Edges with
  // either endpoint hidden also disappear so we don't render orphan
  // strokes.
  const visibleSet = new Set<string>([centreId, ...ring1Ids, ...ring2Ids])
  c.batch(() => {
    c.nodes().forEach((n) => {
      n.style('display', visibleSet.has(n.id()) ? 'element' : 'none')
    })
    c.edges().forEach((e) => {
      const ok = visibleSet.has(e.source().id()) && visibleSet.has(e.target().id())
      e.style('display', ok ? 'element' : 'none')
    })
  })

  c.layout({
    name: 'preset',
    positions: (n: NodeSingular) => {
      const p = out.positions[n.id()]
      return p ? { x: p.x, y: p.y } : { x: 0, y: 0 }
    },
    fit: true,
    padding: 60,
  } as never).run()

  radialModeActive.value = true
  // a11y: announce centre node label (the user lost the mouse-context
  // when the canvas reorganised; SR / keyboard users get a verbal
  // reference).
  const label = String(centre.data('label') ?? centreId)
  radialAriaMessage.value = `Radial view centred on ${label}.`
  return true
}

function exitRadialMode(): void {
  const c = cy
  if (!c || !radialSnapshot) {
    radialModeActive.value = false
    radialSnapshot = null
    radialAriaMessage.value = ''
    return
  }
  const snap = radialSnapshot
  c.batch(() => {
    c.nodes().forEach((n) => {
      const p = snap.positions[n.id()]
      if (p) n.position({ x: p.x, y: p.y })
      const d = snap.displays[n.id()]
      n.style('display', d ?? 'element')
    })
    c.edges().forEach((e) => {
      const d = snap.edgeDisplays[e.id()]
      e.style('display', d ?? 'element')
    })
  })
  radialAriaMessage.value = `Radial view exited.`
  radialModeActive.value = false
  radialSnapshot = null
}

function destroyCy(): void {
  // Do not clear pendingViewportPreserve here — it must survive destroy+redraw until the new cy's layoutstop
  // (ego exit to full graph, Re-layout). egoPriorFullGraphViewportPreserve must survive enter-ego redraw too.
  pendingFocusCameraAfterLayoutHold = null
  // RFC-080 V4: snapshot is bound to the cy instance; a destroy invalidates it.
  // Clear so a subsequent enter starts from a clean state instead of restoring
  // stale positions onto a freshly mounted graph.
  radialSnapshot = null
  radialModeActive.value = false
  radialAriaMessage.value = ''
  graphLayoutGate.invalidate()
  clearSelectedNodeZoomAnchor()
  suspendSelectedNodeZoomAnchorCorrection = 0
  zoomPanListenerAttached = false
  teardownBoxZoomListeners()
  destroyNavigator()
  if (zoomCenterTimer != null) {
    clearTimeout(zoomCenterTimer)
    zoomCenterTimer = null
  }
  if (resizeObs) {
    resizeObs.disconnect()
    resizeObs = null
  }
  if (activeElesLayout) {
    try {
      activeElesLayout.stop()
    } catch {
      /* ignore */
    }
    activeElesLayout = null
  }
  if (import.meta.env.DEV && cy) {
    const w = window as unknown as { __GIKG_CY_DEV__?: Core }
    if (w.__GIKG_CY_DEV__ === cy) {
      delete w.__GIKG_CY_DEV__
    }
  }
  if (cy) {
    try {
      cy.destroy()
    } catch {
      /* ignore */
    }
    cy = null
  }
}

/**
 * Pan/zoom so ``cyId`` is centered (at least ``GRAPH_FOCUS_FRAME_MIN_ZOOM``), optionally framing extra ids —
 * used by ``tryApplyPendingFocus`` (search / digest handoff) and by single-tap graph picks so behaviour matches.
 */
function animateCameraToFocusedNode(
  core: Core,
  cyId: string,
  opts?: { extraRawIds?: readonly string[] | null },
): void {
  const id = cyId.trim()
  if (!id) return
  const n = core.$id(id)
  if (n.empty()) return
  suspendSelectedNodeZoomAnchorCorrection += 1
  try {
    const targetZoom = Math.max(core.zoom(), GRAPH_FOCUS_FRAME_MIN_ZOOM)
    let centerEles = n
    const extras = opts?.extraRawIds
    if (Array.isArray(extras) && extras.length) {
      for (const raw of extras) {
        if (typeof raw !== 'string' || !raw.trim()) continue
        const exId = resolveCyNodeId(core, raw.trim())
        if (!exId) continue
        const en = core.$id(exId)
        if (!en.empty()) {
          centerEles = centerEles.union(en)
        }
      }
    }
    core.animate({
      center: { eles: centerEles },
      zoom: targetZoom,
      duration: 320,
      complete: () => {
        suspendSelectedNodeZoomAnchorCorrection -= 1
        refreshSelectedNodeZoomAnchor(core)
        lastZoomLevel = core.zoom()
        updateZoomPercentDisplay(core)
      },
    })
  } catch {
    suspendSelectedNodeZoomAnchorCorrection -= 1
    try {
      core.center(n)
    } catch {
      /* ignore */
    }
    refreshSelectedNodeZoomAnchor(core)
    lastZoomLevel = core.zoom()
    updateZoomPercentDisplay(core)
  }
}

/** @returns ``true`` when pending focus was applied (camera + selection); caller may skip episode-strip camera. */
function tryApplyPendingFocus(core: Core): boolean {
  const rawId = nav.pendingFocusNodeId
  if (!rawId) return false
  const fallbackRaw = nav.pendingFocusFallbackNodeId
  let cyId = resolveCyNodeId(core, rawId)
  if (!cyId && fallbackRaw) {
    cyId = resolveCyNodeId(core, fallbackRaw)
  }
  /** Do not clear pending: ``redraw`` can leave the graph mid-rebuild; a later ``finishLayoutPass`` / watcher applies. */
  if (!cyId) {
    return false
  }
  const n = core.$id(cyId)
  if (n.empty()) {
    return false
  }
  const focusNode = n.first() as NodeSingular
  core.nodes().unselect()
  focusNode.select()
  selectedNodeId.value = cyId
  try {
    applyGraphSelectionDimFromNode(core, focusNode)
  } catch {
    /* ignore */
  }
  const rawNode = rawNodeForRailInteraction(cyId)
  void openGraphEpisodeOrNodeRail(cyId, rawNode)
  const extras = [...nav.pendingFocusCameraIncludeRawIds]
  nav.clearPendingFocus()
  /** While the canvas host is hidden for first paint, ``animate`` uses wrong bounds; defer until ``releaseGraphCanvasLayoutHold``. */
  if (graphContentHiddenUntilLayout.value) {
    pendingFocusCameraAfterLayoutHold = { cyId, extras }
  } else {
    animateCameraToFocusedNode(core, cyId, {
      extraRawIds: extras.length ? extras : undefined,
    })
  }
  return true
}

/** Re-apply neighbourhood dimming when ``select`` handlers were not wired yet or async rail races layout. */
function reapplySelectionDimmingIfAny(core: Core): void {
  const sid = selectedNodeId.value?.trim()
  if (!sid) return
  try {
    const n = core.$id(sid)
    if (n.empty()) return
    applyGraphSelectionDimFromNode(core, n.first() as NodeSingular)
  } catch {
    /* ignore */
  }
}

function fitAnimated(): void {
  const c = cy
  if (!c) return
  const els = c.elements(':visible')
  if (els.length === 0) return
  suspendSelectedNodeZoomAnchorCorrection += 1
  try {
    c.animate({
      fit: { eles: els, padding: 24 },
      duration: 280,
      complete: () => {
        suspendSelectedNodeZoomAnchorCorrection -= 1
        refreshSelectedNodeZoomAnchor(c)
        lastZoomLevel = c.zoom()
        updateZoomPercentDisplay(c)
      },
    })
  } catch {
    suspendSelectedNodeZoomAnchorCorrection -= 1
    try {
      c.fit(els, 24)
    } catch {
      /* ignore */
    }
    refreshSelectedNodeZoomAnchor(c)
    lastZoomLevel = c.zoom()
    updateZoomPercentDisplay(c)
  }
}

function zoomIn(): void {
  const c = cy
  if (!c) return
  try {
    const z = Math.min(c.zoom() * 1.2, c.maxZoom())
    c.zoom(z)
  } catch {
    /* ignore */
  }
}

function zoomOut(): void {
  const c = cy
  if (!c) return
  try {
    const z = Math.max(c.zoom() / 1.2, c.minZoom())
    c.zoom(z)
  } catch {
    /* ignore */
  }
}

function zoomReset100(): void {
  const c = cy
  if (!c) return
  const els = c.elements(':visible')
  if (els.length === 0) return
  if (zoomCenterTimer != null) {
    clearTimeout(zoomCenterTimer)
    zoomCenterTimer = null
  }
  suspendSelectedNodeZoomAnchorCorrection += 1
  try {
    c.zoom(1)
    c.center(els)
  } catch {
    /* ignore */
  } finally {
    suspendSelectedNodeZoomAnchorCorrection -= 1
    try {
      refreshSelectedNodeZoomAnchor(c)
    } catch {
      /* ignore */
    }
    lastZoomLevel = c.zoom()
    updateZoomPercentDisplay(c)
  }
}

function canvasExportBg(): string {
  try {
    const v = getComputedStyle(document.documentElement)
      .getPropertyValue('--ps-canvas')
      .trim()
    if (v.length) {
      return v
    }
  } catch {
    /* ignore */
  }
  return '#111418'
}

function clearInteractionState(opts?: { skipRedraw?: boolean }): void {
  clearEpisodeRepresentativeGraphState(cy)
  nav.clearPendingFocus()
  nav.clearLibraryEpisodeHighlights()
  nav.clearTopicClusterCanvasCollapsed()
  clearSelectedNodeZoomAnchor()
  const hadEgo = focusNodeId.value !== null
  if (hadEgo) {
    pendingViewportPreserve = egoPriorFullGraphViewportPreserve
    egoPriorFullGraphViewportPreserve = null
  }
  focusNodeId.value = null
  selectedNodeId.value = null
  subject.clearSubject()
  const c = cy
  if (c) {
    try {
      c.nodes().unselect()
    } catch {
      /* ignore */
    }
  }
  if (hadEgo && !opts?.skipRedraw) {
    redraw()
  }
}

function exportGraphPng(): void {
  const c = cy
  if (!c) return
  const els = c.elements(':visible')
  if (els.length === 0) return
  let uri: string
  try {
    uri = c.png({
      output: 'base64uri',
      full: true,
      scale: 2,
      bg: canvasExportBg(),
    })
  } catch {
    return
  }
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')
  const a = document.createElement('a')
  a.href = uri
  a.download = `gi-kg-viewer-${stamp}.png`
  a.rel = 'noopener'
  document.body.appendChild(a)
  a.click()
  a.remove()
}

/** Model-space bbox for the shift-drag rectangle (matches Cytoscape container client coords). */
function modelBoundingBoxFromClientRect(
  core: Core,
  cx1: number,
  cy1: number,
  cx2: number,
  cy2: number,
): { x1: number; y1: number; x2: number; y2: number } | null {
  const r = (core as unknown as {
    renderer: () => {
      projectIntoViewport: (clientX: number, clientY: number) => [number, number]
      invalidateContainerClientCoordsCache?: () => void
    }
  }).renderer()
  try {
    r.invalidateContainerClientCoordsCache?.()
  } catch {
    /* ignore */
  }
  const minX = Math.min(cx1, cx2)
  const maxX = Math.max(cx1, cx2)
  const minY = Math.min(cy1, cy2)
  const maxY = Math.max(cy1, cy2)
  const corners: Array<[number, number]> = [
    [minX, minY],
    [maxX, minY],
    [maxX, maxY],
    [minX, maxY],
  ]
  let x1 = Infinity
  let y1 = Infinity
  let x2 = -Infinity
  let y2 = -Infinity
  for (const [cx, cy] of corners) {
    let mx: number
    let my: number
    try {
      ;[mx, my] = r.projectIntoViewport(cx, cy)
    } catch {
      return null
    }
    if (!Number.isFinite(mx) || !Number.isFinite(my)) {
      return null
    }
    x1 = Math.min(x1, mx)
    y1 = Math.min(y1, my)
    x2 = Math.max(x2, mx)
    y2 = Math.max(y2, my)
  }
  if (!(x2 > x1 && y2 > y1)) {
    return null
  }
  return { x1, y1, x2, y2 }
}

function onWindowMouseMove(ev: MouseEvent): void {
  if (!boxDragging || !canvasHost.value) return
  const br = canvasHost.value.getBoundingClientRect()
  const x1 = boxStartClient.x - br.left
  const y1 = boxStartClient.y - br.top
  const x2 = ev.clientX - br.left
  const y2 = ev.clientY - br.top
  boxZoomRect.left = Math.min(x1, x2)
  boxZoomRect.top = Math.min(y1, y2)
  boxZoomRect.width = Math.abs(x2 - x1)
  boxZoomRect.height = Math.abs(y2 - y1)
  boxZoomRect.show = boxZoomRect.width > 4 && boxZoomRect.height > 4
}

function onWindowMouseUp(ev: MouseEvent): void {
  if (!boxDragging) return
  boxDragging = false
  const c = cy
  const host = canvasHost.value
  boxZoomRect.show = false
  if (c && host) {
    const br = host.getBoundingClientRect()
    const rx1 = boxStartClient.x - br.left
    const ry1 = boxStartClient.y - br.top
    const rx2 = ev.clientX - br.left
    const ry2 = ev.clientY - br.top
    if (Math.abs(rx2 - rx1) > 6 && Math.abs(ry2 - ry1) > 6) {
      const modelBb = modelBoundingBoxFromClientRect(
        c,
        boxStartClient.x,
        boxStartClient.y,
        ev.clientX,
        ev.clientY,
      )
      if (modelBb) {
        suspendSelectedNodeZoomAnchorCorrection += 1
        try {
          // fit() uses getFitViewport(), which returns nothing while panning is disabled — we
          // turn panning off during shift-drag, so re-enable before fit.
          try {
            c.panningEnabled(true)
            c.zoomingEnabled(true)
          } catch {
            /* ignore */
          }
          // Cytoscape accepts a plain { x1,y1,x2,y2 } bbox in model space (undocumented in types).
          ;(c as unknown as { fit: (target: unknown, padding?: number) => void }).fit(modelBb, 32)
        } catch {
          /* ignore */
        } finally {
          suspendSelectedNodeZoomAnchorCorrection -= 1
          refreshSelectedNodeZoomAnchor(c)
          lastZoomLevel = c.zoom()
          updateZoomPercentDisplay(c)
        }
      }
    }
  }
  try {
    c?.panningEnabled(true)
    c?.zoomingEnabled(true)
  } catch {
    /* ignore */
  }
}

function teardownBoxZoomListeners(): void {
  if (!boxListenersAttached) return
  window.removeEventListener('mousemove', onWindowMouseMove)
  window.removeEventListener('mouseup', onWindowMouseUp)
  boxListenersAttached = false
  boxDragging = false
  boxZoomRect.show = false
}

function setupBoxZoomListeners(): void {
  if (boxListenersAttached) return
  window.addEventListener('mousemove', onWindowMouseMove)
  window.addEventListener('mouseup', onWindowMouseUp)
  boxListenersAttached = true
}

function runRelayout(): void {
  const c = cy
  if (!c) return
  const eles = c.elements(':visible')
  if (eles.length === 0) return

  const gen = graphLayoutGate.bump()
  if (activeElesLayout) {
    try {
      activeElesLayout.stop()
    } catch {
      /* ignore */
    }
    activeElesLayout = null
  }

  graphContentHiddenUntilLayout.value = true
  const graphEl = container.value
  if (graphEl) {
    applyGraphCanvasImmediateHide(graphEl)
  }

  pendingViewportPreserve = captureSelectedViewportAnchor(c)
  const name = preferredLayout.value
  const opts = layoutOptionsFor(name)
  let lo: { stop: () => void; one: (ev: string, fn: () => void) => void; run: () => void }
  try {
    lo = eles.layout({
      ...opts,
      name,
    } as never) as typeof lo
  } catch {
    releaseGraphCanvasLayoutHold()
    return
  }
  activeElesLayout = lo
  lo.one('layoutstop', () => {
    if (graphLayoutGate.isStale(gen)) {
      return
    }
    if (activeElesLayout === lo) {
      activeElesLayout = null
    }
    if (!cy || cy !== c) {
      releaseGraphCanvasLayoutHold()
      return
    }
    finishLayoutPass(cy)
  })
  try {
    lo.run()
  } catch {
    activeElesLayout = null
    releaseGraphCanvasLayoutHold()
  }
}

function redraw(): void {
  if (redrawGateDepth > 0) {
    redrawPending = true
    return
  }
  redrawGateDepth += 1
  try {
    graphContentHiddenUntilLayout.value = true

    const priorCore = cy
    const viewArtPreview = gf.viewWithEgo(focusNodeId.value)
    const nextNodeIdSet = new Set<string>()
    if (viewArtPreview) {
      for (const v of toGraphElements(viewArtPreview).visNodes) {
        nextNodeIdSet.add(v.id)
      }
    }
    const selCount = artifacts.selectedRelPaths.length
    const egoCur = focusNodeId.value?.trim() ?? ''
    const selectionGrew =
      selCount > lastSelectedRelPathsCountAfterLayout &&
      lastSelectedRelPathsCountAfterLayout > 0
    const egoUnchanged = egoCur === lastCommittedEgoFocusCyId
    const addedNodeIds = new Set<string>()
    if (selectionGrew && egoUnchanged && priorCore && lastCommittedFilteredNodeIds.size > 0) {
      for (const id of nextNodeIdSet) {
        if (!lastCommittedFilteredNodeIds.has(id)) {
          addedNodeIds.add(id)
        }
      }
    }

    const preservedPositions = new Map<string, CyModelPosition>()
    let useIncrementalLayout = false
    if (
      addedNodeIds.size > 0 &&
      selectionGrew &&
      egoUnchanged &&
      priorCore &&
      lastCommittedFilteredNodeIds.size > 0
    ) {
      priorCore.nodes().forEach((n) => {
        const id = n.id()
        if (!addedNodeIds.has(id)) {
          preservedPositions.set(id, { ...n.position() })
        }
      })
      useIncrementalLayout = preservedPositions.size > 0
    }

    destroyCy()
    const el = container.value
    if (!el) {
      graphCyNodeCount.value = 0
      releaseGraphCanvasLayoutHold()
      lastSelectedRelPathsCountAfterLayout = 0
      lastCommittedFilteredNodeIds.clear()
      lastCommittedEgoFocusCyId = ''
      return
    }

    const art = gf.viewWithEgo(focusNodeId.value)
    if (!art) {
      el.innerHTML =
        '<p class="p-4 text-sm text-muted">Load artifacts and use “Load selected” to render the graph.</p>'
      degreeHistogramCounts.value = {}
      graphCyNodeCount.value = 0
      releaseGraphCanvasLayoutHold()
      lastSelectedRelPathsCountAfterLayout = 0
      lastCommittedFilteredNodeIds.clear()
      lastCommittedEgoFocusCyId = ''
      return
    }

    // RFC-080 V1 — append render-only Episode↔Topic / Episode↔Person
    // aggregated edges when the lens flag is on. Defaults off; takes
    // effect on the next graph rebuild.
    const elements = toCytoElements(art, {
      enableAggregatedEdges: lenses.aggregatedEdges,
    })
    const nodeCount = elements.filter((x) => !('source' in x.data)).length
    if (nodeCount === 0) {
      el.innerHTML =
        '<p class="p-4 text-sm text-muted">No nodes in this view (adjust filters).</p>'
      degreeHistogramCounts.value = {}
      graphCyNodeCount.value = 0
      releaseGraphCanvasLayoutHold()
      lastSelectedRelPathsCountAfterLayout = 0
      lastCommittedFilteredNodeIds.clear()
      lastCommittedEgoFocusCyId = ''
      return
    }

    graphCyNodeCount.value = nodeCount
    el.innerHTML = ''
    applyGraphCanvasImmediateHide(el)
    const layoutName = preferredLayout.value
    const layoutOpts = layoutOptionsFor(layoutName)
    const core = cytoscape({
      container: el,
      elements,
      style: buildCyStyle(),
      wheelSensitivity: 0.35,
    })
    cy = core
    if (import.meta.env.DEV) {
      ;(window as unknown as { __GIKG_CY_DEV__?: Core }).__GIKG_CY_DEV__ = core
    }

    if (useIncrementalLayout) {
      core.batch(() => {
        for (const [id, pos] of preservedPositions) {
          const n = core.$id(id)
          if (!n.empty()) {
            n.position(pos)
          }
        }
      })
      seedPositionsForIncrementalAppend(core, addedNodeIds)
    }

    const layoutGen = graphLayoutGate.bump()
    let initialLo: { stop: () => void; one: (ev: string, fn: () => void) => void; run: () => void }
    const layoutCollection = useIncrementalLayout
      ? core.elements().filter((ele) => {
          if (ele.isNode()) {
            return addedNodeIds.has(ele.id())
          }
          return (
            addedNodeIds.has(ele.source().id()) || addedNodeIds.has(ele.target().id())
          )
        })
      : core.elements()
    try {
      initialLo = layoutCollection.layout({
        ...layoutOpts,
        name: layoutName,
      } as never) as typeof initialLo
    } catch {
      cy = null
      try {
        core.destroy()
      } catch {
        /* ignore */
      }
      graphCyNodeCount.value = 0
      releaseGraphCanvasLayoutHold()
      return
    }
    activeElesLayout = initialLo
    initialLo.one('layoutstop', () => {
      if (graphLayoutGate.isStale(layoutGen)) {
        return
      }
      if (activeElesLayout === initialLo) {
        activeElesLayout = null
      }
      if (!cy || cy !== core) {
        // Stale stop after destroy/redraw — avoid leaving the canvas host frozen (pointer-events).
        releaseGraphCanvasLayoutHold()
        return
      }
      finishLayoutPass(core)
    })
    try {
      initialLo.run()
    } catch {
      activeElesLayout = null
      cy = null
      try {
        core.destroy()
      } catch {
        /* ignore */
      }
      graphCyNodeCount.value = 0
      releaseGraphCanvasLayoutHold()
      return
    }

    const sync = (): void => {
      if (cy === core) core.resize()
    }
    if (typeof ResizeObserver !== 'undefined') {
      resizeObs = new ResizeObserver(sync)
      resizeObs.observe(el)
    }
    requestAnimationFrame(sync)
    requestAnimationFrame(() => requestAnimationFrame(sync))

    setupBoxZoomListeners()

    /** Single-select UX: tap flow clears other nodes; dimming assumes one focused node (see WIP §3.3). */
    core.on('select', 'node', (e) => {
      try {
        applyGraphSelectionDimFromNode(core, e.target as NodeSingular)
      } catch {
        /* ignore */
      }
    })
    core.on('unselect', 'node', () => {
      try {
        const sel = core.nodes(':selected')
        if (sel.length === 0) {
          clearGraphSelectionDim(core)
        } else {
          applyGraphSelectionDimFromNode(core, sel.first() as NodeSingular)
        }
      } catch {
        /* ignore */
      }
    })

    core.on('tap', (evt) => {
    const t = evt.target
    if (t === core) {
      clearEpisodeRepresentativeGraphState(core)
      core.nodes().unselect()
      selectedNodeId.value = null
      subject.clearSubject()
      clearSelectedNodeZoomAnchor()
      return
    }
    if (typeof t.isNode === 'function' && t.isNode()) {
      core.nodes().unselect()
      t.select()
      selectedNodeId.value = t.id()
      refreshSelectedNodeZoomAnchor(core)
      return
    }
    selectedNodeId.value = null
    subject.clearSubject()
    clearSelectedNodeZoomAnchor()
  })

  /** Single-tap rail: ``onetap`` debounces so ``dbltap`` expand does not open rail first. */
  core.on('onetap', (evt) => {
    const t = evt.target
    if (typeof t.isNode === 'function' && t.isNode()) {
      const id = t.id()
      const rawNode = rawNodeForRailInteraction(id)
      void openGraphEpisodeOrNodeRail(id, rawNode)
      /** Match ``tryApplyPendingFocus`` / digest episode handoff: centre the tapped node in view. */
      animateCameraToFocusedNode(core, id)
    }
  })

  core.on('dbltap', (evt) => {
    const raw = evt.originalEvent as MouseEvent | TouchEvent | undefined
    const shift =
      raw && 'shiftKey' in raw ? Boolean((raw as MouseEvent).shiftKey) : false
    const t = evt.target
    if (typeof t.isNode === 'function' && t.isNode()) {
      const id = t.id()
      if (shift) {
        const c = cy
        const cur = focusNodeId.value
        if (cur === id) {
          pendingViewportPreserve = egoPriorFullGraphViewportPreserve
          egoPriorFullGraphViewportPreserve = null
          focusNodeId.value = null
        } else {
          if (cur === null && c) {
            const snap = captureViewportAnchorForCyId(c, id)
            if (snap) {
              egoPriorFullGraphViewportPreserve = snap
            }
          }
          focusNodeId.value = id
        }
        redraw()
        return
      }
      core.nodes().unselect()
      t.select()
      selectedNodeId.value = id
      void expandOrCollapseGraphNode(core, id)
      refreshSelectedNodeZoomAnchor(core)
      return
    }
    if (focusNodeId.value !== null) {
      pendingViewportPreserve = egoPriorFullGraphViewportPreserve
      egoPriorFullGraphViewportPreserve = null
      focusNodeId.value = null
      redraw()
    }
  })

    core.on('mousedown', (evt) => {
      if (evt.target !== core) return
      const o = evt.originalEvent as MouseEvent | undefined
      if (!o?.shiftKey) return
      o.preventDefault()
      boxDragging = true
      boxStartClient = { x: o.clientX, y: o.clientY }
      try {
        core.panningEnabled(false)
      } catch {
        /* ignore */
      }
    })

    const sid = selectedNodeId.value
    if (sid) {
      const n = core.$id(sid)
      if (n.empty()) {
        selectedNodeId.value = null
        subject.clearSubject()
        clearSelectedNodeZoomAnchor()
      } else {
        core.nodes().unselect()
        n.select()
        try {
          applyGraphSelectionDimFromNode(core, n.first() as NodeSingular)
        } catch {
          /* ignore */
        }
      }
    }
  } finally {
    redrawGateDepth -= 1
    if (redrawPending) {
      redrawPending = false
      void nextTick(() => {
        redraw()
      })
    }
  }
}

const typeHistogramCounts = computed(() => {
  const art = gf.viewWithEgo(focusNodeId.value)
  const nodes = art?.data.nodes
  if (!nodes) return {} as Record<string, number>
  return visualNodeTypeCounts(nodes as RawGraphNode[])
})

const typeFilterKeys = computed(() => {
  const st = gf.state
  if (!st) return [] as string[]
  return Object.keys(st.allowedTypes).sort()
})

/** Avoid Vue “unhandled error in watcher” / uncaught promise if cytoscape callbacks throw. */
function safeGraphWatch(label: string, fn: () => void): void {
  try {
    fn()
  } catch (e) {
    console.warn(`[GraphCanvas] watcher (${label}):`, e)
  }
}

function onCycleLayout(): void {
  ge.cyclePreferredLayout()
  runRelayout()
}

function reopenGestureOverlay(): void {
  gestureOverlayRef.value?.reopen()
}

function hideMinimap(): void {
  ge.minimapOpen = false
}

watch(
  () => artifacts.selectedRelPaths.length,
  (len) => {
    if (len === 0) {
      graphExpansion.resetExpansionState()
    }
  },
)

const corpusSelectionProbeKey = computed(() =>
  [...artifacts.selectedRelPaths]
    .map((p) => p.replace(/\\/g, '/').trim())
    .filter(Boolean)
    .sort()
    .join('\n'),
)

watch(corpusSelectionProbeKey, () => {
  graphExpansion.invalidateCorpusBeyondHints()
  scheduleNodeEpisodesCorpusBeyondProbes()
})

watch(
  () => [shell.corpusPath, shell.corpusLibraryApiAvailable] as const,
  () => {
    graphExpansion.invalidateCorpusBeyondHints()
    scheduleNodeEpisodesCorpusBeyondProbes()
  },
)

watch(
  () => [searchStore.results, nav.libraryHighlightSourceIds] as const,
  () => {
    safeGraphWatch('searchHighlights', () => {
      const c = cy
      if (c) applySearchHighlights(c)
    })
  },
  { flush: 'post', deep: true },
)

watch(
  focusNodeId,
  (v) => {
    safeGraphWatch('egoFocus', () => {
      nav.setGraphEgoFocusCyId(v)
    })
  },
  { immediate: true },
)

watch(
  () => gf.filteredArtifact,
  () => {
    void nextTick(() => {
      safeGraphWatch('filteredArtifact', () => {
        ge.resetForNewArtifact()
        focusNodeId.value = null
        selectedNodeId.value = null
        // Keep the subject rail on **Episode** (Library / Digest) or **Graph node** (e.g. TopicCluster
        // detail). Otherwise clearing the subject wipes ids and feels like Search replaced detail — bad
        // when the graph reloads after **Load** on a cluster member (append artifacts) or
        // **Open in graph**: user expects panels to stay and only the canvas to expand.
        const keepDetailRailOpen =
          subject.kind === 'episode' || subject.kind === 'graph-node'
        if (!keepDetailRailOpen) {
          subject.clearSubject()
        }
        nav.clearLibraryEpisodeHighlights()
        nav.clearTopicClusterCanvasCollapsed()
        pendingViewportPreserve = null
        egoPriorFullGraphViewportPreserve = null
        const restoreGraphNodeId =
          subject.kind === 'graph-node' ? subject.graphNodeCyId?.trim() || '' : ''
        /** Episode rail + graph: same bug class as graph-node — without this, ``selectedNodeId`` is
         * cleared then ``redraw()`` mounts a new Cytoscape with **no** selection, so dimming classes
         * never apply and every node reads at full opacity (~1–2s later when merged/filtered artifact
         * settles after catalog/async). */
        let restoreEpisodeCyId = ''
        if (subject.kind === 'episode') {
          const meta = subject.episodeMetadataPath?.trim() || ''
          const pref = subject.graphConnectionsCyId?.trim() || ''
          if (pref) {
            restoreEpisodeCyId = pref
          } else if (meta) {
            restoreEpisodeCyId =
              findEpisodeGraphNodeIdForMetadataPath(gf.filteredArtifact, meta) || ''
          }
        }
        if (restoreGraphNodeId) {
          nav.requestFocusNode(restoreGraphNodeId)
        } else if (restoreEpisodeCyId) {
          nav.requestFocusNode(restoreEpisodeCyId)
        }
        redraw()
      })
    })
  },
  { flush: 'post' },
)

watch(
  activeDegreeBucket,
  () => {
    safeGraphWatch('degreeBucket', () => {
      const c = cy
      if (!c) return
      const snap = captureSelectedViewportAnchor(c)
      applyDegreeVisibility(c)
      applyViewportPreserveOrFit(c, snap)
      lastZoomLevel = c.zoom()
      updateZoomPercentDisplay(c)
    })
  },
)

watch(
  minimapOpen,
  (open) => {
    safeGraphWatch('minimap', () => {
      const c = cy
      if (!c) return
      if (open) {
        void nextTick(() => {
          try {
            setupNavigator(c)
          } catch (e) {
            console.warn('[GraphCanvas] watcher (minimap nextTick):', e)
          }
        })
      } else {
        destroyNavigator()
      }
    })
  },
)

watch(
  () => themeStore.choice,
  () => {
    safeGraphWatch('theme', () => {
      pendingViewportPreserve = null
      egoPriorFullGraphViewportPreserve = null
      void nextTick(() => {
        try {
          redraw()
        } catch (e) {
          console.warn('[GraphCanvas] watcher (theme nextTick):', e)
        }
      })
    })
  },
)

watch(
  () => [...nav.topicClusterCanvasCollapsedIds],
  () => {
    safeGraphWatch('topicClusterCollapse', () => {
      const c = cy
      if (c) {
        applyTopicClusterMemberCollapse(c)
      }
    })
  },
  { deep: true },
)

watch(
  () =>
    [
      nav.pendingFocusNodeId,
      nav.pendingFocusFallbackNodeId,
      nav.pendingFocusCameraIncludeRawIds.join('\0'),
      subject.kind,
      subject.episodeMetadataPath,
      subject.episodeUiLabel,
      subject.graphConnectionsCyId,
      gf.fullArtifact,
      gf.filteredArtifact,
    ] as const,
  () => {
    void nextTick(() => {
      const c = cy
      if (!c || graphContentHiddenUntilLayout.value) {
        return
      }
      try {
        // While the first layout pass is running, skip: `finishLayoutPass` will call
        // `tryApplyPendingFocus` after `layoutstop`. Otherwise we consume pending focus early and
        // the post-layout pass sees nothing (broken restore after graph rebuild).
        const applied = tryApplyPendingFocus(c)
        applyEpisodeRepresentativeFocusIfNeeded(c, { skipCamera: applied })
      } catch (e) {
        console.warn('[GraphCanvas] watcher (episode territory + pending focus):', e)
      }
    })
  },
  { flush: 'post' },
)

/* RFC-080 V4 — keyboard wiring. Escape always exits radial mode if
 * active (regardless of focus location, matching the RFC). Alt+R
 * toggles: enter centred on the currently-selected node, or exit if
 * already active. The user can drive the mode entirely from the
 * keyboard until the bottom-bar lens menu ships. */
let radialKeydownHandler: ((e: KeyboardEvent) => void) | null = null

function attachRadialKeydown(): void {
  radialKeydownHandler = (e: KeyboardEvent): void => {
    if (e.key === 'Escape' && radialModeActive.value) {
      e.preventDefault()
      exitRadialMode()
      return
    }
    // Alt+R: toggle. Lower-case match so both `r` and `R` (Shift+Alt+R) work.
    if (e.altKey && (e.key === 'r' || e.key === 'R')) {
      e.preventDefault()
      if (radialModeActive.value) {
        exitRadialMode()
        return
      }
      const c = cy
      if (!c) return
      const sel = c.$('node:selected')
      const target = sel.empty() ? null : sel.first()
      if (!target) return
      enterRadialMode(target.id())
    }
  }
  window.addEventListener('keydown', radialKeydownHandler, true)
}

function detachRadialKeydown(): void {
  if (radialKeydownHandler) {
    try {
      window.removeEventListener('keydown', radialKeydownHandler, true)
    } catch {
      /* ignore */
    }
    radialKeydownHandler = null
  }
}

onMounted(() => {
  if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
    reducedMotionMql = window.matchMedia('(prefers-reduced-motion: reduce)')
    graphPrefersReducedMotion.value = reducedMotionMql.matches
    reducedMotionMqlHandler = (): void => {
      graphPrefersReducedMotion.value = reducedMotionMql!.matches
      const c = cy
      if (!c) {
        return
      }
      try {
        c.style().fromJson(buildCyStyle() as never).update()
        syncGraphLabelTierClasses(c)
      } catch {
        /* ignore */
      }
    }
    reducedMotionMql.addEventListener('change', reducedMotionMqlHandler)
  }
  attachRadialKeydown()
  safeGraphWatch('onMounted', () => {
    redraw()
  })
})

onActivated(() => {
  const c = cy
  if (!c) {
    return
  }
  void nextTick(() => {
    try {
      c.resize()
    } catch {
      /* ignore */
    }
    requestAnimationFrame(() => {
      try {
        c.resize()
      } catch {
        /* ignore */
      }
    })
    // Re-apply pending graph focus / episode strip after returning to the tab.
    if (!graphContentHiddenUntilLayout.value) {
      try {
        const applied = tryApplyPendingFocus(c)
        applyEpisodeRepresentativeFocusIfNeeded(c, { skipCamera: applied })
      } catch (e) {
        console.warn('[GraphCanvas] onActivated graph focus sync:', e)
      }
    }
  })
})

onUnmounted(() => {
  if (reducedMotionMql && reducedMotionMqlHandler) {
    try {
      reducedMotionMql.removeEventListener('change', reducedMotionMqlHandler)
    } catch {
      /* ignore */
    }
    reducedMotionMql = null
    reducedMotionMqlHandler = null
  }
  if (nodeEpisodesCorpusBeyondDebounce != null) {
    clearTimeout(nodeEpisodesCorpusBeyondDebounce)
    nodeEpisodesCorpusBeyondDebounce = null
  }
  detachRadialKeydown()
  destroyCy()
  graphCyNodeCount.value = 0
  pendingViewportPreserve = null
  egoPriorFullGraphViewportPreserve = null
  releaseGraphCanvasLayoutHold()
})

defineExpose({
  fitAnimated,
  redraw,
  clearInteractionState,
  exportGraphPng,
})
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col rounded border border-border bg-canvas">
    <div
      v-if="gf.fullArtifact"
      class="flex min-h-7 min-w-0 flex-wrap items-center gap-2 border-b border-border bg-canvas py-0 pl-2 pr-1"
    >
      <GraphStatusLine variant="summary" bare />
      <span
        v-if="gf.state && searchHighlightCount > 0"
        data-testid="graph-search-highlight-chip"
        class="shrink-0 rounded-full bg-yellow-500/20 px-2 py-0.5 text-[10px] font-medium text-yellow-600"
      >
        {{ searchHighlightCount }}
        {{ searchHighlightCount === 1 ? 'highlight' : 'highlights' }}
      </span>
      <button
        type="button"
        class="ml-auto shrink-0 rounded border border-border px-1.5 py-px text-[10px] leading-none text-surface-foreground hover:bg-overlay"
        data-testid="graph-gesture-overlay-reopen"
        aria-label="Show graph gestures help"
        @click="reopenGestureOverlay"
      >
        Gestures
      </button>
    </div>
    <div
      v-if="gf.state"
      class="flex min-h-7 flex-wrap items-center gap-x-2 gap-y-0.5 border-b border-border px-2 py-0 text-surface-foreground leading-none"
      data-testid="graph-toolbar-types"
    >
      <span class="inline-flex items-center self-center text-[10px] font-semibold uppercase tracking-wide text-muted">
        Types
      </span>
      <span
        v-if="!gf.fullArtifact && searchHighlightCount > 0"
        data-testid="graph-search-highlight-chip"
        class="shrink-0 rounded-full bg-yellow-500/20 px-2 py-0.5 text-[10px] font-medium text-yellow-600"
      >
        {{ searchHighlightCount }}
        {{ searchHighlightCount === 1 ? 'highlight' : 'highlights' }}
      </span>
      <button
        v-if="gf.graphTypesDeviateFromDefaults"
        type="button"
        class="inline-flex items-center rounded bg-muted/25 px-1.5 py-px text-[9px] font-medium leading-none text-muted hover:bg-overlay"
        data-testid="graph-types-reset"
        @click="gf.resetGraphTypeVisibilityDefaults()"
      >
        filters active — reset
      </button>
      <button
        type="button"
        class="inline-flex items-center py-px text-[10px] leading-none text-primary underline"
        @click="gf.selectAllTypes()"
      >
        all
      </button>
      <button
        type="button"
        class="inline-flex items-center py-px text-[10px] leading-none text-primary underline"
        @click="gf.deselectAllTypes()"
      >
        none
      </button>
      <label
        v-for="t in typeFilterKeys"
        :key="t"
        class="flex cursor-pointer items-center gap-1 py-px text-[10px] leading-none"
      >
        <input
          type="checkbox"
          class="size-3 shrink-0 rounded border-border"
          :checked="gf.state!.allowedTypes[t]"
          @change="gf.toggleAllowedType(t)"
        >
        <span
          class="h-2.5 w-2.5 shrink-0 rounded-sm ring-1 ring-black/15 dark:ring-white/20"
          :style="{ backgroundColor: graphNodeFill(String(t)) }"
          :title="`${String(t)} (node fill)`"
          aria-hidden="true"
        />
        <span>{{ graphNodeLegendLabel(t) }}</span>
        <span class="font-medium text-muted">({{ typeHistogramCounts[t] ?? 0 }})</span>
      </label>
      <GraphFiltersPopover :degree-histogram-counts="degreeHistogramCounts" />
    </div>

    <div class="flex min-h-0 min-w-0 flex-1 flex-col">
      <div
        v-if="episodeTerritoryMode === 'empty'"
        class="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-border bg-elevated/50 px-2 py-1 text-[10px] leading-snug text-muted"
        data-testid="graph-episode-territory-strip"
      >
        <span class="min-w-0 flex-1">
          Episode not in current graph view —
          <button
            type="button"
            class="rounded px-0.5 font-medium text-primary underline hover:opacity-90 disabled:opacity-40"
            data-testid="graph-episode-territory-load"
            :disabled="episodeTerritoryLoadBusy || !shell.corpusPath.trim() || !shell.healthStatus"
            @click="void loadEpisodeSliceForTerritoryStrip()"
          >
            Load into graph
          </button>
        </span>
        <button
          type="button"
          class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[9px] font-medium text-surface-foreground hover:bg-overlay"
          data-testid="graph-episode-territory-dismiss"
          @click="dismissEpisodeTerritoryStrip"
        >
          Dismiss
        </button>
      </div>
      <div class="flex min-h-0 min-w-0 flex-1 flex-col">
        <div
          ref="canvasHost"
          tabindex="-1"
          class="relative isolate min-h-0 min-w-0 flex-1 overflow-hidden outline-none"
          :aria-busy="graphContentHiddenUntilLayout ? 'true' : 'false'"
        >
        <!--
          Hide / block only the Cytoscape layer during layout — not the whole host.
          pointer-events-none on the parent previously made the graph toolbar and zoom cluster
          unreliable in some browsers while graphContentHiddenUntilLayout stayed true.
        -->
        <div
          ref="container"
          class="graph-canvas absolute inset-0 min-h-0"
          :class="
            graphContentHiddenUntilLayout
              ? 'pointer-events-none invisible opacity-0'
              : 'visible opacity-100'
          "
        />
        <div
          v-if="graphContentHiddenUntilLayout && graphCyNodeCount > 0"
          class="pointer-events-none absolute inset-0 z-[3] flex items-center justify-center bg-surface/35 backdrop-blur-[1px]"
          aria-live="polite"
          data-testid="graph-layout-loading"
        >
          <span
            class="rounded border border-border/70 bg-surface/95 px-2 py-1 text-[11px] text-muted shadow-sm"
          >
            Laying out graph…
          </span>
        </div>
        <!--
          RFC-080 V4 — radial focus mode. The aria-live region announces
          enter/exit to screen readers; the visible test-id pip is for
          dev / Playwright reach (no user-facing toggle UI in this
          slice — Alt+R toggles, Escape exits; bottom-bar lens menu
          ships in a follow-up).
        -->
        <div
          aria-live="polite"
          aria-atomic="true"
          class="sr-only"
        >
          {{ radialAriaMessage }}
        </div>
        <div
          v-if="radialModeActive"
          data-testid="graph-radial-mode-active"
          class="pointer-events-none absolute right-2 top-2 z-[4] rounded border border-border/70 bg-surface/95 px-2 py-0.5 text-[10px] font-medium text-surface-foreground shadow-sm"
        >
          Radial · Esc to exit
        </div>
        <GraphGestureOverlay
          ref="gestureOverlayRef"
          :has-nodes="graphCyNodeCount > 0"
          @dismissed="focusCanvasHost"
        />
        <div
          v-show="boxZoomRect.show"
          class="pointer-events-none absolute z-[5] box-border border-2 border-dashed border-primary bg-primary/10"
          :style="{
            left: `${boxZoomRect.left}px`,
            top: `${boxZoomRect.top}px`,
            width: `${boxZoomRect.width}px`,
            height: `${boxZoomRect.height}px`,
          }"
        />
        <div
          v-show="minimapOpen"
          class="pointer-events-auto absolute bottom-2 left-2 z-10 max-h-[min(13.5rem,35%)] max-w-[min(10.5rem,calc(100%-1rem))]"
        >
          <div
            class="relative h-[7.5rem] w-[10.5rem] overflow-hidden rounded border border-border bg-surface shadow-md"
          >
            <button
              type="button"
              class="absolute right-0.5 top-0.5 z-20 rounded px-1 text-[10px] leading-none text-muted hover:text-surface-foreground"
              data-testid="graph-minimap-close"
              aria-label="Hide minimap"
              @click="hideMinimap"
            >
              ×
            </button>
            <div
              id="gi-kg-graph-minimap"
              ref="minimapHost"
              data-testid="graph-minimap"
              class="h-full w-full overflow-hidden"
              aria-label="Graph minimap"
            />
          </div>
        </div>
        </div>
        <GraphBottomBar
          v-if="gf.state"
          :show-lens-controls="!!gf.fullArtifact"
          :show-gestures-in-bottom-bar="!gf.fullArtifact"
          :zoom-percent="zoomPercent"
          :search-highlight-count="searchHighlightCount"
          :preferred-layout="preferredLayout"
          @fit="fitAnimated"
          @zoom-in="zoomIn"
          @zoom-out="zoomOut"
          @zoom-reset="zoomReset100"
          @export-png="exportGraphPng"
          @reopen-gestures="reopenGestureOverlay"
          @relayout="runRelayout"
          @cycle-layout="onCycleLayout"
          @request-corpus-graph-sync="emit('request-corpus-graph-sync')"
          @request-graph-full-reset="emit('request-graph-full-reset')"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
/*
 * With `container: '#…'`, the host never gets `.cytoscape-navigator`, so the
 * package CSS for `> img` / `> canvas` does not apply; keep the thumbnail
 * clipped to the inset panel.
 */
#gi-kg-graph-minimap :deep(img) {
  max-width: 100%;
  max-height: 100%;
}
</style>
