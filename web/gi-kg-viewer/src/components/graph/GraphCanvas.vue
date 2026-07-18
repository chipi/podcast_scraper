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
import { useGraphLoadModeStore } from '../../stores/graphLoadMode'
import { useGraphThemeFocusStore } from '../../stores/graphThemeFocus'
import { useGraphTopDownStore } from '../../stores/graphTopDown'
import { useGraphAnalyticsStore } from '../../stores/graphAnalytics'
import { useGraphHandoffStore } from '../../stores/graphHandoff'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useSearchStore } from '../../stores/search'
import { useActiveSearchContextStore } from '../../stores/activeSearchContext'
import { useShellStore } from '../../stores/shell'
import { useThemeStore } from '../../stores/theme'
import type { RawGraphNode } from '../../types/artifact'
import type { TopicClustersDocument } from '../../api/corpusTopicClustersApi'
import {
  THEME_REGION_PALETTE_SIZE,
  themeRegionIndex,
} from '../../utils/themeRegionPalette'
import {
  applyCoGuestEdges,
  applyConsensusEdges,
  applyCredibilityBorder,
  applyPersonCommunityRegions,
  applyVelocityHalo,
  clearCoGuestEdges,
  clearConsensusEdges,
  clearCredibilityBorder,
  clearPersonCommunityRegions,
  clearVelocityHalo,
  type CoGuestEnvelopeData,
  type ConsensusEnvelopeData,
  type GroundingEnvelopeData,
  type VelocityEnvelopeData,
} from '../../utils/cyGraphLensOverlays'
import { fetchCachedCorpusEnvelope } from '../../composables/useEnrichmentEnvelopeCache'
import { degreeBucketFor, emptyDegreeCounts } from '../../utils/graphDegreeBuckets'
import {
  findEpisodeGraphNodeIdForMetadataPath,
  findEpisodeGraphNodeIdForMetadataPathOrEpisodeId,
  graphCyIdRepresentsEpisodeNode,
  logicalEpisodeIdFromGraphNodeId,
  metadataPathFromEpisodeProperties,
  normalizeCorpusMetadataPath,
  resolveEpisodeMetadataFromLoadedArtifacts,
  resolveEpisodeMetadataViaCorpusCatalog,
} from '../../utils/graphEpisodeMetadata'
import {
  computeNodeIdSetDifference,
  decideReconcileAction,
} from '../../utils/graphHandoffInvariant'
import { resolveHandoffCandidateNode } from '../../utils/graphHandoffRestore'
import * as giKgCoseLayout from '../../utils/cyCoseLayoutOptions'
import {
  syncGraphLabelTierClasses,
  syncGraphNodeVisibilityTierClasses,
} from '../../utils/cyGraphLabelTier'
import { buildGiKgCyStylesheet, cytoscapeSideLabelMarginXCallback } from '../../utils/cyGraphStylesheet'
import {
  computeRadialPositions,
  type RadialSnapshot,
} from '../../utils/cyRadialLayout'
import {
  computeEpisodeTimelinePositions,
  deterministicJitter,
  weightedMeanXFromEpisodes,
  type TimelineEpisodeInput,
  type TimelinePosition,
} from '../../utils/cyTimelineLayout'
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
import GraphFilterBar from './GraphFilterBar.vue'
import GraphGestureOverlay from './GraphGestureOverlay.vue'
import GraphThemeLegend from './GraphThemeLegend.vue'
import GraphStatusLine from './GraphStatusLine.vue'

registerNavigator(cytoscape)

const emit = defineEmits<{
  'request-corpus-graph-sync': []
  'request-graph-full-reset': []
}>()

const gf = useGraphFilterStore()
const lenses = useGraphLensesStore()
const themeFocus = useGraphThemeFocusStore()
const loadMode = useGraphLoadModeStore()
const topDown = useGraphTopDownStore()
const ge = useGraphExplorerStore()
const { preferredLayout, minimapOpen, activeDegreeBucket } = storeToRefs(ge)
const nav = useGraphNavigationStore()
const graphHandoff = useGraphHandoffStore()
const graphAnalytics = useGraphAnalyticsStore()
const subject = useSubjectStore()
const artifacts = useArtifactsStore()
const graphExpansion = useGraphExpansionStore()
const { expandedBySeed } = storeToRefs(graphExpansion)
const shell = useShellStore()
const searchStore = useSearchStore()
const activeSearchContext = useActiveSearchContextStore()
const themeStore = useThemeStore()

const graphEpisodeOpenGate = new StaleGeneration()

/** Minimum zoom when animating to a focused node (digest/search hand-off + canvas single-tap). */
const GRAPH_FOCUS_FRAME_MIN_ZOOM = 1.3

/** Episode on Graph: select Episode node for subject episode; Cytoscape 1-hop neighbourhood dim. */
const episodeTerritoryMode = ref<'off' | 'empty'>('off')
const episodeTerritoryLoadBusy = ref(false)
const episodeTerritoryDismissed = ref(false)
/** Auto-load gate (#696): tracks metadata paths we've already tried to
 * auto-fetch on this canvas session so a transient fetch failure doesn't
 * loop. Cleared when the operator dismisses the strip + retries via the
 * manual load button (which calls ``loadEpisodeSliceForTerritoryStrip``
 * directly), or when the focused episode changes. */
const episodeTerritoryAutoLoadTriedPaths = ref<Set<string>>(new Set())

watch(
  () => subject.episodeMetadataPath,
  () => {
    episodeTerritoryDismissed.value = false
    episodeTerritoryAutoLoadTriedPaths.value = new Set()
  },
)

/** Library → Graph hand-off (#696): when the operator clicks an Episode
 * row in Library / Search, the subject store gets ``episodeMetadataPath``
 * but the graph canvas may not have that episode's GI/KG slice loaded.
 * The existing ``applyEpisodeRepresentativeFocusIfNeeded`` then sets
 * ``episodeTerritoryMode = 'empty'`` and shows a strip with a manual
 * "load slice" button — but the operator's intent on every previous
 * click was clearly "show me this episode in the graph". Auto-fire the
 * same load that the manual button does so the Library → Graph switch
 * just works.
 *
 * Guarded by ``episodeTerritoryAutoLoadTriedPaths`` so a transient
 * 5xx / network failure doesn't infinite-loop; ``episodeTerritoryDismissed``
 * preserves the manual-control escape hatch (operator can dismiss the
 * strip and the auto-load won't re-try until they re-focus the
 * episode). */
/** When the user navigates to a different episode, clear the
 * "already tried" guard so the new episode gets its own auto-load
 * attempt. Without this, a Library → episode A → (failed slice) →
 * Library → episode A → click leaves the guard set forever and
 * camera centering stays broken until full page reload. */
watch(
  () => subject.episodeMetadataPath,
  (next, prev) => {
    if (next !== prev) {
      episodeTerritoryAutoLoadTriedPaths.value.clear()
    }
  },
)

watch(
  () => [
    episodeTerritoryMode.value,
    subject.episodeMetadataPath,
    episodeTerritoryDismissed.value,
  ] as const,
  () => {
    if (episodeTerritoryMode.value !== 'empty') {
      return
    }
    if (episodeTerritoryDismissed.value) {
      return
    }
    const meta = subject.episodeMetadataPath?.trim()
    if (!meta) {
      return
    }
    if (episodeTerritoryAutoLoadTriedPaths.value.has(meta)) {
      return
    }
    if (episodeTerritoryLoadBusy.value) {
      return
    }
    episodeTerritoryAutoLoadTriedPaths.value.add(meta)
    void loadEpisodeSliceForTerritoryStrip()
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
    subject.focusEpisode(meta, {
      graphConnectionsCyId: cyId,
      ...(eid ? { episodeId: eid } : {}),
    })
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
/** Debounce rapid-fire redraw() calls from multiple watchers firing in sequence. */
let redrawDebounceTimer: ReturnType<typeof setTimeout> | null = null
/** Prevents immediate redraws after layout completion (cooldown for watcher cascades). */
let layoutCompletionCooldownUntil = 0
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
/**
 * F4 — single-retry budget per envelope generation for the self-healing
 * reconciliation pass. Tracks the last envelope generation that consumed
 * its retry; if the same generation hits the invariant again, we accept
 * divergence + log instead of looping.
 */
let handoffReconcileRetryGen: number | null = null
/** Snapshot of ``focusNodeId`` before the ``filteredArtifact`` watcher clears it; used for incremental check. */
let priorEgoBeforeWatcherClear = ''
/** Track previous artifact's node set to detect incremental appends (superset) vs full replacements. */
let prevFilteredArtifactNodeIds = new Set<string>()

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

/**
 * #967 — when a full-replacement redraw early-restores the FSM-applied selection at
 * cy-creation (closing the no-selection window), the layout positions aren't final yet,
 * so the camera centre is deferred to ``finishLayoutPass``. This carries the cy id whose
 * camera still needs centring; cleared once honoured (or on a redraw that supersedes it).
 */
let pendingEarlyHandoffCameraCyId = ''

let pendingViewportPreserve: ViewportPreserveSnap | null = null
/** Full-graph viewport before entering 1-hop (shift+dbl-click); applied when returning to full graph. */
let egoPriorFullGraphViewportPreserve: ViewportPreserveSnap | null = null

/** Target rendered (screen) position for the selected node; kept stable across wheel + toolbar zoom. */
let selectedNodeZoomAnchor: { x: number; y: number } | null = null

/** Skip pan-by-anchor in zoom handler while applyViewportPreserveOrFit runs (zoom+pan ordering). */
let suspendSelectedNodeZoomAnchorCorrection = 0

/**
 * Camera focus tracking: when a focus action is in progress (Digest pill, Dashboard click,
 * Library episode open, etc.), the detail panel may open and resize the graph canvas
 * AFTER the camera animation completes. This causes the centered node to drift off-center.
 *
 * Track the focus target and a deadline; the ResizeObserver re-centers on this node
 * whenever the canvas resizes within the deadline window.
 */
let pendingRecenterCyId: string | null = null
let pendingRecenterUntil = 0
function armPendingRecenter(cyId: string, durationMs = 5000): void {
  pendingRecenterCyId = cyId
  pendingRecenterUntil = Date.now() + durationMs
}
/**
 * Recenter on the pending focus target if still armed. Used by ResizeObserver
 * (canvas size changed) and by deferred timers after animation (catches resizes
 * that didn't fire on the observer, e.g. tab switches, rapid clicks, etc.).
 */
function recenterIfPending(core: Core): void {
  if (!pendingRecenterCyId || Date.now() >= pendingRecenterUntil) {
    return
  }
  try {
    const tgt = core.$id(pendingRecenterCyId)
    if (!tgt.empty()) {
      core.resize()
      core.center(tgt)
    }
  } catch {
    // ignore — pending recenter is best-effort.
  }
}

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
    // FIRST: If there's a pending focus node, skip all viewport logic and let tryApplyPendingFocus handle the camera
    if (nav.pendingFocusNodeId) {
      return
    }
    // SECOND: If requestFitAfterLoad is set (external load without focus), force fit at 50% zoom
    if (nav.requestFitAfterLoad) {
      try {
        // For external loads without focus, center on graph with reasonable zoom instead of fitting to extremes
        const els = core.elements(':visible')
        const bbox = els.boundingBox()
        const centerX = (bbox.x1 + bbox.x2) / 2
        const centerY = (bbox.y1 + bbox.y2) / 2
        core.zoom(0.5)  // 50% zoom as a reasonable default
        core.center()  // Center on viewport center
        // Pan to center the graph's center point
        core.panBy({ x: -centerX * 0.5, y: -centerY * 0.5 })
      } catch {
        /* ignore */
      }
      return
    }
    // THIRD: Try to preserve viewport from snapshot (if valid)
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
    // LAST: Default fit to visible elements
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
  applyContextEmphasis(core)
}

/**
 * PRD-033 FR5.1 — emphasize (size + ring) nodes whose episode is relevant to the
 * active search context (RFC-094 OQ-2). Episode nodes match by their stripped id;
 * Insight/Quote nodes by their ``episodeId`` data. No-op when no context is active.
 */
function applyContextEmphasis(core: Core): void {
  core.nodes().removeClass('context-relevant')
  if (!activeSearchContext.active) return
  core.batch(() => {
    core.nodes().forEach((n) => {
      const dataEp = n.data('episodeId')
      let epId = typeof dataEp === 'string' ? dataEp.trim() : ''
      if (!epId) {
        const id = n.id()
        if (id.startsWith('episode:')) epId = id.slice('episode:'.length)
      }
      if (epId && activeSearchContext.relevanceFor(epId)) {
        n.addClass('context-relevant')
      }
    })
  })
}

function buildCyStyle() {
  return [
    ...(buildGiKgCyStylesheet({
      includeSearchHit: true,
      // PRD-033 FR5.1 — emphasize nodes relevant to the active search context.
      includeContextEmphasis: true,
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

/** graph-v3 tier 7-3 — theme-focus dim.
 *  Called from the legend focus bus (`useGraphThemeFocusStore`). Every node
 *  whose `themeClusterId` is IN the provided set is treated as focused;
 *  everything else is dimmed. Edges are dimmed unless BOTH endpoints are in
 *  the focus set. `themeClusterId` is propagated onto Insight / Episode /
 *  Person / Podcast / Org nodes upstream (see graph-v3 tier T), so the
 *  focus signal reaches the whole community, not just its TopicCluster
 *  parent. Empty set clears back to the default view.  */
function applyGraphSelectionDimFromThemeIds(core: Core, themeIds: Set<string>): void {
  if (themeIds.size === 0) {
    clearGraphSelectionDim(core)
    return
  }
  core.batch(() => {
    core.nodes().addClass('graph-dimmed')
    core.edges().addClass('graph-edge-dimmed')
    core.nodes().forEach((n) => {
      const tid = n.data('themeClusterId')
      if (typeof tid === 'string' && themeIds.has(tid)) {
        n.addClass('graph-neighbour').removeClass('graph-dimmed')
      }
    })
    core.edges().forEach((ee) => {
      const sDim = ee.source().hasClass('graph-dimmed')
      const tDim = ee.target().hasClass('graph-dimmed')
      if (!sDim && !tDim) {
        ee.addClass('graph-edge-neighbour').removeClass('graph-edge-dimmed')
      }
    })
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
  /** Use the episode-id fallback variant: the unified-merge graph emits Episode nodes
   *  with ``__unified_ep__:UUID`` ids whose properties (``podcast_id``, ``title``,
   *  ``publish_date``, ``duration_ms``, ``feed_id``) do NOT include ``metadata_relative_path``,
   *  so a path-only lookup always returns null and camera centering never animates.
   *  ``findEpisodeGraphNodeIdForMetadataPathOrEpisodeId`` falls back to matching by
   *  ``logicalEpisodeIdFromGraphNodeId`` (UUID extraction) when path-based resolution misses. */
  const best = findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
    gf.filteredArtifact,
    meta,
    subject.episodeId,
  )
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
    /** Order matters: ``clearEpisodeRepresentativeGraphState`` resets mode to 'off' internally; we MUST write 'empty' AFTER so the auto-load watcher sees the correct final value within Vue's batching window (otherwise auto-load never fires and the episode never enters the graph). */
    clearEpisodeRepresentativeGraphState(core)
    episodeTerritoryMode.value = 'empty'
    return
  }
  const coll = core.$id(cyEpisodeId)
  if (coll.empty()) {
    /** See above: clear before setting 'empty' so the watcher observes the correct final mode. */
    clearEpisodeRepresentativeGraphState(core)
    episodeTerritoryMode.value = 'empty'
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
  // F2 — generation token for stale-await checks (decision #4 / spec § 8 check points).
  const territoryGen = graphHandoff.generation
  try {
    // Advance to loading_fetch if we're at idle/ready (FSM may already be there
    // via handoffRequested from the entry point).
    if (graphHandoff.state === 'idle' || graphHandoff.state === 'ready') {
      graphHandoff.advanceState('loading_fetch')
    }
    const d = await fetchCorpusEpisodeDetail(root, meta)
    // F2 — stale-check after fetch await (decision #4 / 8+ check points).
    if (graphHandoff.isStale(territoryGen)) {
      return
    }
    /** Capture the episode UUID so ``applyEpisodeRepresentativeFocusIfNeeded`` can resolve the
     *  Cytoscape node id via ``__unified_ep__:UUID`` when Episode rows lack ``metadata_relative_path``
     *  in their properties (the unified-merge graph case). Without this fallback, camera centering
     *  after Library / Search / Dashboard handoff stays broken. */
    if (d.episode_id?.trim()) {
      subject.setEpisodeId(d.episode_id.trim())
    }
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
    /** Tag this append as external (Library-style) navigation so the
     *  ``filteredArtifact`` watcher does NOT short-circuit the
     *  ``scheduleRedraw()`` call for "incremental append, no pending
     *  focus" — without this, second/third "Open in graph" clicks
     *  load the artifact into ``filteredArtifact`` but never flush
     *  the new nodes into Cytoscape's ``core``, so ``$id(cy).empty()``
     *  is true and camera animation never fires. */
    artifacts.setLoadSource('subject-external')
    // Advance the FSM to ``loading_merge`` only when there's an
    // in-flight envelope it's driving (``loading_fetch`` or
    // ``loading_bootstrap``). This function is called from two places:
    //
    //   (a) As the primary fetch+merge for a Library / Episode-panel
    //       handoff — FSM is in ``loading_fetch``, advance is correct.
    //   (b) Reactively from subject changes that happen *after* the FSM
    //       has settled to ``ready`` (e.g. the user clicks a different
    //       Library row's preview, or a subject-rail tab return). In
    //       this case the FSM should NOT be advanced backwards into
    //       ``loading_merge`` — that's the "details panel impacts graph"
    //       loop we want to prevent.
    //
    // The guard below already protects case (b) — only advance if the
    // primary handoff is mid-flight. Case (a) gets the advance; case
    // (b) leaves FSM in ``ready``.
    if (graphHandoff.state === 'loading_fetch' || graphHandoff.state === 'loading_bootstrap') {
      graphHandoff.advanceState('loading_merge')
    }
    await artifacts.appendRelativeArtifacts(paths)
    if (graphHandoff.isStale(territoryGen)) {
      return
    }
    subject.setEpisodeUiLabel(d.episode_title ?? null)
    episodeTerritoryDismissed.value = false
  } catch (e) {
    // C6 / decision #15 — surface failed handoffs through the FSM so the
    // error strip in C7 can render. Replaces the silent swallow that left
    // users with "I clicked, nothing happened, no feedback" UX.
    const reason = e instanceof Error ? e.message : 'territory fetch failed'
    graphHandoff.handoffFailed(`territory fetch: ${reason}`)
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

// graph-v3 U — palette + hash extracted to `utils/themeRegionPalette.ts`
// so the legend + tests can resolve the same colour for a given `thc:...`
// id without duplicating the constants here.

/** graph-v3 R-V + Tier 5A-2 — paint theme-cluster region classes.
 *
 *  Propagation now runs artifact-side in `applyThemeClustersOverlay` so
 *  every raw graph node with a theme membership already carries a
 *  `themeClusterId` on its data (Topics + Episodes as direct seeds;
 *  Insights + Persons + Orgs + Podcasts by edge-walk propagation).
 *  This function only PAINTS: for each node with themeClusterId set,
 *  add the matching `theme-region-N` class based on the stable hash.
 *  Enricher-gated caller (finishLayoutPass / watcher) keeps this a
 *  no-op when the artifact isn't loaded. */
function applyThemeRegionClasses(
  core: Core,
  doc: TopicClustersDocument | null,
): void {
  if (!doc?.clusters?.length) return
  core.batch(() => {
    for (let i = 0; i < THEME_REGION_PALETTE_SIZE; i++) {
      core.nodes().removeClass(`theme-region-${i}`)
    }
    core.nodes().forEach((n) => {
      const raw = n.data('themeClusterId')
      if (typeof raw !== 'string' || !raw.trim()) return
      n.addClass(`theme-region-${themeRegionIndex(raw)}`)
    })
  })
}

function clearThemeRegionClasses(core: Core): void {
  core.batch(() => {
    for (let i = 0; i < THEME_REGION_PALETTE_SIZE; i++) {
      core.nodes().removeClass(`theme-region-${i}`)
    }
  })
}

/** graph-v3 Tier 5C/5D — refresh every enricher-based lens overlay
 *  based on the current lens flags + corpus. Each lens fetches its
 *  envelope via the cached helper (so subsequent calls are a Map hit)
 *  and either paints classes / adds edges OR clears them. Called from
 *  finishLayoutPass (after each redraw) + from per-lens watchers so a
 *  live toggle takes effect without a full re-layout. */
function refreshEnricherLensOverlays(): void {
  const core = cy
  if (!core) return
  const root = shell.corpusPath.trim()
  // Velocity halo (Topic/Person)
  if (lenses.velocityHalo && root) {
    void fetchCachedCorpusEnvelope<VelocityEnvelopeData>(root, 'temporal_velocity')
      .then((env) => {
        if (!cy) return
        applyVelocityHalo(cy, env?.data ?? null)
      })
      .catch(() => {
        /* silently degrade — lens stays clear */
      })
  } else {
    clearVelocityHalo(core)
  }
  // Person credibility border
  if (lenses.personCredibility && root) {
    void fetchCachedCorpusEnvelope<GroundingEnvelopeData>(root, 'grounding_rate')
      .then((env) => {
        if (!cy) return
        applyCredibilityBorder(cy, env?.data ?? null)
      })
      .catch(() => {
        /* silently degrade */
      })
  } else {
    clearCredibilityBorder(core)
  }
  // Consensus edges
  if (lenses.consensusEdges && root) {
    void fetchCachedCorpusEnvelope<ConsensusEnvelopeData>(root, 'topic_consensus')
      .then((env) => {
        if (!cy) return
        applyConsensusEdges(cy, env?.data ?? null)
      })
      .catch(() => {
        /* silently degrade */
      })
  } else {
    clearConsensusEdges(core)
  }
  // Co-guest edges
  if (lenses.coGuestEdges && root) {
    void fetchCachedCorpusEnvelope<CoGuestEnvelopeData>(root, 'guest_coappearance')
      .then((env) => {
        if (!cy) return
        applyCoGuestEdges(cy, env?.data ?? null)
      })
      .catch(() => {
        /* silently degrade */
      })
  } else {
    clearCoGuestEdges(core)
  }
  // graph-v3 tier 7-4 — Person community underlay regions
  if (lenses.personCommunities && root) {
    void fetchCachedCorpusEnvelope<CoGuestEnvelopeData>(root, 'guest_coappearance')
      .then((env) => {
        if (!cy) return
        applyPersonCommunityRegions(cy, env?.data ?? null)
      })
      .catch(() => {
        /* silently degrade */
      })
  } else {
    clearPersonCommunityRegions(core)
  }
}

/** graph-v3 Tier 5B — annotate bridge nodes with the themes they bridge.
 *  For each node carrying the `graph-bridge` class, walk its neighbourhood
 *  and collect the distinct themeClusterId values touched by neighbours;
 *  when the set has >=2 entries, store both the ids and the human labels
 *  on the bridge node's data as `bridgedThemes` / `bridgedThemeLabels`.
 *
 *  Compositional signal: makes K's rose ring analytically meaningful when
 *  the theme regions lens is on ("bridge between AI-and-jobs and
 *  interest-rates"). No-op when either lens is off or when the theme-cluster
 *  artifact isn't loaded — the empty label set falls out naturally. */
function annotateBridgesWithThemes(
  core: Core,
  doc: TopicClustersDocument | null,
): void {
  const labelById = new Map<string, string>()
  for (const cl of doc?.clusters ?? []) {
    const id = typeof cl?.graph_compound_parent_id === 'string' ? cl.graph_compound_parent_id.trim() : ''
    if (!id) continue
    const lbl = typeof cl?.canonical_label === 'string' && cl.canonical_label.trim() ? cl.canonical_label.trim() : id
    labelById.set(id, lbl)
  }
  core.batch(() => {
    core.nodes('.graph-bridge').forEach((bridge) => {
      const themes = new Set<string>()
      bridge.neighborhood('node').forEach((nb) => {
        const t = nb.data('themeClusterId')
        if (typeof t === 'string' && t && labelById.has(t)) themes.add(t)
      })
      if (themes.size >= 2) {
        const arr = Array.from(themes).sort()
        bridge.data('bridgedThemes', arr)
        bridge.data(
          'bridgedThemeLabels',
          arr.map((t) => labelById.get(t) ?? t),
        )
      } else {
        try {
          bridge.removeData('bridgedThemes')
          bridge.removeData('bridgedThemeLabels')
        } catch {
          /* removeData is not available on all cytoscape versions; ignore */
        }
      }
    })
  })
}

/** graph-v3 Tier 6-4 — bridge hover tooltip.
 *  When the cursor lands on a `.graph-bridge` node, mutate the canvas
 *  container's native `title` attribute so the OS tooltip surfaces the
 *  themes this bridge spans (populated by `annotateBridgesWithThemes`).
 *  Native `title` gives the ~0.5s hover delay users expect and zero
 *  extra DOM/JS surface — Cytoscape draws to canvas so there is no
 *  per-node element to hang tippy off cheaply. Non-bridge hovers clear
 *  the attribute so we never leave a stale tooltip attached. */
function maybeSetBridgeHoverTitle(
  el: HTMLElement | null,
  node: NodeSingular,
): void {
  if (!el) return
  if (!node.hasClass('graph-bridge')) {
    if (el.hasAttribute('title')) el.removeAttribute('title')
    return
  }
  const raw = node.data('bridgedThemeLabels')
  const labels = Array.isArray(raw)
    ? raw.filter((v): v is string => typeof v === 'string' && v.trim().length > 0)
    : []
  if (labels.length < 2) {
    /* Bridge without theme-cluster context (theme regions lens off or
       artifact missing) — still surface a minimal tooltip so users
       understand the rose ring. */
    el.setAttribute('title', 'Bridge node — connects distinct neighbourhoods')
    return
  }
  el.setAttribute('title', `Bridge: ${labels.join(' ↔ ')}`)
}

function clearBridgeHoverTitle(el: HTMLElement | null): void {
  if (el && el.hasAttribute('title')) el.removeAttribute('title')
}

/** graph-v3 K — bridge nodes via normalized betweenness centrality.
 *  Runs once post-layout. Semantically-eligible node types (Topic,
 *  Podcast, Entity_person, Entity_organization) with betweenness above
 *  the threshold get a `graph-bridge` class; the stylesheet paints a
 *  distinctive dashed rose border so bridging entities stand out.
 *
 *  Episode + Insight are excluded up front. On this data-model episodes
 *  always sit between insights and topics/persons and insights always
 *  sit between quotes and topics, so their betweenness is high by
 *  construction — flagging them as bridges makes the class synonymous
 *  with "structural connector" (noise), not "community bridge" (signal).
 *
 *  Cost: O(V*E) — ~833 nodes / ~2-4k edges on prod-v2 measured under 200ms. */
function applyBridgeNodeClass(core: Core): void {
  try {
    const bc = core.elements().betweennessCentrality({ directed: false })
    /* Threshold chosen after scanning prod-v2 by type: Topic p90 ≈ 0.005,
       Entity_organization p90 ≈ 0.05, Podcast p90 ≈ 0.35. 0.05 catches
       the analytically interesting bridges across all four types without
       flooding the canvas with false positives. */
    const threshold = 0.05
    const eligibleSelector =
      '[type = "Topic"], [type = "Podcast"], [type = "Entity_person"], [type = "Entity_organization"]'
    core.batch(() => {
      core.nodes().removeClass('graph-bridge')
      core.nodes(eligibleSelector).forEach((n) => {
        const score = bc.betweennessNormalized(n)
        if (Number.isFinite(score) && score >= threshold) {
          n.addClass('graph-bridge')
        }
      })
    })
  } catch {
    /* betweenness fails on empty graphs; safe to skip */
  }
}

/** WIP §4.3 — hub emphasis from graph degree (post-layout).
 *  graph-v3 J — extended beyond Topic to Episode + Entity_person +
 *  Entity_organization so the mapData(degreeHeat) size specialization
 *  actually fires for those types (upstream corpus doesn't ship
 *  degreeHeat for non-Topic nodes). Cap 30 stays uniform; Topics tend to
 *  saturate first because they're the connectors. */
function applyTopicDegreeHeat(core: Core): void {
  const maxDegree = 30
  core.batch(() => {
    const sized = core.nodes(
      '[type = "Topic"], [type = "Episode"], [type = "Entity_person"], [type = "Entity_organization"]',
    )
    sized.forEach((n) => {
      const d = n.degree(false)
      const heat = Math.min(1, d / maxDegree)
      try {
        n.data('degreeHeat', heat)
      } catch {
        /* ignore */
      }
    })
    // Topic-specific hub class kept for existing legend / focus selectors.
    core.nodes('[type = "Topic"]').forEach((n) => {
      const heat = Number(n.data('degreeHeat') ?? 0)
      if (heat > 0.7) {
        n.addClass('graph-topic-heat-high')
      } else {
        n.removeClass('graph-topic-heat-high')
      }
    })
  })
}

function layoutOptionsFor(name: string, opts?: { numIter?: number }): Record<string, unknown> {
  if (name === 'fcose') {
    // Namespace import + fallback: avoids rare `ReferenceError` from named-import/HMR chunks.
    try {
      const fn = giKgCoseLayout.giKgCoseLayoutOptionsMain
      if (typeof fn === 'function') {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return fn(opts?.numIter) as any
      }
    } catch (e) {
      console.warn('[GraphCanvas] giKgCoseLayoutOptionsMain failed, using fallback', e)
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return giKgCoseLayout.giKgCoseLayoutOptionsMainFallback(opts?.numIter) as any
  }
  // 'timeline' is handled by `timelineLayoutSpec(core)` at the caller —
  // it needs the live Cytoscape collection to read publishDate +
  // walk edges for Topic/Person band placement; layoutOptionsFor
  // doesn't have access to the eles. Returning a no-op spec here
  // would be wrong; the dispatch call sites special-case it.
  return { name, padding: 36 }
}

const TIMELINE_TOPIC_BAND_OFFSET = 100
const TIMELINE_PERSON_BAND_OFFSET = -180
const TIMELINE_INSIGHT_OFFSET = 30
const TIMELINE_QUOTE_OFFSET = -30
const TIMELINE_DEFAULT_X = 0

/** RFC-080 V3 — count of Episodes parked at the missing-date spot in the
 * last applied timeline layout. Surfaced via the bottom-bar lens menu (or
 * a quiet inline note) once that UI lands; until then, devs can read it
 * via Vue devtools. */
const timelineMissingDateCount = ref(0)

/** Build a Cytoscape layout spec for the timeline lens. Reads
 * publishDate (Unix ms) from Episode node data, computes quantile
 * positions (default axis), then propagates Topic / Person / Insight /
 * Quote positions via weighted-mean of connected episodes (RFC-080 V3).
 *
 * Returns `{ name: 'preset', positions, padding }` ready to hand to
 * `eles.layout(spec)`. Empty / no-Episode collections fall back to a
 * grid layout so the canvas still renders cleanly.
 */
function timelineLayoutSpec(core: Core): Record<string, unknown> {
  const episodeNodes = core.nodes('node[type = "Episode"]')
  if (episodeNodes.empty()) {
    return { name: 'grid', padding: 36 }
  }

  const containerEl = container.value
  const canvasWidth = Math.max(600, containerEl?.clientWidth ?? 1000)
  const canvasMidY = Math.max(200, (containerEl?.clientHeight ?? 600) / 2)
  const geometry = {
    canvasWidth,
    canvasMidY,
    jitterRange: 40,
  }

  const episodeInputs: TimelineEpisodeInput[] = []
  episodeNodes.forEach((n) => {
    const raw = n.data('publishDate')
    const dateMs = typeof raw === 'number' && Number.isFinite(raw) ? raw : null
    episodeInputs.push({ id: n.id(), dateMs })
  })
  const { positions: episodePositions, missingDateIds } =
    computeEpisodeTimelinePositions(episodeInputs, geometry, 'quantile')
  timelineMissingDateCount.value = missingDateIds.length

  const positions: Record<string, TimelinePosition> = { ...episodePositions }

  // Topic / Person / Entity_* nodes — weighted mean of connected
  // episode x-positions; band offset by node type so the spine stays
  // visually distinct.
  const placeBand = (selector: string, bandOffset: number): void => {
    core.nodes(selector).forEach((n) => {
      const connectedXs: number[] = []
      n.neighborhood('node[type = "Episode"]').forEach((e) => {
        const p = positions[e.id()]
        if (p) connectedXs.push(p.x)
      })
      const meanX = weightedMeanXFromEpisodes(connectedXs)
      const x = meanX ?? TIMELINE_DEFAULT_X
      const y = canvasMidY + bandOffset + deterministicJitter(n.id(), 30)
      positions[n.id()] = { x, y }
    })
  }
  placeBand('node[type = "Topic"]', TIMELINE_TOPIC_BAND_OFFSET)
  placeBand('node[type = "TopicCluster"]', TIMELINE_TOPIC_BAND_OFFSET)
  placeBand('node[type = "Entity_person"]', TIMELINE_PERSON_BAND_OFFSET)
  placeBand('node[type = "Entity_organization"]', TIMELINE_PERSON_BAND_OFFSET)

  // Insights / Quotes ride near their parent Episode (small signed
  // offset so children don't all land on the parent's exact (x, y)).
  const placeChildren = (selector: string, yOffset: number): void => {
    core.nodes(selector).forEach((n) => {
      const eid = String(n.data('episodeId') ?? '').trim()
      const epPos = eid ? positions[eid] : undefined
      if (epPos) {
        positions[n.id()] = {
          x: epPos.x + deterministicJitter(n.id(), 20),
          y: epPos.y + yOffset + deterministicJitter(n.id(), 15),
        }
        return
      }
      // Fall back to a connected-Episode walk if data.episodeId
      // wasn't set (legacy artifacts).
      const connectedXs: number[] = []
      n.neighborhood('node[type = "Episode"]').forEach((e) => {
        const p = positions[e.id()]
        if (p) connectedXs.push(p.x)
      })
      const meanX = weightedMeanXFromEpisodes(connectedXs)
      const x = meanX ?? TIMELINE_DEFAULT_X
      positions[n.id()] = {
        x,
        y: canvasMidY + yOffset + deterministicJitter(n.id(), 15),
      }
    })
  }
  placeChildren('node[type = "Insight"]', TIMELINE_INSIGHT_OFFSET)
  placeChildren('node[type = "Quote"]', TIMELINE_QUOTE_OFFSET)

  return {
    name: 'preset',
    fit: true,
    padding: 60,
    positions: (n: NodeSingular) => {
      const p = positions[n.id()]
      return p ? { x: p.x, y: p.y } : { x: TIMELINE_DEFAULT_X, y: canvasMidY }
    },
  }
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
      syncGraphNodeVisibilityTierClasses(c)
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
    syncGraphNodeVisibilityTierClasses(c)
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
  if (lenses.bridgeRing) {
    applyBridgeNodeClass(cy)
  } else {
    cy.nodes().removeClass('graph-bridge')
  }
  if (lenses.themeClusterRegions) {
    applyThemeRegionClasses(cy, artifacts.themeClustersDoc)
  } else {
    clearThemeRegionClasses(cy)
  }
  // graph-v3 Tier 5B — when both bridge + theme lenses are on, tag each bridge
  // node with the set of themes it connects. Data-only for now (surfaced in
  // NodeDetail below); a future iteration could paint a specific glyph.
  annotateBridgesWithThemes(cy, artifacts.themeClustersDoc)
  /* graph-v3 Tier 5C/5D — enricher-based lens overlays. Fire-and-forget
     async fetches (cache-warm after first call). Each apply function is
     a no-op when the envelope is null. */
  refreshEnricherLensOverlays()
  applyDegreeVisibility(cy)
  applyViewportPreserveOrFit(cy, snap)
  // Clear the fit request flag after applying viewport (fit or preserve)
  nav.clearRequestFitAfterLoad()

  lastZoomLevel = cy.zoom()
  updateZoomPercentDisplay(cy)
  syncGraphLabelTierClasses(cy)
  syncGraphNodeVisibilityTierClasses(cy)
  attachZoomRecenter(core)
  applySearchHighlights(core)
  /** ``tryApplyPendingFocus`` first so Library/Digest **Open in graph** camera wins; episode strip skips
   * duplicate ``animateCameraToFocusedNode`` when pending focus was consumed. */
  const appliedPending = tryApplyPendingFocus(core)
  applyEpisodeRepresentativeFocusIfNeeded(core, { skipCamera: appliedPending })
  // GH #771 — restore the FSM-applied selection + camera after a redraw
  // that destroyed them. Canonical trigger: KG-second-wave full-redraw
  // arrives ~1-2 s after a Digest topic-pill handoff applied (GI-only
  // graph 214 nodes → GI+KG graph 595 nodes); the full layout rebuilds
  // Cytoscape with no selection and ``fit()`` collapses zoom to ~0.10.
  // Without this hook the FSM is "applied" but the canvas looks blank.
  //
  // The gate accepts two recoverable cases:
  //
  //   (a) ``lastResult.status === 'applied'`` — clean apply followed by a
  //       downstream redraw that destroyed selection.
  //
  //   (b) ``lastResult.status === 'failed' && reason.startsWith('stuck-timeout')`` —
  //       the FSM gave up waiting (``STUCK_TIMEOUT_MS``) but the underlying
  //       layout may still be in flight and a subsequent layoutstop can
  //       restore the user's intent. The stuck-timer hook in
  //       ``stores/graphHandoff.ts`` preserves ``envelope.cyId`` as
  //       ``lastResult.appliedCyId`` precisely for this path.
  //
  // Other failure modes (``territory-fetch-404`` etc.) are NOT restored —
  // their target may not exist in the rebuilt graph. The candidate-id
  // existence check below naturally filters these out, but we also gate
  // on the reason prefix as an explicit contract.
  const lr = graphHandoff.lastResult
  const lrRecoverable =
    lr?.status === 'applied' ||
    (lr?.status === 'failed' && (lr.reason?.startsWith('stuck-timeout') ?? false))
  // Read cy's *actual* selection state, not the Vue ref. The two diverge
  // during a full-redraw: Cytoscape's ``cy.elements().remove()`` wipes
  // selection but ``selectedNodeId.value`` isn't auto-synced (no listener
  // on selection-cleared from rebuild). Without this fix the gate
  // `!selectedNodeId.value` would be false during the KG-second-wave
  // layoutstop where selection is genuinely lost, restore would skip,
  // and the user would see ~2 s of empty selection + fit-all camera
  // until a downstream watcher eventually re-applies (GH #771 follow-up,
  // Test A timeline 2026-05-14: t+4s collapse → t+6s recover; with this
  // fix the recovery happens at t+4s in the same layoutstop).
  const cyHasSelection = core.nodes(':selected').length > 0
  if (
    !cyHasSelection &&
    !appliedPending &&
    graphHandoff.pending === null &&
    lrRecoverable
  ) {
    const restored = resolveHandoffCandidateNode(core, lr?.appliedCyId?.trim() || '')
    if (restored) {
      core.nodes().unselect()
      restored.select()
      selectedNodeId.value = restored.id()
      try {
        applyGraphSelectionDimFromNode(core, restored)
      } catch {
        /* dimming is cosmetic; never let a styling error abort restore */
      }
      // Centre the camera on the restored node with the regular
      // animation path so zoom + pan look intentional (not a fit-all
      // jump). Preserves zoom semantics via the function's
      // ``GRAPH_FOCUS_FRAME_MIN_ZOOM`` floor.
      animateCameraToFocusedNode(core, restored.id())
    }
    pendingEarlyHandoffCameraCyId = ''
  } else if (pendingEarlyHandoffCameraCyId) {
    // #967 — the full-replacement early-restore already re-selected the FSM node at
    // cy-creation (so the gate above sees ``cyHasSelection`` and skips), but the camera
    // still sits at the post-``fit()`` zoom-collapse. Centre it now that layout positions
    // are final. Always drop the marker; only centre when nothing else owns the camera
    // this pass (a pending / episode-rep focus wins when present).
    const camId = pendingEarlyHandoffCameraCyId
    pendingEarlyHandoffCameraCyId = ''
    if (!appliedPending && graphHandoff.pending === null) {
      const camNode = resolveHandoffCandidateNode(core, camId)
      if (camNode) {
        try {
          applyGraphSelectionDimFromNode(core, camNode)
        } catch {
          /* dimming is cosmetic; never let a styling error abort restore */
        }
        animateCameraToFocusedNode(core, camNode.id())
      }
    }
  }
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

  // Analytics — the size / dynamics signal: how big the graph is after this committed layout and
  // how much of it is the navigation trail. One sample per redraw → a time series of growth/shrink.
  graphAnalytics.track('graph_redraw', {
    nodes: core.nodes().length,
    edges: core.edges().length,
    trail_size: nav.trailNodeIds.length,
  })

  // F4 — production self-healing invariant + retry budget (decision #5 /
  // FSM spec § exact predicate). Set-difference check between the logical
  // artifact view (under current ego) and Cytoscape's `core.nodes()`.
  // Predicate + reconcile-action decision are pure functions in
  // `utils/graphHandoffInvariant.ts` (unit-tested in T2a); this block is
  // the Cytoscape-aware consumer that actually applies the reconciliation.
  // L6 — always stash an invariant snapshot, even on the empty-view fast
  // path. Tests read it via ``__GIKG_FSM__.lastInvariant``; absence ⇒ "no
  // layout pass ran"; empty diff ⇒ "layout ran, canvas matches view".
  let invariantMissing: string[] = []
  let invariantExtra: string[] = []
  try {
    const viewArtForInvariant = gf.viewWithEgo(focusNodeId.value)
    if (viewArtForInvariant) {
      const expected = new Set(
        toGraphElements(viewArtForInvariant).visNodes.map((v) => v.id),
      )
      const actual = new Set<string>()
      core.nodes().forEach((n) => {
        actual.add(n.id())
      })
      const { missing, extra } = computeNodeIdSetDifference(expected, actual)
      invariantMissing = missing
      invariantExtra = extra
      if (missing.length > 0) {
        const envelopeGen = graphHandoff.pending?.generation ?? null
        const alreadyRetried =
          envelopeGen !== null &&
          handoffReconcileRetryGen === envelopeGen
        const decision = decideReconcileAction(missing, alreadyRetried)
        if (decision === 'reconcile' && envelopeGen !== null) {
          // Targeted reconciliation: re-add missing visNodes from the logical
          // view. A subsequent layoutstop will run the invariant again and
          // accept divergence if still violated (single retry budget).
          handoffReconcileRetryGen = envelopeGen
          try {
            const elements = toGraphElements(viewArtForInvariant)
            const nodesToAdd = elements.visNodes.filter((v) =>
              missing.includes(v.id),
            )
            if (nodesToAdd.length > 0) {
              core.add(
                nodesToAdd.map((n) => ({
                  group: 'nodes',
                  data: { ...n, id: n.id },
                })),
              )
              console.warn(
                `[graphHandoff invariant] reconciled missing=${missing.length} via targeted core.add (retry budget consumed for gen=${envelopeGen})`,
              )
            }
          } catch (e) {
            console.warn(
              `[graphHandoff invariant] reconciliation failed: ${e instanceof Error ? e.message : String(e)}`,
            )
          }
        } else if (decision === 'accept-divergence') {
          // Either retry already used, or missing.length above threshold —
          // accept divergence + log.
          console.warn(
            `[graphHandoff invariant] divergence accepted: missing=${missing.length} sample=${missing.slice(0, 3).join(',')} retried=${alreadyRetried}`,
          )
        }
      }
    }
  } catch {
    // Invariant check must not affect runtime behaviour.
  }
  // L6 — always record (even if the try block threw, ``invariantMissing``
  // and ``invariantExtra`` default to empty arrays so the matrix can
  // distinguish "no layoutstop yet" from "layoutstop ran, view consistent").
  graphHandoff.recordInvariant(invariantMissing, invariantExtra)
  // C5 — mark the in-flight handoff as applied so the FSM transitions back
  // to `ready` and the stuck timer disarms. Without this the FSM would stay
  // in `loading_fetch` after every entry-point click and false-alarm.
  //
  // Fall back to the pending envelope's intended ``cyId`` when neither
  // ``selectedNodeId`` nor ``focusNodeId`` is set (typical on the first
  // layoutstop of a KG-second-wave load: artifacts merged but the target
  // node hasn't been resolved + selected yet). Without this fallback
  // ``lastResult.appliedCyId`` would be empty and the post-layout restore
  // hook at the top of ``finishLayoutPass`` would have nothing to anchor
  // on — selection + camera stay lost across subsequent layoutstops
  // (the GH #771 failure mode for rapid digest pills on a heavy graph).
  if (graphHandoff.pending) {
    // V2-class fix: verify the appliedCyId actually exists in cy before
    // declaring applied. ``graphHandoff.pending.cyId`` may be undefined
    // when the envelope is keyed by metadata path (Episode panel kind
    // 'episode' carries ``metadataPath`` / ``episodeId`` but no ``cyId``)
    // or absent altogether (Digest band pill — bucket id, no real node).
    // Try in order: selected → focus → cyId → episode resolver via
    // metadata/episodeId (V5 fix — Episode-panel hot-state landed
    // ``finishLayoutPass`` with empty selection + null focus and the
    // pending had no cyId, so the previous resolver candidates list was
    // empty → handoffFailed fired even though the target episode IS in
    // cy).
    const candidates = [
      selectedNodeId.value || '',
      focusNodeId.value || '',
      graphHandoff.pending.cyId || '',
      // Focus fallback (e.g. a `quote` hit has no node of its own → its Episode).
      // Without this the redraw path can't resolve a fallback-only target and would
      // hand off to handoffFailed even though the fallback IS in cy.
      nav.pendingFocusFallbackNodeId || '',
    ].filter(Boolean)
    let appliedCyId = ''
    for (const c of candidates) {
      const resolved = resolveCyNodeId(core, c)
      if (resolved) {
        appliedCyId = resolved
        break
      }
    }
    if (!appliedCyId && graphHandoff.pending.kind === 'episode') {
      // V5 — episode envelopes carry metadataPath + episodeId, not cyId.
      // First try the logical artifact resolver (finds via metadata-path
      // match on Episode-typed nodes), then fall back to direct cy id
      // pattern lookup for the well-known episode id schemes. The fallback
      // matters when the artifact-side filteredArtifact hasn't yet
      // re-rendered to include the just-appended episode at the moment
      // finishLayoutPass runs (race between artifact pinia + cy core).
      const epCy = findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
        gf.filteredArtifact,
        graphHandoff.pending.metadataPath ?? '',
        graphHandoff.pending.episodeId,
      )
      if (epCy) {
        const resolved = resolveCyNodeId(core, epCy)
        if (resolved) appliedCyId = resolved
      }
      if (!appliedCyId && graphHandoff.pending.episodeId) {
        // #775 — direct cy id lookup by episode_id. ``resolveCyNodeId``
        // already tries the 10 prefix variants; the episode_id is the
        // logical id used to derive ``__unified_ep__:<UUID>`` and
        // ``g:episode:<UUID>`` / ``k:episode:<UUID>``.
        const directResolved = resolveCyNodeId(core, graphHandoff.pending.episodeId)
        if (directResolved) appliedCyId = directResolved
      }
      if (!appliedCyId) {
        // #775 — last resort: scan cy directly for an Episode-typed node
        // whose data matches the envelope's metadataPath / episodeId.
        // ``filteredArtifact`` may lag cy briefly after a full redraw
        // (Pinia state update timing), so artifact-based resolvers can
        // miss. Iterating cy is O(N) but bounded by node count (~400
        // for production-shaped) — acceptable for the fallback path.
        const wantMeta = graphHandoff.pending.metadataPath ?? ''
        const wantEid = graphHandoff.pending.episodeId ?? ''
        const found = core.nodes().filter((n) => {
          const data = n.data() as Record<string, unknown>
          const type = String(data.type ?? data.kind ?? '')
          if (type !== 'Episode' && type !== 'episode') return false
          const props = (data.properties ?? {}) as Record<string, unknown>
          const mp = String(props.metadata_relative_path ?? data.metadata_relative_path ?? '')
          const eid = String(props.episode_id ?? data.episode_id ?? '')
          return Boolean((wantMeta && mp === wantMeta) || (wantEid && eid === wantEid))
        })
        if (found.length > 0) {
          appliedCyId = found.first().id()
        }
      }
    }
    if (appliedCyId) {
      graphHandoff.recordApplied(appliedCyId)
    } else {
      // Race window: ``filteredArtifact`` (pinia) has the target episode but
      // cy hasn't yet rendered it on this ``layoutstop`` — happens under
      // parallel-worker CPU pressure (Tier-2 P2.5 flake on ci-ui-full). If
      // the artifact-side resolver can find the target Episode by
      // ``metadataPath`` / ``episodeId``, don't fail synchronously; leave
      // the FSM in ``applying`` with the current ``pending`` and let the
      // next ``layoutstop`` re-scan cy (the stuck-timer at ``STUCK_TIMEOUT_MS``
      // still bounds pathological cases). Only fail synchronously when the
      // artifact ALSO doesn't have the target — that's a genuine miss.
      const pending = graphHandoff.pending
      const artifactHasTarget =
        pending &&
        pending.kind === 'episode' &&
        !!findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
          gf.filteredArtifact,
          pending.metadataPath ?? '',
          pending.episodeId,
        )
      if (!artifactHasTarget) {
        graphHandoff.handoffFailed(
          `apply failed: no cy node found for envelope target (cyId=${graphHandoff.pending.cyId ?? 'none'}, metadataPath=${graphHandoff.pending.metadataPath ?? 'none'})`,
        )
      }
      // If artifactHasTarget: the graphFilters watcher will fire another
      // redraw + layoutstop shortly; ``finishLayoutPass`` will re-run and
      // this time the cy-side resolver should succeed.
    }
  } else if (
    graphHandoff.state === 'applying' ||
    graphHandoff.state === 'loading_fetch' ||
    graphHandoff.state === 'loading_bootstrap' ||
    graphHandoff.state === 'loading_merge' ||
    graphHandoff.state === 'redrawing_incremental' ||
    graphHandoff.state === 'redrawing_full'
  ) {
    // No in-flight envelope, but FSM is in some intermediate state from
    // a filter-class side-effect (layout cycle, relayout, lens toggle,
    // minimap toggle that triggers a subject-driven artifact append,
    // etc.). The FSM should not stay parked in an intermediate state
    // when there's no envelope to drive the transitions — advance to
    // ``ready`` so the next user action gets a clean baseline.
    graphHandoff.advanceState('ready')
  }
  // Set cooldown to prevent immediate redraws from watchers reacting to layout state changes
  layoutCompletionCooldownUntil = Date.now() + 300 // 300ms cooldown
  releaseGraphPaintAfterLayout(core)
  // Layout just completed and the canvas may have been resizing during it
  // (Library/Search/Dashboard handoff: graph mounting + detail panel opening
  // simultaneously). If a focus action is in flight, recenter the target.
  recenterIfPending(core)
  setTimeout(() => {
    if (cy === core) recenterIfPending(core)
  }, 250)
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
  if (n.empty()) {
    return
  }
  // F3a — generation-token check point #7 (FSM spec § 8 sites). Capture
  // generation at entry; abort the animation if a newer handoff supersedes it
  // (e.g. user clicks a different node mid-animation). Also stops `cy.animate`
  // already in flight by setting a fresh `centerEles` only when fresh.
  const animateGen = graphHandoff.generation
  // Arm pending-recenter so any canvas resize over the next window recenters on
  // this node. Catches detail-panel open transitions, tab switches (Library /
  // Dashboard / Search → Graph), and other layout shifts that happen during or
  // after the camera animation.
  armPendingRecenter(cyId)
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
    if (graphHandoff.isStale(animateGen)) {
      suspendSelectedNodeZoomAnchorCorrection -= 1
      return
    }
    core.animate({
      center: { eles: centerEles },
      zoom: targetZoom,
      duration: 320,
      complete: () => {
        // F3a — generation check inside animation `complete` (FSM spec § 8 sites).
        // If the user has clicked something else mid-animation, don't continue
        // with stale recenter logic.
        if (graphHandoff.isStale(animateGen)) {
          suspendSelectedNodeZoomAnchorCorrection -= 1
          return
        }
        // Canvas may have resized during animation (detail panel opening, tab switch,
        // initial graph mount on Library/Dashboard handoff). Force resize + recenter
        // against CURRENT canvas dimensions.
        try {
          core.resize()
          core.center(centerEles)
        } catch {
          // ignore — best-effort post-animation recenter.
        }
        suspendSelectedNodeZoomAnchorCorrection -= 1
        refreshSelectedNodeZoomAnchor(core)
        lastZoomLevel = core.zoom()
        updateZoomPercentDisplay(core)
      },
    })
    // F3d — INTENTIONAL TIME-BASED SAFETY NETS. ResizeObserver may not fire
    // for every layout shift (tab switches that mount the graph for the first
    // time; async artifact loads; tab switches that resize the canvas mid-
    // animation). The 400/900/1800ms cadence catches three distinct timing
    // classes: post-paint (~400ms), post-detail-panel-open (~900ms), and
    // post-async-load (~1800ms). Each `recenterIfPending` is a no-op if the
    // pending recenter has already been consumed or expired.
    //
    // These are TIMEOUTS (real-time safety bounds), not synchronization
    // primitives — the FSM "no time-based gates" rule (concern #3) explicitly
    // carves out timeouts. Per FSM design: replace with event-driven barriers
    // only when there's a corresponding Cytoscape event that fires reliably
    // for every meaningful resize. Today there isn't one.
    // #767-C — safety-net tail timings live in
    // ``RECENTER_SAFETY_TAIL_TIMINGS_MS`` (source-of-truth +
    // behavior-contract artifact). One entry = one ``setTimeout`` armed
    // per call; the constant pins the before/after for the trim.
    for (const ms of giKgCoseLayout.RECENTER_SAFETY_TAIL_TIMINGS_MS) {
      setTimeout(() => {
        if (cy === core) recenterIfPending(core)
      }, ms)
    }
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
/** graph-v3 tier 8-3 — walk the full artifact + theme doc to find the
 *  super-theme housing a pending-focus node id; expand it so the next
 *  redraw's `tryApplyPendingFocus` can resolve the target. Safe to call
 *  with any raw id (fallback ids, bare ids, prefixed ids) — the id
 *  lookup is best-effort and returns silently on miss. */
function maybeExpandTopDownForPendingFocus(rawId: string): void {
  const full = artifacts.displayArtifact?.data
  if (!full) return
  const themeDoc = artifacts.themeClustersDoc
  if (!themeDoc?.clusters?.length) return
  const clusterToSuper = new Map<string, string>()
  for (const cl of themeDoc.clusters) {
    const cid =
      typeof cl?.graph_compound_parent_id === 'string'
        ? cl.graph_compound_parent_id.trim()
        : ''
    const sid = typeof cl?.super_theme_id === 'string' ? cl.super_theme_id.trim() : ''
    if (cid && sid) clusterToSuper.set(cid, sid)
  }
  if (clusterToSuper.size === 0) return
  const nodes = Array.isArray(full.nodes) ? full.nodes : []
  const findNode = (id: string) => {
    const wanted = id.trim()
    if (!wanted) return undefined
    for (const n of nodes) {
      if (n && n.id != null && String(n.id) === wanted) return n
    }
    return undefined
  }
  let target = findNode(rawId)
  if (!target) {
    /* Try common id-prefix normalizations before giving up. */
    if (rawId.startsWith('g:') || rawId.startsWith('k:')) {
      target = findNode(rawId.slice(2))
    } else {
      target = findNode(`g:${rawId}`) ?? findNode(`k:${rawId}`)
    }
  }
  if (!target) return
  const tcid =
    typeof (target as { themeClusterId?: unknown }).themeClusterId === 'string'
      ? String((target as { themeClusterId?: unknown }).themeClusterId).trim()
      : ''
  if (!tcid) return
  const sid = clusterToSuper.get(tcid)
  if (!sid) return
  if (topDown.isExpanded(sid)) return
  topDown.expandSuperTheme(sid)
}

function tryApplyPendingFocus(core: Core): boolean {
  // F3a — generation-token check point #6 (FSM spec § 8 sites). If a newer
  // handoff has bumped generation since this watcher fired, abandon the apply.
  const entryGen = graphHandoff.generation
  const rawId = nav.pendingFocusNodeId
  if (!rawId) return false
  const fallbackRaw = nav.pendingFocusFallbackNodeId
  let cyId = resolveCyNodeId(core, rawId)
  let usedFallback = false
  if (!cyId && fallbackRaw) {
    cyId = resolveCyNodeId(core, fallbackRaw)
    usedFallback = true
  }
  /** Do not clear pending: ``redraw`` can leave the graph mid-rebuild; a later ``finishLayoutPass`` / watcher applies. */
  if (!cyId) {
    /* graph-v3 tier 8-3 — search reveals hidden. In top-down mode the
     * search target may live under a collapsed super-theme. Look up
     * the target's themeClusterId in the FULL display artifact,
     * roll it up to super_theme_id, and expand that super-theme.
     * The store change re-derives topDownDisplayArtifact and the
     * next redraw's tryApplyPendingFocus call succeeds. */
    if (loadMode.isTopDown) {
      maybeExpandTopDownForPendingFocus(rawId)
    }
    return false
  }
  if (graphHandoff.isStale(entryGen)) {
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
  // Clear zoom anchor before animation to prevent zoom event handler from interfering with camera centering
  clearSelectedNodeZoomAnchor()
  try {
    applyGraphSelectionDimFromNode(core, focusNode)
  } catch {
    /* ignore */
  }
  // RAIL target ≠ camera target when we fell back: a `quote` search hit has no cy node of its
  // own, so ``cyId`` here is its *Episode* (camera anchor). But the user picked the quote, and
  // ``onFocusHit`` already opened the quote's detail rail from the artifact (``subject`` =
  // quote). Opening the Episode rail now would clobber it. So only (re)open the rail when we
  // resolved the PRIMARY; on a fallback, leave the subject rail on the user's actual target.
  // (Pre-#967 the cose layout stayed busy long enough that this fallback never fired here;
  // fcose goes idle fast, exposing the clobber — hence the explicit guard.)
  if (!usedFallback) {
    const rawNode = rawNodeForRailInteraction(cyId)
    void openGraphEpisodeOrNodeRail(cyId, rawNode)
  }
  const extras = [...nav.pendingFocusCameraIncludeRawIds]
  nav.clearPendingFocus()
  // Record the apply with the FSM so it transitions back to ``ready``.
  // ``finishLayoutPass`` would do this too, but on rapid sequential clicks the
  // layoutstop listener captured by an earlier ``redraw()`` often bails at the
  // ``cy !== core`` guard (a later redraw stomped its captured ``core``); the
  // FSM would then sit in ``loading_fetch`` until the stuck-timeout. Calling
  // ``recordApplied`` here gives every successful focus-apply a path to
  // ``ready`` regardless of which redraw's layoutstop wins the race.
  if (graphHandoff.pending) {
    graphHandoff.recordApplied(cyId)
  }
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

/**
 * Try to resolve and apply an in-flight FSM envelope WITHOUT going through
 * a full ``finishLayoutPass``. Used by ``onActivated`` to recover from the
 * "user tabbed away mid-handoff, came back, FSM still in-flight, no
 * layoutstop coming" UX hole. Returns true when an apply landed; FSM is
 * advanced to ``ready`` on success or ``failed`` on miss.
 *
 * Mirrors the resolver order of ``finishLayoutPass`` (selected → focus →
 * cyId → episode resolver → direct cy id → cy scan); the difference is
 * that this is callable outside the layoutstop path.
 */
function tryApplyPendingFsmEnvelopeFromTabReturn(core: Core): boolean {
  const env = graphHandoff.pending
  if (!env) return false
  // Only fire from ``loading_*`` / ``redrawing_*`` / ``applying`` states.
  // If FSM is already ``ready``, leave it alone.
  const inFlight =
    graphHandoff.state === 'loading_fetch' ||
    graphHandoff.state === 'loading_bootstrap' ||
    graphHandoff.state === 'loading_merge' ||
    graphHandoff.state === 'redrawing_incremental' ||
    graphHandoff.state === 'redrawing_full' ||
    graphHandoff.state === 'applying'
  if (!inFlight) return false
  // Bail when an artifact load OR a Cytoscape animation is in flight.
  // Both signals mean the natural ``finishLayoutPass`` chain is about
  // to (or did just) apply + animate the camera; firing apply +
  // animateCamera here would race and leave the target offscreen
  // (H2.6 Library → Digest hot-state regression). The recovery hook
  // is for the "natural chain stalled" scenario only — tab-switched
  // away mid-load, came back to a quiescent canvas with a pending
  // envelope nobody is driving.
  if (artifacts.loading) return false
  try {
    if (core.animated()) return false
  } catch {
    /* defensive: cy may be in a transitional state */
  }

  const candidates = [
    selectedNodeId.value || '',
    focusNodeId.value || '',
    env.cyId || '',
  ].filter(Boolean)
  let appliedCyId = ''
  for (const c of candidates) {
    const resolved = resolveCyNodeId(core, c)
    if (resolved) {
      appliedCyId = resolved
      break
    }
  }
  if (!appliedCyId && env.kind === 'episode') {
    const epCy = findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
      gf.filteredArtifact,
      env.metadataPath ?? '',
      env.episodeId,
    )
    if (epCy) {
      const resolved = resolveCyNodeId(core, epCy)
      if (resolved) appliedCyId = resolved
    }
    if (!appliedCyId && env.episodeId) {
      const directResolved = resolveCyNodeId(core, env.episodeId)
      if (directResolved) appliedCyId = directResolved
    }
    if (!appliedCyId) {
      const wantMeta = env.metadataPath ?? ''
      const wantEid = env.episodeId ?? ''
      const found = core.nodes().filter((n) => {
        const data = n.data() as Record<string, unknown>
        const type = String(data.type ?? data.kind ?? '')
        if (type !== 'Episode' && type !== 'episode') return false
        const props = (data.properties ?? {}) as Record<string, unknown>
        const mp = String(props.metadata_relative_path ?? data.metadata_relative_path ?? '')
        const eid = String(props.episode_id ?? data.episode_id ?? '')
        return Boolean((wantMeta && mp === wantMeta) || (wantEid && eid === wantEid))
      })
      if (found.length > 0) appliedCyId = found.first().id()
    }
  }
  if (appliedCyId) {
    const n = core.$id(appliedCyId)
    if (!n.empty()) {
      try {
        core.nodes().unselect()
        ;(n.first() as NodeSingular).select()
        selectedNodeId.value = appliedCyId
        applyGraphSelectionDimFromNode(core, n.first() as NodeSingular)
      } catch {
        /* ignore */
      }
      // Camera-animate IS required here. The naive concern was that
      // ``finishLayoutPass`` would later fire its own animate and the
      // two would race — but ``recordApplied`` below clears
      // ``graphHandoff.pending``, which is precisely what
      // ``finishLayoutPass``'s apply branch keys off. With pending
      // cleared, ``finishLayoutPass`` only advances FSM state (no
      // animate). So this animate is the ONLY camera centering for
      // the tab-return path; omitting it leaves the camera at default
      // layout coords (Tier-2 P1.1 cold-click regression: rendered=
      // (513, -470) for an episode that should be near viewport
      // center).
      try {
        animateCameraToFocusedNode(core, appliedCyId)
      } catch {
        /* ignore */
      }
    }
    graphHandoff.recordApplied(appliedCyId)
    return true
  }
  // Target not in cy → can't apply yet. Do NOT mark failed (a layoutstop
  // may still drive apply correctly); leave FSM in-flight + let stuck-
  // timer surface the failure if no progress.
  return false
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
  /* graph-v3 E — mirror the on-screen canvas token so PNG exports match
     what the user sees. Falls through to --ps-canvas on light theme where
     --ps-graph-canvas aliases it. */
  try {
    const root = document.documentElement
    const g = getComputedStyle(root).getPropertyValue('--ps-graph-canvas').trim()
    if (g.length) return g
    const v = getComputedStyle(root).getPropertyValue('--ps-canvas').trim()
    if (v.length) return v
  } catch {
    /* ignore */
  }
  return '#0a0d10'
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

  // Only preserve viewport if not explicitly requesting a fit after load
  if (!nav.requestFitAfterLoad) {
    pendingViewportPreserve = captureSelectedViewportAnchor(c)
  } else {
    pendingViewportPreserve = null
  }
  const name = preferredLayout.value
  // RFC-080 V3: timeline reads cy node data (publishDate, episodeId)
  // so it builds its preset spec live; other layouts use static opts.
  const layoutSpec =
    name === 'timeline'
      ? timelineLayoutSpec(c)
      : { ...layoutOptionsFor(name), name }
  let lo: { stop: () => void; one: (ev: string, fn: () => void) => void; run: () => void }
  try {
    lo = eles.layout(layoutSpec as never) as typeof lo
  } catch {
    releaseGraphCanvasLayoutHold()
    return
  }
  activeElesLayout = lo
  // F2 — advance FSM through loading_merge → redrawing_full when a relayout starts.
  // The orchestrator runtime now tracks state at every barrier point.
  if (graphHandoff.state === 'loading_merge') {
    graphHandoff.advanceState('redrawing_full')
  } else if (
    graphHandoff.state === 'loading_fetch' ||
    graphHandoff.state === 'loading_bootstrap'
  ) {
    // V5 fix: hot-state Episode panel "Open in graph" fires
    // ``handoffRequested`` (state → loading_fetch) and then calls
    // ``artifacts.appendRelativeArtifacts`` directly — bypassing the
    // ``loadEpisodeSliceForTerritoryStrip`` path that would normally
    // advance loading_fetch → loading_merge. When that artifact change
    // triggers this ``redraw()``, FSM is still in loading_fetch and
    // the transition to ``redrawing_full`` is invalid — the FSM sits
    // until the 15s stuck-timer fires (V5 reproducer). Walk through
    // ``loading_merge`` first so the pipeline reaches the apply phase.
    graphHandoff.advanceState('loading_merge')
    graphHandoff.advanceState('redrawing_full')
  } else if (graphHandoff.state === 'idle' || graphHandoff.state === 'ready') {
    // Layout from internal scheduler (theme change, lens toggle, etc.) — start
    // a fresh redraw cycle from quiescent state.
    graphHandoff.advanceState('redrawing_full')
  }
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
    // F2 — advance redrawing_full → applying via the FSM event API.
    graphHandoff.notifyLayoutStop()
    finishLayoutPass(cy)
  })
  try {
    lo.run()
  } catch {
    activeElesLayout = null
    releaseGraphCanvasLayoutHold()
  }
}

/**
 * Schedule a debounced redraw to prevent rapid-fire destroy/recreate cycles
 * when multiple watchers fire in sequence (e.g., filteredArtifact changes).
 */
function scheduleRedraw(): void {
  // Skip if currently in a redraw cycle or actively laying out
  if (redrawGateDepth > 0 || activeElesLayout !== null) {
    redrawPending = true
    return
  }
  // V5 fix: previously this set ``redrawPending = true`` and bailed, but
  // ``redrawPending`` is only flushed at the END of a running ``redraw()``
  // (the ``finally`` block at the bottom of ``redraw``). With no redraw
  // running, the pending flag never gets consumed → FSM-pipeline-driving
  // redraws triggered during the post-layout cooldown window are lost
  // entirely. Episode-panel "Open in graph" hot-state hits this:
  // ``handoffRequested`` → ``loading_fetch``, then ``appendRelativeArtifacts``
  // → filteredArtifact watcher → scheduleRedraw → bails on cooldown → no
  // redraw fires → FSM stuck for 15 s until stuck-timer (V5).
  //
  // Schedule a deferred redraw that fires AFTER the cooldown ends so the
  // FSM-driving pipeline actually runs.
  if (Date.now() < layoutCompletionCooldownUntil) {
    redrawPending = true
    if (redrawDebounceTimer) {
      clearTimeout(redrawDebounceTimer)
    }
    const remaining = Math.max(0, layoutCompletionCooldownUntil - Date.now())
    // Add the existing 150ms debounce ON TOP of the cooldown remainder so
    // we don't fire mid-cooldown and so subsequent watchers coalesce.
    redrawDebounceTimer = setTimeout(() => {
      redrawDebounceTimer = null
      // Re-enter scheduleRedraw rather than calling redraw() directly so
      // we re-check redrawGateDepth / activeElesLayout / cooldown (which
      // may have been bumped again in the interim).
      if (redrawPending) {
        redrawPending = false
        scheduleRedraw()
      }
    }, remaining + 150)
    return
  }
  if (redrawDebounceTimer) {
    clearTimeout(redrawDebounceTimer)
  }
  // #767-B — debounce window source-of-truth + behavior contract:
  // ``redrawDebounceMs(hasPendingHandoff)``. Returns 0 ms when an FSM
  // envelope is pending (cross-surface handoff in flight), 150 ms
  // otherwise (coalesce Vue nextTick cascades).
  redrawDebounceTimer = setTimeout(() => {
    redrawDebounceTimer = null
    redraw()
  }, giKgCoseLayout.redrawDebounceMs(graphHandoff.pending !== null))
}

function redraw(): void {
  if (redrawGateDepth > 0) {
    redrawPending = true
    return
  }
  redrawGateDepth += 1
  try {
    const priorCore = cy
    const viewArtPreview = gf.viewWithEgo(focusNodeId.value)
    const nextNodeIdSet = new Set<string>()
    if (viewArtPreview) {
      for (const v of toGraphElements(viewArtPreview).visNodes) {
        nextNodeIdSet.add(v.id)
      }
    }
    const selCount = artifacts.selectedRelPaths.length
    const egoCur = priorEgoBeforeWatcherClear
    const selectionGrew =
      selCount > lastSelectedRelPathsCountAfterLayout &&
      lastSelectedRelPathsCountAfterLayout > 0
    const egoUnchanged = egoCur === lastCommittedEgoFocusCyId
    
    // Incremental ONLY when expanding the SAME cluster/ego (appending members from rail Load).
    // If ego changed (user clicked a DIFFERENT cluster card), do a full redraw to avoid unbounded growth.
    const egoChanged = egoCur !== lastCommittedEgoFocusCyId && egoCur !== '' && lastCommittedEgoFocusCyId !== ''
    
    /**
     * ``selectionReplaced`` and ``contextSwitch`` are hard-coded ``false`` and load-bearing — do
     * NOT delete without an FSM-based replacement (planned for C6, /Users/markodragoljevic/.claude/plans/in-this-b-tanch-gentle-pillow.md).
     * They invert the incremental-layout gate on lines below: with both ``false``, the predicate
     * becomes "incremental allowed unless egoChanged or external-nav", which is the *opposite*
     * of the original gate. Removing them reverts behaviour to a state nobody has tested in
     * months. The C2 stabilization keeps them as documented inverted gates; C6 replaces both
     * with explicit FSM transitions (``redrawing_incremental`` vs ``redrawing_full``).
     */
    const selectionReplaced = false
    const contextSwitch = false

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
    
    // Check if this is an external navigation (Digest/Library) - if so, force full layout
    const isExternalNavigation =
      artifacts.currentLoadSource === 'digest-external' ||
      artifacts.currentLoadSource === 'subject-external'
    
    // #775 — incremental layout was designed for small expansions
    // (NodeDetail Load, sibling-merge: typically 5-30 added nodes). Large
    // additions (whole episode's worth, 100+ nodes) hit Cytoscape COSE
    // convergence failures on the filtered layout collection: edges
    // reference existing-but-not-included nodes; COSE can't position new
    // nodes against anchors it doesn't see. Falling back to full redraw
    // is more reliable and the user-visible difference is minimal at
    // this scale (full graph re-layout vs partial new-only layout).
    const INCREMENTAL_LAYOUT_MAX_ADDS = 100
    const tooBigForIncremental = addedNodeIds.size > INCREMENTAL_LAYOUT_MAX_ADDS

    if (
      !egoChanged &&
      !selectionReplaced &&
      !contextSwitch &&
      !isExternalNavigation &&  // Force full layout for external navigation
      !tooBigForIncremental &&  // #775 — large adds → full redraw
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

    if (!useIncrementalLayout) {
      graphContentHiddenUntilLayout.value = true
    }

    if (useIncrementalLayout && priorCore) {
      // True incremental: add/remove elements on existing instance, no canvas blank
      // Capture current viewport to preserve zoom/pan after incremental layout,
      // UNLESS requestFitAfterLoad is set (external load without focus target)
      if (!nav.requestFitAfterLoad) {
        pendingViewportPreserve = captureSelectedViewportAnchor(priorCore)
      } else {
        pendingViewportPreserve = null
      }
      
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
        destroyCy()
        el.innerHTML =
          '<p class="p-4 text-sm text-muted">Load artifacts and use "Load selected" to render the graph.</p>'
        degreeHistogramCounts.value = {}
        graphCyNodeCount.value = 0
        releaseGraphCanvasLayoutHold()
        lastSelectedRelPathsCountAfterLayout = 0
        lastCommittedFilteredNodeIds.clear()
        lastCommittedEgoFocusCyId = ''
        return
      }

      const elements = toCytoElements(art, {
        enableAggregatedEdges: lenses.aggregatedEdges,
      })
      const nodeCount = elements.filter((x) => !('source' in x.data)).length
      if (nodeCount === 0) {
        destroyCy()
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

      // Remove elements no longer in the graph
      const nextEleIds = new Set(elements.map((e) => e.data.id))
      const toRemove = priorCore.elements().filter((ele) => !nextEleIds.has(ele.id()))
      console.log('[GraphCanvas incremental] Removing:', toRemove.length, 'elements')
      priorCore.remove(toRemove)

      // Add new elements to existing instance
      const existingIds = new Set<string>()
      priorCore.elements().forEach((ele) => {
        const id = ele.id()
        if (id) existingIds.add(id)
      })
      const toAdd = elements.filter((e) => {
        const id = e.data.id
        return id && !existingIds.has(id)
      })
      console.log('[GraphCanvas incremental] Adding:', toAdd.length, 'elements', 'Total after:', priorCore.elements().length + toAdd.length)
      priorCore.add(toAdd)

      // Seed positions for new nodes
      seedPositionsForIncrementalAppend(priorCore, addedNodeIds)

      graphCyNodeCount.value = nodeCount

      const layoutGen = graphLayoutGate.bump()
      const layoutName = preferredLayout.value
      const layoutCollection = priorCore.elements().filter((ele) => {
        if (ele.isNode()) {
          return addedNodeIds.has(ele.id())
        }
        return (
          addedNodeIds.has(ele.source().id()) || addedNodeIds.has(ele.target().id())
        )
      })

      let initialLo: { stop: () => void; one: (ev: string, fn: () => void) => void; run: () => void }
      try {
        const layoutOpts = layoutOptionsFor(layoutName)
        const initialSpec =
          layoutName === 'timeline'
            ? timelineLayoutSpec(priorCore)
            : { ...layoutOpts, name: layoutName }
        initialLo = layoutCollection.layout(initialSpec as never) as typeof initialLo
      } catch {
        releaseGraphCanvasLayoutHold()
        return
      }

      activeElesLayout = initialLo
      // V5 — notify the FSM that a layout is running so the 15 s stuck-timer
      // doesn't fire on big incremental layouts (300+ added elements can
      // take >15 s on real-backend hot-state Episode-panel handoff).
      graphHandoff.notifyLayoutStart()
      // #775 — graceful layout timeout. The Cytoscape COSE layout on the
      // filtered incremental collection sometimes doesn't converge (the
      // collection contains edges referencing existing-but-not-included
      // nodes; COSE struggles to position new nodes relative to anchors
      // it doesn't see). Without this, the FSM stays in loading_fetch
      // indefinitely (the stuck-timer now reschedules while a layout is
      // active — see notifyLayoutStart). Cap at 8 s: most incremental
      // layouts complete in <2 s; legitimate big ones in <5 s; 8 s is a
      // safety net, not a normal completion time.
      let layoutSettled = false
      const layoutTimeoutMs = 8000
      const layoutTimeout = setTimeout(() => {
        if (layoutSettled) return
        layoutSettled = true
        console.warn(
          `[GraphCanvas incremental] layout timeout after ${layoutTimeoutMs}ms — aborting and forcing layoutstop (added=${addedNodeIds.size} total=${priorCore.elements().length})`,
        )
        try {
          initialLo.stop()
        } catch {
          /* ignore */
        }
        graphHandoff.notifyLayoutStop()
        if (graphLayoutGate.isStale(layoutGen)) return
        if (activeElesLayout === initialLo) {
          activeElesLayout = null
        }
        if (!cy || cy !== priorCore) {
          releaseGraphCanvasLayoutHold()
          return
        }
        finishLayoutPass(priorCore)
      }, layoutTimeoutMs)
      initialLo.one('layoutstop', () => {
        if (layoutSettled) return
        layoutSettled = true
        clearTimeout(layoutTimeout)
        graphHandoff.notifyLayoutStop()
        const stale = graphLayoutGate.isStale(layoutGen)
        if (stale) {
          return
        }
        if (activeElesLayout === initialLo) {
          activeElesLayout = null
        }
        if (!cy || cy !== priorCore) {
          releaseGraphCanvasLayoutHold()
          return
        }
        finishLayoutPass(priorCore)
      })
      try {
        initialLo.run()
      } catch {
        activeElesLayout = null
        releaseGraphCanvasLayoutHold()
      }
      return
    }

    // FULL REDRAW PATH
    console.log('[GraphCanvas] FULL REDRAW - destroying and recreating graph')
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
    // RFC-080 V3 — timeline computes per-node positions from live cy
    // data, so it returns a self-contained spec (`{ name: 'preset',
    // positions, ... }`). For other layouts the caller still spreads
    // `name` over the static opts.
    //
    // #767-A — for cose redraws under an external-nav load source, cap
    // ``numIter`` by node count (``200 + 8 × N``). The default 2500
    // iterations is calibrated for first paint on an unknown corpus
    // size; the actual converge-by iteration is well below 2500 for
    // graphs ≤ 300 nodes. External-nav redraws (Library / Digest
    // "Open in graph") are the visible latency tax — capping shaves
    // 100-600 ms off cose convergence without affecting final layout
    // quality (cose has already settled by the cap).
    const isExternalNavRedraw =
      layoutName === 'fcose' &&
      (artifacts.currentLoadSource === 'subject-external' ||
        artifacts.currentLoadSource === 'digest-external')
    const numIterCap = isExternalNavRedraw
      ? giKgCoseLayout.giKgCoseNumIterCapped(nodeCount)
      : undefined
    const layoutOpts = layoutOptionsFor(layoutName, { numIter: numIterCap })
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

    // GH #771 follow-up — restore selection on the freshly built ``cy`` IMMEDIATELY.
    // Previously this lived at the bottom of ``redraw()`` (still does, as a
    // belt-and-braces fallback); on heavy KG-second-wave redraws (213 → 595
    // nodes) the gap between ``cytoscape({elements})`` returning and the
    // bottom of ``redraw()`` was ~1 second of user-visible "selection lost +
    // fit-all camera". Selecting at cy-creation closes that window —
    // ``cy.$(':selected')`` reads correct from the first frame the new cy
    // is observable. ``selectedNodeId.value`` is the Vue ref that ``redraw``
    // never clears for the incremental-append path (only the full-replacement
    // path clears it), so during a KG-second-wave merge it still carries
    // the user's intent from the prior FSM apply.
    const earlySid = selectedNodeId.value
    pendingEarlyHandoffCameraCyId = ''
    if (earlySid) {
      const earlyNode = core.$id(earlySid)
      if (earlyNode.length > 0) {
        earlyNode.select()
      }
    } else {
      // #967 — the full-replacement path cleared ``selectedNodeId`` above, so the
      // ``selectedNodeId``-driven restore can't fire. If the FSM's last apply is
      // recoverable (and no fresh handoff is pending), re-select that node NOW to close
      // the no-selection window that the ego→full reload otherwise leaves open until
      // ``finishLayoutPass``. Positions aren't final yet, so defer the camera centre to
      // layoutstop via ``pendingEarlyHandoffCameraCyId`` (selecting pre-layout is safe;
      // animating the camera to a not-yet-laid-out node is not).
      const lr0 = graphHandoff.lastResult
      const lr0Recoverable =
        lr0?.status === 'applied' ||
        (lr0?.status === 'failed' && (lr0.reason?.startsWith('stuck-timeout') ?? false))
      if (lr0Recoverable && graphHandoff.pending === null) {
        const earlyHandoff = resolveHandoffCandidateNode(core, lr0?.appliedCyId?.trim() || '')
        if (earlyHandoff) {
          earlyHandoff.select()
          selectedNodeId.value = earlyHandoff.id()
          pendingEarlyHandoffCameraCyId = earlyHandoff.id()
        }
      }
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
      const initialSpec =
        layoutName === 'timeline'
          ? timelineLayoutSpec(core)
          : { ...layoutOpts, name: layoutName }
      initialLo = layoutCollection.layout(initialSpec as never) as typeof initialLo
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
    // Advance FSM state through the redraw barrier so ``layoutstop`` lands
    // in a state where the FSM accepts the transition. Without this,
    // ``redraw()`` runs ``initialLo`` while state is still ``loading_fetch``
    // (the territory-strip's ``advanceState('loading_merge')`` hasn't fired
    // yet because the subject watcher runs *after* the initial
    // ``appendRelativeArtifacts → filteredArtifact change`` cascade). The
    // layoutstop event is then dropped (loading_fetch isn't a valid
    // pre-state for layoutstop) and ``finishLayoutPass`` runs with the FSM
    // marooned in ``loading_fetch``. Walking the FSM through the proper
    // pipeline here is what the state-walking integration test asserts.
    if (
      graphHandoff.state === 'loading_fetch' ||
      graphHandoff.state === 'loading_bootstrap'
    ) {
      graphHandoff.advanceState('loading_merge')
    }
    if (graphHandoff.state === 'loading_merge') {
      graphHandoff.advanceState('redrawing_full')
    }
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
      if (cy !== core) return
      core.resize()
      // While a focus action is in flight, recenter on the focus target so the
      // node stays in the middle of the visible graph viewport (catches detail
      // panel transitions, tab switches, layout shifts).
      recenterIfPending(core)
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
    // Analytics — a tap on a node is a user gesture (unlike 'select', which also fires on
    // programmatic focus), so it's the honest "user clicked a node" signal.
    core.on('tap', 'node', (e) => {
      const node = e.target as NodeSingular
      graphAnalytics.track('graph_node_tap', {
        id: node.id(),
        kind: String(node.data('type') ?? 'unknown'),
      })
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

    /* graph-v3 D — reveal neighbourhood on hover without needing a click.
       Selection wins: if a node is :selected, skip so tapping-then-hovering
       doesn't strip the selection dim. Transition-duration 120ms in the
       stylesheet smooths the flicker as the cursor slides between nodes. */
    core.on('mouseover', 'node', (e) => {
      try {
        const target = e.target as NodeSingular
        maybeSetBridgeHoverTitle(container.value, target)
        if (core.nodes(':selected').length > 0) return
        applyGraphSelectionDimFromNode(core, target)
      } catch {
        /* ignore */
      }
    })
    core.on('mouseout', 'node', () => {
      try {
        clearBridgeHoverTitle(container.value)
        if (core.nodes(':selected').length > 0) return
        clearGraphSelectionDim(core)
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
      // graph-v3 tier 7-3 — tapping the empty canvas clears legend focus too.
      themeFocus.clearFocus()
      return
    }
    if (typeof t.isNode === 'function' && t.isNode()) {
      /* graph-v3 tier 8-2 — tapping a SuperTheme node in top-down mode
       * toggles its expand state instead of running the normal select
       * flow. The store change re-derives `topDownDisplayArtifact`
       * which re-renders the graph with the projected children. */
      if (loadMode.isTopDown && t.data('type') === 'SuperTheme') {
        topDown.toggleSuperTheme(t.id())
        return
      }
      // Tapping a node hands control back to the selection-dim path; drop
      // any active theme focus so the two dim sources don't fight.
      themeFocus.clearFocus()
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
      // graph-v3 tier 8-2 — SuperTheme tap is handled by the `tap` handler
      // (expand toggle); don't open the detail rail for synthetic nodes.
      if (loadMode.isTopDown && t.data('type') === 'SuperTheme') return
      // F1.2 — fire FSM canvasTapped before opening rail. The tap is direct
      // (no load barriers needed); FSM transitions to `applying` then `ready`
      // via supersession or finishLayoutPass apply phase.
      const rawNode = rawNodeForRailInteraction(id)
      const isEpisodeNode = rawNode?.type === 'Episode'
      graphHandoff.canvasTapped({
        kind: isEpisodeNode ? 'episode' : 'graph-node',
        cyId: id,
        source: 'canvas-tap',
        loadSource: 'graph-internal',
        camera: { kind: 'center', cyId: id },
      })
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
      /* graph-v3 tier 8-6 — dbltap on a SuperTheme in top-down mode is
       * semantically the same as tapping it: expand. Projected children
       * are already visible (their super-theme is already expanded, or
       * they wouldn't render), and the ego path uses `displayArtifact`
       * (the FULL artifact), not the top-down slice — so there's no
       * super-theme-expansion work to do for children here. */
      if (loadMode.isTopDown && t.data('type') === 'SuperTheme') {
        topDown.toggleSuperTheme(id)
        return
      }
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
      // F1.3 — fire FSM expansionRequested. Decision #5: expansion always queues
      // (additive; cancelling loses user work). Definition X: graph-internal =
      // preserves layout. Camera stays where the user left it.
      graphHandoff.expansionRequested({
        kind: 'graph-node',
        cyId: id,
        source: 'double-tap-expand',
        loadSource: 'graph-internal',
        camera: { kind: 'preserve' },
      })
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
        scheduleRedraw()
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

// PRD-033 FR5.1 — re-emphasize when the active search context changes (incl. clear).
watch(
  () => activeSearchContext.byEpisode,
  () => {
    safeGraphWatch('contextEmphasis', () => {
      const c = cy
      if (c) applyContextEmphasis(c)
    })
  },
  { flush: 'post' },
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
        // Detect if this is an incremental append (superset) vs a full replacement
        const currentNodeIds = new Set<string>()
        if (gf.filteredArtifact) {
          for (const n of gf.filteredArtifact.data.nodes || []) {
            if (n.id && typeof n.id === 'string') {
              currentNodeIds.add(n.id)
            }
          }
        }

        // Check if ALL previous nodes are still present (superset = incremental append)
        let isIncrementalAppend = false
        if (prevFilteredArtifactNodeIds.size > 0 && currentNodeIds.size >= prevFilteredArtifactNodeIds.size) {
          isIncrementalAppend = true
          for (const id of prevFilteredArtifactNodeIds) {
            if (!currentNodeIds.has(id)) {
              isIncrementalAppend = false
              break
            }
          }
        }

        // Update tracked set for next comparison
        prevFilteredArtifactNodeIds = currentNodeIds

        // F3c — replaces the previous early-return at this site (was the
        // load-bearing-wrong gate from the original WIP analysis). Per the FSM
        // spec § "Always redraw on growth": during a node-id-set growth event,
        // schedule a redraw unconditionally. The disruptive-state-reset ops
        // below (focus clear, subject clear, etc.) are still gated to incremental
        // appends so we don't tear down the rail when the user is just expanding
        // the existing graph (e.g. NodeDetail Load, double-tap expand).
        const isExternalNavigation =
          artifacts.currentLoadSource === 'digest-external' ||
          artifacts.currentLoadSource === 'subject-external'

        // Clear load source — observed once, regardless of the path taken below.
        artifacts.clearLoadSource()

        if (isIncrementalAppend && !nav.pendingFocusNodeId && !isExternalNavigation) {
          // Internal incremental append (auto-merge, double-tap expand, etc.):
          // schedule redraw so Cytoscape picks up the new nodes, but DON'T
          // tear down the user's selection / subject / highlights. C7's
          // self-healing invariant in finishLayoutPass catches any divergence
          // between the merged artifact and the cy core.
          scheduleRedraw()
          return
        }

        // Full replacement: proceed with normal cleanup
        ge.resetForNewArtifact()
        priorEgoBeforeWatcherClear = focusNodeId.value?.trim() ?? ''
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
        // Suppress the restore-preference emission when either:
        //   (a) a cross-surface handoff is already in flight, OR
        //   (b) the proposed restore target normalises to the same
        //       logical node the FSM JUST applied successfully.
        //
        // (a) catches the "click 2nd digest pill" case (Symptom 2): the
        // stale prior-pill id would otherwise be re-dispatched as a fresh
        // envelope, bump the generation, and supersede the click intent.
        //
        // (b) catches the "heavy graph, KG loads later" case (Symptom 3):
        // a successful FSM cycle applies (e.g.) ``g:topic:foo`` from a
        // GI-only artifact set. KG data arrives via downstream watchers,
        // grows ``filteredArtifact`` (214 → 595 nodes), and the focus-apply
        // path overwrites ``subject.graphNodeCyId`` with ``k:topic:foo``
        // (FSM resolution prefers k: when both g: and k: exist for the
        // same logical id). The meta-watcher would then emit a fresh
        // restore-preference for ``k:topic:foo`` — same logical target,
        // wrong viewport math because the layout is still settling, and
        // the camera collapses to fit-all. ``sameLogicalCyId`` strips the
        // 1-char ``g:`` / ``k:`` prefix used by GI/KG artifacts so the
        // FSM's last applied id and the proposed restore id compare as
        // equal whenever they refer to the same logical topic / entity.
        const handoffInFlight = graphHandoff.pending !== null
        const sameLogicalCyId = (a: string, b: string): boolean => {
          if (!a || !b) return false
          if (a === b) return true
          const strip = (s: string): string =>
            s.startsWith('g:') || s.startsWith('k:') ? s.slice(2) : s
          return strip(a) === strip(b)
        }
        const lastAppliedCyId =
          graphHandoff.lastResult?.status === 'applied'
            ? graphHandoff.lastResult.appliedCyId?.trim() || ''
            : ''
        const restoreTargetAlreadyApplied = (cyId: string): boolean =>
          !!lastAppliedCyId && sameLogicalCyId(cyId, lastAppliedCyId)
        // Architectural principle: **the details / subject rail must not
        // drive graph state via watchers.** Only deliberate user clicks
        // route through the FSM. This meta-watcher used to fire
        // ``handoffRequested({source:'restore-preference'})`` whenever the
        // artifact set changed — that was a side-effect → handoff loop.
        // The post-layout restore-applied hook at the top of
        // ``finishLayoutPass`` is the canonical path: it re-selects the
        // FSM's last applied target after any redraw that destroyed
        // selection. The watcher's role here is reduced to:
        //   1. Capture the prior subject's intent in ``pendingFocusNodeId``
        //      via ``nav.requestFocusNode`` (legacy pending-focus path,
        //      consumed by the watcher at this file's nav.pendingFocusNodeId
        //      watch).
        //   2. ``scheduleRedraw()`` for the new artifact set.
        // No autonomous FSM envelope dispatch from a side-effect path.
        if (
          restoreGraphNodeId &&
          !handoffInFlight &&
          !restoreTargetAlreadyApplied(restoreGraphNodeId)
        ) {
          nav.requestFocusNode(restoreGraphNodeId)
        } else if (
          restoreEpisodeCyId &&
          !handoffInFlight &&
          !restoreTargetAlreadyApplied(restoreEpisodeCyId)
        ) {
          nav.requestFocusNode(restoreEpisodeCyId)
        }
        scheduleRedraw()
      })
    })
  },
  { flush: 'post' },
)

// Watch for pending focus changes to apply focus even when artifacts don't change
// (e.g., clicking multiple topic pills from the same episode in Digest)
watch(
  () => nav.pendingFocusNodeId,
  (newFocusId) => {
    if (!newFocusId || !cy) return
    const core = cy
    // Apply immediately ONLY when the PRIMARY target is already in cy. Resolve via
    // ``resolveCyNodeId`` (not a bare ``$id``) so prefixed ids (``g:insight:…``) match.
    // Do NOT apply the fallback here: the primary (e.g. a quote that becomes a node after
    // the handoff's redraw) must get first chance — applying the fallback now would
    // preempt it and focus the wrong node.
    if (resolveCyNodeId(core, newFocusId)) {
      void nextTick(() => {
        safeGraphWatch('pendingFocus', () => {
          if (cy) {
            tryApplyPendingFocus(cy)
          }
        })
      })
      return
    }
    // Primary not in cy yet. A handoff redraw may still add it (e.g. a quote that becomes a
    // node once its episode's gi.json merges), and that redraw + its fcose layout (#967) can
    // take many frames — so a single-frame idle-check races the layout and would apply the
    // fallback before the primary renders. Instead POLL: each frame, bail if the handoff
    // settled elsewhere (``finishLayoutPass``), apply the moment the primary appears, and keep
    // waiting while a redraw/layout is in flight OR a small frame budget remains. Only once the
    // canvas is genuinely IDLE *and* that budget is spent (no redraw is coming → the primary
    // will never load) do we resolve the fallback (``tryApplyPendingFocus`` tries primary then
    // fallback) or fail fast — instead of waiting out the 15s stuck-timeout.
    let framesWaited = 0
    /* ~650ms is fine when the render loop is uncontended, but on ci-ui-full's
       parallel-worker runs the same test surface (Tier-2 P2.5) intermittently
       needs more time. Widen to ~2 s (120 frames at 60 Hz) so the polling
       loop stays alive long enough for the eventual layoutstop that adds the
       target node under CPU pressure. The stuck-timer at STUCK_TIMEOUT_MS
       (15 s) still bounds pathological cases. */
    const FOCUS_RESOLVE_FRAME_BUDGET = 120
    const pollForFocusTarget = (): void => {
      if (!cy || !graphHandoff.pending) return // resolved / failed elsewhere
      if (resolveCyNodeId(cy, newFocusId)) {
        safeGraphWatch('pendingFocus', () => {
          if (cy) tryApplyPendingFocus(cy)
        })
        return
      }
      const idle =
        !redrawPending &&
        redrawDebounceTimer == null &&
        redrawGateDepth === 0 &&
        !graphContentHiddenUntilLayout.value
      if (!idle || framesWaited++ < FOCUS_RESOLVE_FRAME_BUDGET) {
        requestAnimationFrame(pollForFocusTarget)
        return
      }
      // Idle + budget spent + primary still absent → it will never load. Resolve the fallback
      // (its episode) or fail fast.
      const fb = nav.pendingFocusFallbackNodeId
      if (fb && resolveCyNodeId(cy, fb)) {
        safeGraphWatch('pendingFocus', () => {
          if (cy) tryApplyPendingFocus(cy)
        })
        return
      }
      graphHandoff.handoffFailed(
        `no graph node for focus target "${newFocusId}"` + (fb ? ` (fallback "${fb}")` : ''),
      )
      nav.clearPendingFocus()
    }
    void nextTick(() => requestAnimationFrame(pollForFocusTarget))
  },
  { flush: 'post' },
)

// #6 L0 — the breadcrumb trail grew (a node navigated-to in the graph rail): redraw so the trail
// nodes viewWithEgo now unions in get rendered + connected. Guarded by the usual redraw scheduler.
watch(
  () => nav.trailNodeIds,
  () => {
    safeGraphWatch('trailChanged', () => scheduleRedraw())
  },
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

/* graph-v3 R-V — lens toggles for theme-cluster regions + bridge take
   effect without a full re-layout. Class add/remove only, so pan / zoom /
   selection state is preserved. Also watches the theme-cluster doc
   itself: switching corpora reloads the artifact, so the region tint
   needs to refresh even if the lens flag hasn't changed. */
watch(
  () => lenses.themeClusterRegions,
  (on) => {
    safeGraphWatch('themeClusterRegions', () => {
      const c = cy
      if (!c) return
      if (on) applyThemeRegionClasses(c, artifacts.themeClustersDoc)
      else clearThemeRegionClasses(c)
    })
  },
)

watch(
  () => artifacts.themeClustersDoc,
  () => {
    safeGraphWatch('themeClustersDoc', () => {
      const c = cy
      if (!c) return
      if (lenses.themeClusterRegions) {
        applyThemeRegionClasses(c, artifacts.themeClustersDoc)
      }
    })
  },
)

/* graph-v3 Tier 5C/5D — enricher-lens toggles route through
   refreshEnricherLensOverlays which handles both enable + disable
   (fetch + apply / clear). Each watcher is intentionally tiny — the
   real work is factored into the helper so all four toggles share the
   same fetch cadence and error handling. */
watch(
  () => [
    lenses.velocityHalo,
    lenses.personCredibility,
    lenses.consensusEdges,
    lenses.coGuestEdges,
    lenses.personCommunities,
  ] as const,
  () => {
    safeGraphWatch('enricherLenses', () => refreshEnricherLensOverlays())
  },
)

watch(
  () => lenses.bridgeRing,
  (on) => {
    safeGraphWatch('bridgeRing', () => {
      const c = cy
      if (!c) return
      if (on) applyBridgeNodeClass(c)
      else c.nodes().removeClass('graph-bridge')
    })
  },
)

/** graph-v3 tier 8-2 — reset expansions when leaving top-down mode.
 *  Not persisted across mode flips: entering top-down again starts at
 *  the clean super-theme preview. USERPREFS-1 still remembers the
 *  overall mode so cross-tab / cross-device state carries. */
watch(
  () => loadMode.isTopDown,
  (isTopDown, wasTopDown) => {
    if (wasTopDown && !isTopDown) topDown.clearExpanded()
  },
)

/** graph-v3 tier 7-3 — legend focus bus.
 *  React to `useGraphThemeFocusStore` changes. Skip when a node is
 *  currently :selected (existing selection-dim wins) so a legend click
 *  can't strip the user's node-selection context. */
watch(
  () => themeFocus.focusedThemeIds,
  (ids) => {
    safeGraphWatch('themeFocus', () => {
      const c = cy
      if (!c) return
      if (c.nodes(':selected').length > 0) return
      applyGraphSelectionDimFromThemeIds(c, ids)
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
          scheduleRedraw()
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
      subject.episodeId,
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
    /* graph-v3 tier 8-5 — Shift+E toggles load-mode (Top-down ↔ Everything).
     * `key === 'E'` because Shift is held; guard against Alt/Ctrl/Meta so
     * this doesn't collide with browser / OS shortcuts. Ignore when focus
     * is in an input to preserve capital E typing. */
    if (
      e.key === 'E' &&
      e.shiftKey &&
      !e.altKey &&
      !e.ctrlKey &&
      !e.metaKey &&
      !(
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        (e.target instanceof HTMLElement && e.target.isContentEditable)
      )
    ) {
      e.preventDefault()
      loadMode.toggleMode()
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
        syncGraphNodeVisibilityTierClasses(c)
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
        // UX fix: when the user tabbed away mid-handoff, the FSM is left
        // in an in-flight state (``loading_fetch`` / ``loading_merge`` /
        // ``redrawing_*``) with a pending envelope. ``tryApplyPendingFocus``
        // only covers the ``nav.pendingFocusNodeId`` case (set by Search /
        // Digest paths); Library L1 doesn't set it, so the FSM would
        // stuck-timeout 15 s after the user returned. Drive forward from
        // the FSM's own pending envelope so the user sees the episode
        // open, not an empty canvas + error strip.
        if (!applied && graphHandoff.pending) {
          tryApplyPendingFsmEnvelopeFromTabReturn(c)
        }
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
    <GraphFilterBar
      :type-histogram-counts="typeHistogramCounts"
      :degree-histogram-counts="degreeHistogramCounts"
    />

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
        <!-- RFC-080 V3 — timeline lens missing-date marker. Surfaces the
             count of episodes parked at the leftmost spot so users see
             the "no date" pile isn't a layout glitch. -->
        <div
          v-if="preferredLayout === 'timeline' && timelineMissingDateCount > 0"
          data-testid="graph-timeline-missing-date"
          class="pointer-events-none absolute bottom-2 left-2 z-[4] rounded border border-border/70 bg-surface/95 px-2 py-0.5 text-[10px] text-muted shadow-sm"
        >
          {{ timelineMissingDateCount }}
          {{ timelineMissingDateCount === 1 ? 'episode has' : 'episodes have' }}
          no date (parked at left)
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
        <!-- graph-v3 Tier 5A-1 — theme-cluster legend, opposite corner from minimap. -->
        <GraphThemeLegend />
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

/* graph-v3 A — graph canvas leans darker than the shell in dark theme
   (light theme falls through to --ps-canvas via tokens.css). */
.graph-canvas {
  background-color: var(--ps-graph-canvas);
}
</style>
